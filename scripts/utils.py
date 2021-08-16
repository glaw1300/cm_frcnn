from functools import wraps
import inspect
import numpy as np
import json
import os
import glob
import tqdm
import matplotlib.pyplot as plt
import pandas as pd
from decimal import Decimal
# dataset loading
import logging
from PIL import Image
from detectron2.structures import BoxMode
import copy

try:
    GDAL_LOADED = True
    from osgeo import osr, ogr, gdal
except:
    GDAL_LOADED = False
    print("Failed to import GDAL")
from dataclasses import dataclass
from torch import Tensor

# algorithm for calculating pixel location on image from:
# https://stackoverflow.com/questions/58623254/find-pixel-coordinates-from-lat-long-point-in-geotiff-using-python-and-gdal
def _world_to_pixel(geo_matrix, x, y):
    """
    Uses a gdal geomatrix (gdal.GetGeoTransform()) to calculate
    the pixel location of a geospatial coordinate
    """
    ul_x= geo_matrix[0]
    ul_y = geo_matrix[3]
    x_dist = geo_matrix[1]
    y_dist = geo_matrix[5]
    pixel = int((x - ul_x) / x_dist)
    line = -int((ul_y - y) / y_dist)
    return pixel, line

def coords_to_pix(path_to_tif, coords):
    """
    Given a file (.tif) and coordinate pair (lat, long), calculate pixel location of coord
    """
    assert GDAL_LOADED, "GDAL not loaded"

    ds = gdal.Open(path_to_tif)
    target = osr.SpatialReference(wkt=ds.GetProjection())

    source = osr.SpatialReference()
    source.ImportFromEPSG(4326)

    transform = osr.CoordinateTransformation(source, target)

    point = ogr.Geometry(ogr.wkbPoint)

    point.AddPoint(*coords[::-1])
    point.Transform(transform)

    return _world_to_pixel(ds.GetGeoTransform(), point.GetX(), point.GetY())

def pix_to_coords(path_to_tif, pxs):
    assert GDAL_LOADED, "GDAL not loaded"

    x, y = pxs

    ds = gdal.Open(path_to_tif)
    xoffset, px_w, rot1, yoffset, rot2, px_h = ds.GetGeoTransform()

    # supposing x and y are your pixel coordinate this
    # is how to get the coordinate in space.
    posX = px_w * x + rot1 * y + xoffset
    posY = rot2 * x + px_h * y + yoffset

    # shift to the center of the pixel
    posX += px_w / 2.0
    posY += px_h / 2.0

    # get CRS from dataset
    crs = osr.SpatialReference()
    crs.ImportFromWkt(ds.GetProjectionRef())
    # create lat/long crs with WGS84 datum
    crsGeo = osr.SpatialReference()
    crsGeo.ImportFromEPSG(4326) # 4326 is the EPSG id of lat/long crs
    t = osr.CoordinateTransformation(crs, crsGeo)
    long, lat, z = t.TransformPoint(posX, posY)

    return lat, long

def map_annots_to_image(ogtif, newtif, annots, tifpath="../tifs"):
    # get path to tifs
    ogpath = os.path.join(tifpath, ogtif)
    newpath = os.path.join(tifpath, newtif)

    # iterate over each annotation
    newa = []
    for a in annots:
        bbox = a["bbox"]
        x1, y1 = bbox[0], bbox[1]
        x2, y2 = x1 + bbox[2], y1 + bbox[3]
        # map three corners of box to lat/long
        tlc = pix_to_coords(ogpath, (x1, y1))
        trc = pix_to_coords(ogpath, (x2, y1))
        blc = pix_to_coords(ogpath, (x1, y2))

        # map those coords onto the new image to get width and height
        x1n, y1n = coords_to_pix(newpath, tlc)
        x2n, _ = coords_to_pix(newpath, trc)
        _, y2n = coords_to_pix(newpath, blc)

        # with top left as new fixed point, compute width and height and add annot
        newa.append({"category_id":a["category_id"], "bbox_mode":a["bbox_mode"],
                     "bbox":[x1n, y1n, x2n-x1n, y2n-y1n]})

    return newa


def load_dimac_dataset(csvs, exclude, img_dir, labels, excel_path):
    # load in boxes from csv and add to img list
    srcs = {}

    # read in each annotation in each csv
    # CSV FORMAT: label,xmin,ymin,box_width,box_height,img_name,img_width,img_height
    for csv in csvs:
        with open(csv, "r") as f:
            line = f.readline().strip("\n")
            # for each box
            while line:

                vals = line.split(",")
                # omit well pads and processing for now
                if vals[0] in exclude:
                    line = f.readline().strip("\n")
                    continue

                # box parameters
                xmin = float(vals[1])
                ymin = float(vals[2])
                xw = float(vals[3])
                yh = float(vals[4])

                # get source id, group each image by source id
                src_id = int(vals[5].split("_")[0])
                if src_id not in srcs:
                    srcs[src_id] = {}

                # if the image is not already added to the source dict, add it
                # even tho we reference tmp, it will update srcs (just for conciseness)
                tmp = srcs[src_id]
                img_name = vals[5]
                if img_name not in tmp:
                    tmp[img_name] = {} # add image info to tmp
                    # current path to files
                    tmp[img_name]["file_name"] = os.path.join(img_dir, img_name)
                    # source_id for later addition of unannotated files
                    tmp[img_name]["src_id"] = src_id
                    # mismatch in image sizes in annotatoin, just load in image :(
                    with Image.open(os.path.join(img_path, img_name)) as i:
                        img_w, img_h = i.size

                    tmp[img_name]["width"] = img_w#min(864, int(vals[6]))
                    tmp[img_name]["height"] = img_h#min(864, int(vals[7]))
                    # image id can just be image name
                    tmp[img_name]["image_id"] = img_name
                    tmp[img_name]["annotations"] = []

                # add bounding box
                tmp[img_name]["annotations"].append({"bbox":[xmin, ymin, xw, yh],
                                                   "bbox_mode": BoxMode.XYWH_ABS,
                                                   "category_id": labels[vals[0]]})

                # next line
                line = f.readline().strip("\n")

    # for each row in the excel sheet (of form file name, use), confirm which images have
    # annotations and fill in those that don't by selecting another image of same source id
    use_files = pd.read_excel(excel_path, usecols = [1,2], names=["file", "use"])
    fnames = use_files[use_files["use"] == "x"]["file"].values
    final = []
    for fname in fnames:
        # extract source id from fname (format ./srcid_date_number.tif.jpeg)
        src_id = int(fname.split("_")[0].lstrip("./"))

        # check if annotation already exists, add it to final, otherwise take another annotation from same source
        # if no annotated files, ignore
        if src_id not in srcs:
            continue

        img_info = srcs[src_id].get(fname.lstrip("./"), None)

        if not img_info:
            #create new entry and add to final
            key, new_img_info = dict(srcs[src_id]).popitem()
            # duplicate dictionary for modifictation
            new_img_info = dict(new_img_info)
            #print(f"building new image for {fname} based off of {key}")
            # keep annotations, change everything else
            new_img_info["file_name"] = os.path.join(img_dir, fname.lstrip("./"))
            # mismatch in image sizes in annotatoin, just load in image :(
            with Image.open(new_img_info["file_name"]) as i:
                img_w, img_h = i.size

            new_img_info["width"] = img_w#min(864, int(vals[6]))
            new_img_info["height"] = img_h#min(864, int(vals[7]))
            # image id can just be image name
            new_img_info["image_id"] = fname.lstrip("./")
            # add to final
            #print(f"adding {new_img_info} to final")
            final.append(new_img_info)
        else:
            final.append(img_info)

    return np.array(final)


def initializer(func):
    """
    Automatically assigns the parameters.

    >>> class process:
    ...     @initializer
    ...     def __init__(self, cmd, reachable=False, user='root'):
    ...         pass
    >>> p = process('halt', True)
    >>> p.cmd, p.reachable, p.user
    ('halt', True, 'root')
    """
    insp = inspect.getfullargspec(func)

    @wraps(func)
    def wrapper(self, *args, **kargs):
        for name, arg in list(zip(insp.args[1:], args)) + list(kargs.items()):
            setattr(self, name, arg)

        for name, default in zip(reversed(insp.args), reversed(insp.defaults)):
            if not hasattr(self, name):
                setattr(self, name, default)

        func(self, *args, **kargs)

    return wrapper

def load_json_arr(json_path):
    lines = []
    with open(json_path, 'r') as f:
        for line in f:
            lines.append(json.loads(line))
    return lines

"""
Confusion matrix utils
"""
def pprint_cm(cm, row_labels, col_labels):
    matrix = list(map(list, list(cm)))
    matrix.insert(0, [" "] + col_labels)

    for row, label in zip(matrix[1:], row_labels):
        row.insert(0, label)

    s = [[str(e) for e in row] for row in matrix]
    lens = [max(map(len, col)) for col in zip(*s)]
    fmt = '\t'.join('{{:{}}}'.format(x) for x in lens)
    table = [fmt.format(*row) for row in s]
    print('\n'.join(table))

def get_iou(pred_box, gt_box):
    """
    pred_box: x1, y1, x2, y2
    gt_box: x1, y1, w, h
    """
    # intersection box
    x1 = max(pred_box[0], gt_box[0])
    y1 = max(pred_box[1], gt_box[1])
    x2 = min(pred_box[2], gt_box[0]+gt_box[2])
    y2 = min(pred_box[3], gt_box[1]+gt_box[3])

    # area of intersection (return 0 for none)
    w = x2 - x1
    h = y2 - y1

    if w <= 0 or h <= 0:
        return 0.
    i = w*h

    # area of union
    p = (pred_box[2] - pred_box[0]) * (pred_box[3]-pred_box[1])
    gt = gt_box[2] * gt_box[3]

    u = p + gt - i # subtract for double counting

    return i / (u + 1e-7) # avoid division by 0

def score_prediction(prediction, gt, nlabels=4, iou_threshold = .5):
    """
    prediction: output of Detectron2 Predictor
        -see https://detectron2.readthedocs.io/en/latest/tutorials/models.html#model-output-format
    gt: one entry from DatasetCatalog
        -see https://detectron2.readthedocs.io/en/latest/tutorials/datasets.html#use-custom-datasets
    """
    # confusion matrix structure: each index matches to label id + 1 for background
    # horizontal rows: target values (ground truth) -> last row is detected background
    # vertical columns: what the model actually predicted -> last column is missed detection
    bg = nlabels # index of background
    cm = np.zeros((bg+1, bg+1)) # confusion matrx

    # all of the ious
    inst = prediction["instances"]
    ious = np.zeros((len(gt["annotations"]), len(inst.pred_boxes)))

    # for each annotation find all ious compared to all predictions
    for a_ind, annot in enumerate(gt["annotations"]):
        #
        for p_ind, (pred_box, pred_class) in enumerate(zip(inst.pred_boxes, inst.pred_classes)):
            pred_box = pred_box.tolist()
            pred_class = int(pred_class)

            # get iou of annotation and prediction
            iou = get_iou(pred_box, annot["bbox"])

            # if iou exceeds threshold, record
            if iou >= iou_threshold:
                ious[a_ind, p_ind] = iou

    # for each annotation, fill in confusion matrix with max iou (or indicate miss if all 0)
    for a_ind in range(len(gt["annotations"])):
        # if every value is less than threshold, missed detection
        if np.all(ious[a_ind] < iou_threshold):
            cm[gt["annotations"][a_ind]["category_id"], bg] += 1
        # take best guess based on prediction confidence
        else:
            # for all legitimate guesses, get prediction confidence
            max_score = 0
            max_ind = None
            scores = prediction["instances"].scores
            # all guesses exceeding threshold
            for ind in np.nonzero(ious[a_ind] >= iou_threshold)[0]:
                # if score greater than max, take that ind
                if scores[ind] >= max_score:
                    max_ind = ind
                    max_score = scores[ind]
            # increment highest scoring category
            cm[gt["annotations"][a_ind]["category_id"], int(inst.pred_classes[max_ind])] += 1

    # for each detection, if it has not already been counted, count in cm
    for p_ind, p_class in enumerate(prediction["instances"].pred_classes):
        # annotation this class corresponds to
        a_ind = np.argmax(ious[:, p_ind])

        # if not an overlapping detection, record as labelling background
        if np.sum(ious[:, p_ind]) == 0:
            cm[bg, int(p_class)] += 1

        # otherwise, it is will have been selected if it was the best label for the annotation

    return cm

def load_dataset_from_csvs(self, excel_path="data/annotations/fnames.xlsx"):
    """
    must return list of dictionaries, each dict is item in dataset
    - instances will be formated for instance detection
    https://detectron2.readthedocs.io/en/latest/tutorials/datasets.html#use-custom-datasets
    file_name
    height
    width
    image_id
    annotations: list[dict]
        - bbox list[float] (list of 4 numbers representing bounding box)
        - bbox_mode int (structures.BoxMode)
        - category_id
    """
    # load in boxes from csv and add to img list
    srcs = {}

    # labels must be loaded to load dataset
    if len(self._labels) == 0:
        logging.log(logging.WARNING, "Labels had not been loaded, loading now")
        self.load_labels()

    # if has json, load data from json
    if self.has_json:
        logging.log(logging.INFO, "Loading data from JSON")
        self._load_json(["data"])
        return

    # path to csvs
    csvs = glob.glob(os.path.join(self.csv_path, "*.csv"))
    # read in each annotation in each csv
    # CSV FORMAT: label,xmin,ymin,box_width,box_height,img_name,img_width,img_height
    for csv in csvs:
        with open(csv, "r") as f:
            line = f.readline().strip("\n")
            # for each box
            while line:
                vals = line.split(",")
                # omit well pads and processing for now
                if vals[0] in self.exclude:
                    line = f.readline().strip("\n")
                    continue

                img_name = vals[5]
                # if image doesn't exist, move on
                if not os.path.exists(os.path.join(self.img_path, img_name)):
                    line = f.readline().strip("\n")
                    continue

                # box parameters
                xmin = float(vals[1])
                ymin = float(vals[2])
                xw = float(vals[3])
                yh = float(vals[4])

                # get source id, group each image by source id
                src_id = int(vals[5].split("_")[0])
                if src_id not in srcs:
                    srcs[src_id] = {}

                # if the image is not already added to the source dict, add it
                # even tho we reference tmp, it will update srcs (just for conciseness)
                tmp = srcs[src_id]

                if img_name not in tmp:
                    tmp[img_name] = {} # add image info to tmp
                    # current path to files
                    tmp[img_name]["file_name"] = os.path.join(self.img_path, img_name)
                    # source_id for later addition of unannotated files
                    tmp[img_name]["src_id"] = src_id
                    # mismatch in image sizes in annotatoin, just load in image :(
                    with Image.open(os.path.join(self.img_path, img_name)) as i:
                        img_w, img_h = i.size

                    tmp[img_name]["width"] = img_w#min(864, int(vals[6]))
                    tmp[img_name]["height"] = img_h#min(864, int(vals[7]))
                    # image id can just be image name
                    tmp[img_name]["image_id"] = img_name
                    tmp[img_name]["annotations"] = []

                # add bounding box
                tmp[img_name]["annotations"].append({"bbox":[xmin, ymin, xw, yh],
                                                   "bbox_mode": BoxMode.XYWH_ABS,
                                                   "category_id": self._labels.index(vals[0])})

                # next line
                line = f.readline().strip("\n")

    # for each row in the excel sheet (of form file name, use), confirm which images have
    # annotations and fill in those that don't by selecting another image of same source id
    use_files = pd.read_excel(excel_path, usecols = [1,2], names=["file", "use"])
    fnames = use_files[use_files["use"] == "x"]["file"].values
    final = []
    for fname in fnames:
        # extract source id from fname (format ./srcid_date_number.tif.jpeg)
        src_id = int(fname.split("_")[0].lstrip("./"))

        # check if annotation already exists, add it to final, otherwise take another annotation from same source
        # if no annotated files, ignore
        if src_id not in srcs:
            continue

        # get this specific file from the source id
        img_info = srcs[src_id].get(fname.lstrip("./"), None)

        # if it doesn't exist, copy existing entry from srcs and insert
        if not img_info:
            #create new entry and add to final
            key, new_img_info = dict(srcs[src_id]).popitem()
            # duplicate dictionary for modifictation
            new_img_info = copy.deepcopy(new_img_info)
            # map annotations from original image to new
            #new_img_info["annotations"] = map_annots_to_image(new_img_info["file_name"]
            #                .split("/")[-1].strip(".jpeg"), fname.lstrip("./").strip(".jpeg"),
            #                new_img_info["annotations"])
            #print(f"building new image for {fname} based off of {key}")
            # keep annotations, change everything else
            new_img_info["file_name"] = os.path.join(self.img_path, fname.lstrip("./"))
            if not os.path.exists(new_img_info["file_name"]):
                continue
            # mismatch in image sizes in annotatoin, just load in image :(
            with Image.open(new_img_info["file_name"]) as i:
                img_w, img_h = i.size

            new_img_info["width"] = img_w#min(864, int(vals[6]))
            new_img_info["height"] = img_h#min(864, int(vals[7]))
            # image id can just be image name
            new_img_info["image_id"] = fname.lstrip("./")
            # add to final
            #print(f"adding {new_img_info} to final")
            final.append(new_img_info)
        else:
            final.append(img_info)

    if len(final) == 0:
        raise MissingDataError("dataset")

    self._data = np.array(final)

def make_splits_by_src_id(self, splits_map={"train":70, "valid":85, "test":100}):
    self._assert_loaded(self._data, "dataset")

    # iterate over data, group by source id to splits
    for d in self._data:
        if d["src_id"] % 100 <= splits_map["train"]:
            self._splits_data["train"].append(d)
        elif d["src_id"] % 100 <= splits_map["valid"]:
            self._splits_data["valid"].append(d)
        else:
            self._splits_data["test"].append(d)


class MissingDataError(Exception):
    """
    Exception for failure to load data
        src: which data source failed to load
        message: what to sya
    """
    def __init__(self, src, message="Loaded 0 entries for data: "):
        self.message = message + src
        super().__init__(self.message)

@dataclass
class TmpPrediction:
    pred_boxes: Tensor
    pred_classes: Tensor
    scores: Tensor
