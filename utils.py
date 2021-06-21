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

def load_dimac_dataset(csv_glob, exclude, excel_path, img_dir, labels):
    # load in boxes from csv and add to img list
    srcs = {}

    csvs = glob(csv_glob)
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


class MissingDataError(Exception):
    """
    Exception for failure to load data
        src: which data source failed to load
        message: what to sya
    """
    def __init__(self, src, message="Loaded 0 entries for data: "):
        self.message = message + src
        super().__init__(self.message)
