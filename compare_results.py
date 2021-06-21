import numpy as np
from osgeo import osr, ogr, gdal
from faster_rcnn import CNN
import cv2
from itertools import groupby
import pandas as pd
import glob
import os
import pickle
import plotter
import matplotlib.pyplot as plt
import utils

"""
finding source pixel
"""
# algorithm for calculating pixel location on image from:
# https://stackoverflow.com/questions/58623254/find-pixel-coordinates-from-lat-long-point-in-geotiff-using-python-and-gdal
def world_to_pixel(geo_matrix, x, y):
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

def calc_coords_to_pix(path_to_tif, coords):
    """
    Given a file (.tif) and coordinate pair (lat, long), calculate pixel location of coord
    """
    ds = gdal.Open(path_to_tif)
    target = osr.SpatialReference(wkt=ds.GetProjection())

    source = osr.SpatialReference()
    source.ImportFromEPSG(4326)

    transform = osr.CoordinateTransformation(source, target)

    point = ogr.Geometry(ogr.wkbPoint)

    point.AddPoint(*coords)
    point.Transform(transform)

    return world_to_pixel(ds.GetGeoTransform(), point.GetX(), point.GetY())

"""
calculating closest box
"""
def dist(p, x, y):
    return np.sqrt((p[0] - x)**2 + (p[1] - y)**2)

def dist2(x1, y1, x2, y2):
    return (x1 - x2)**2 + (y1 - y2)**2

def within(p, x1, y1, x2, y2):
    return p[0] >= x1 and p[0] <= x2 and p[1] >= y1 and p[1] <= y2

def dist_to_side(p, x1, y1, x2, y2, l):
    """
    Given the bounding boxes are of form xywh, pass the x and y coords of a given
    side as well as a tuple (x, y) that is the point

    should calculate if the point lies within a bounding box in a separate method
    """
    # algorithm from: https://stackoverflow.com/questions/849211/shortest-distance-between-a-point-and-a-line-segment
    # calculate shortest distance of point to line segment
    t = ((p[0] - x1) * (x2 - x1) + (p[1] - y1) * (y2 - y1)) / l**2
    t = max(0, min(1, t))
    return dist(p, x1 + t * (x2 - x1), y1 + t * (y2 - y1))

def calc_min_dist_to_pred(pred_box, p):
    """
    pred_box: 4 item list of form xywh
    p: point to calculate dist to (x, y)
    """
    x1, y1, x2, y2 = pred_box

    # if point is within box, return 0
    if within(p, x1, y1, x2, y2):
        return 0

    # otherwise, find min distance to a side
    l = dist_to_side(p, x1, y1, x1, y2, y2-y1)
    r = dist_to_side(p, x2, y1, x2, y2, y2-y1)
    t = dist_to_side(p, x1, y2, x2, y2, x2-x1)
    b = dist_to_side(p, x1, y1, x2, y1, x2-x1)

    return min(l, r, t, b)

"""
decide which images are best
"""
def variance_of_laplacian(image):
    return cv2.Laplacian(image, cv2.CV_64F).var()

def get_least_blurry_img(tifs):
    max_lapl = 0
    best_img = ""
    for tif in tifs:
        vol = variance_of_laplacian(cv2.imread(tif))
        if vol > max_lapl:
            best_img = tif
            max_lapl = vol
    return best_img

"""
predict from images and determine source
"""
def predict_source_freq_from_tifs(tifs, cnn, coords):
    """
    Iterate over all tifs, get closest distance source, compare to others
    """
    # get all predicted sources
    pred_srcs = {}
    # keep the most predictions as best image
    best_img = ""
    max_npreds = 0
    # iterate over tifs
    for tif in tifs:
        # get img center
        x, y = calc_coords_to_pix(tif, coords)

        # load image
        img = cv2.imread(tif)

        # if out of picture, disregard
        if x < 0 or x > len(img[0]) or y <= 0 or y >= len(img):
            continue

        # predict
        preds = cnn.predict_image(img, view=False, fname=tif)
        inst = preds["instances"]
        # if no predictions, move on
        npreds = len(inst.pred_boxes)
        if npreds == 0:
            print(f"No predictions for: {tif}")
            continue
        elif npreds > max_npreds:
            best_img = tif
            max_npreds = npreds

        # compute min distance and get min prediction category
        min_dist = float("inf")
        min_src = None
        for box, pred_class in zip(inst.pred_boxes, inst.pred_classes):
            box_dist = calc_min_dist_to_pred(box.tolist(), (x, y))
            # update min distance
            if box_dist < min_dist:
                min_dist = min(min_dist, box_dist)
                min_src = int(pred_class)
        # increment frequency of source
        if min_src not in pred_srcs:
            pred_srcs[min_src] = 0
        pred_srcs[min_src] += 1

    return pred_srcs, best_img

def pred_to_label(id, id_to_label):
    """
    Possible predictions:
    well, pipeline, compressor, tank, processing, NA

    let wellhead and pumpjack and slugcatcher == well
    let compressor == compressor
    let tank == tank
    """
    cnn_label = id_to_label[id]

    if cnn_label == "wellhead" or cnn_label == "pumpjack" or cnn_label == "slugcatcher" or cnn_label == "pond" or cnn_label == "flare":
        return "well"
    elif cnn_label == "compressor":
        return "compressor"
    elif cnn_label == "tank":
        return "tank"
    # unreachable
    return "NA"

def parse_pred_sources(srcs, id_to_label, gt):
    # if have more than one "closest" source, choose the one that is gt, flag
    max_freq = max(srcs.values())
    maxs = []
    # get max closest sources
    for k, v in srcs.items():
        if v == max_freq:
            maxs.append(k)

    if len(maxs) > 1:
        print("Tie between:", maxs)
        # if one is prediction, choose
        for m in maxs:
            if pred_to_label(maxs[0], id_to_label) == gt:
                return gt
        # otherwise, take first
        return pred_to_label(maxs[0], id_to_label)
    else:
        return pred_to_label(maxs[0], id_to_label)

"""
results parsers/plotters
"""
def load_data(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def parse_results_by_label(data, load=False):
    """
    row form: pred, gt, file, is valid

    if is valid == None, either no images or was excluded
    if is valid == False and pred == None, no predictions
    otherwise, is valid determines whether good prediction or not
    """
    # if load, data is path to pickle and load it in
    if load:
        data = load_data(data)

    # keep track of data for each label, ignore, and no preds
    scores = {"exclude": 0, "nopreds":0}

    # iterate over data in row
    for src, row in data.items():
        # unpack row
        pred, gt, f, isvalid = row

        # refer to parsing logic in method description
        if isvalid == None:
            scores["exclude"] += 1
        elif isvalid == False and pred == None:
            scores["nopreds"] += 1
        else:
            # add key if not in scores
            # form [0, 0] is [success, fails]
            if pred not in scores:
                scores[pred] = [0, 0]
            if isvalid:
                scores[pred][0] += 1
            else:
                scores[pred][1] += 1

    return scores

def plot_confusion_matrix(cm, row, col):
    fig, ax = plt.subplots()
    im, cbar = plotter.heatmap(cm, row, col, ax=ax, cmap="YlOrRd", title=f"Confusion matrix of predictions vs truth - sources", ylabel="Ground Truth", xlabel="Model prediction")
    texts = plotter.annotate_heatmap(im, valfmt="{x:.0f}")
    plt.show(block=False)

def barplot_scores(scores):
    # parse each score, store label and percentages
    labels = []
    falsepct = []
    fillers = []
    annots = [] # annotations of form ("annotation", x, y, va)
    idx = 0
    for label, score in scores.items():
        labels.append(label)
        # if score is list, calc percentage, otherwise, add filler
        # annotation for filler is number
        if type(score) == int:
            fillers.append(1)
            falsepct.append(0)
            annots.append((score, idx, .5, "center")) # in middle
        # annotation for percentage is true pct
        else:
            fillers.append(0)
            falsepct.append(score[1]/sum(score))
            annots.append(("%.3f"%(score[0]/sum(score)), idx, score[1]/sum(score), "bottom"))
        # step along bars for annots
        idx +=1

    # plot each bar
    fig, ax = plt.subplots()
    # true predictions
    ax.bar(labels, [1]*len(labels), label="Correct", color="g")
    # false above true
    ax.bar(labels, falsepct, label="False", color="r")
    # fill in non-percentages black
    ax.bar(labels, fillers, label="Non-percentages", color="grey")

    # annotate each bar
    for a in annots:
        plt.annotate(a[0], xy=(a[1], a[2]), ha="center", va=a[3])

    plt.title("Ground truth sources to detected")
    plt.xlabel("Source")
    plt.ylabel("Frequency (/Number)")

    plt.legend()
    plt.show(block=False)


def calc_comp_confusion_matrix_from_data(data, pred_labels, load=False):
    # load data
    if load:
        data = load_data(data)

    # 2 more columns for excluded and no prediction
    cm = np.zeros((len(pred_labels), len(pred_labels) + 2))
    # parse data
    for key, tup in data.items():
        pred, gt, f, isvalid = tup
        # ground truth index
        if type(gt) == float:
            gt = "NA"
        gtind = pred_labels.index(gt)

        # excluded
        if isvalid == None:
            cm[gtind, -1] += 1
        # missed
        elif isvalid == False and pred==None:
            cm[gtind, -2] += 1
        else:
            pind = pred_labels.index(pred)
            cm[gtind, pind] += 1

    return cm

def explore_nopreds(data, load=False, cnn=None, path_to_tifs="../tifs"):
    if load:
        data = load_data(data)

    if cnn == None:
        cnn = CNN()
        cnn.setup_model()

    # keep track of results
    res = {}
    missed = 0
    # no preds are when img exists but pred is none
    for src, row in data.items():
        pred, gt, f, isvalid = row

        if isvalid == False and pred is None and f is not None:
            print(f"Attempting {src}")
            haspred = False
            # get all images and see if can get prediction
            tifs = glob.glob(os.path.join(path_to_tifs, str(src)+"_*.tif"))

            for tif in tifs:
                # load image
                img = cv2.imread(tif)

                # predict
                preds = cnn.predict_image(img, view=False, fname=tif)
                inst = preds["instances"]
                # if no predictions, move on
                npreds = len(inst.pred_boxes)

                # if has prediction, break and store file
                if npreds > 0:
                    print(f"Found det for {src} - {tif}")
                    haspred = True
                    res[src] = tif
                    break

            # if never had prediction, count
            if not haspred:
                missed += 1

    print(f"Number of sources recovered: {len(res)}")
    print(f"Percentage recovered: {len(res)/(len(res) + missed)}")

    with open("alternates.pickle", "wb") as f:
        pickle.dump(res, f)

    return res

"""
main
"""
def compare_source_to_gt(path_to_tifs="../tifs", fast=True, plot=True,
                         source_list="../permian_source_list_EST_03252021.xlsx",
                         exclude = ["processing", "pipeline", "NA"],
                         alternates = "alternates.pickle"):
    # load in source list
    df = pd.read_excel(source_list)
    df["source_type"] = df["source_type"].fillna("NA")

    # for all the predictions
    preds = {}

    # load in model
    cnn = CNN()
    cnn.setup_model()
    label_to_id = cnn.get_labels()
    id_to_label = {v:k for k, v in label_to_id.items()}

    # keep metrics
    success = 0
    fail = 0

    if alternates:
        with open(alternates, "rb") as f:
            alts = pickle.load(f)

    # iterate over each source
    for ix, row in df.iterrows():
        # extract relevant info for source
        lat = float(row["source_lat"])
        long = float(row["source_lon"])
        sw = int(row["source_id"].lstrip("P"))
        gt_src = row["source_type"] # ground truth source

        # get all tiffs for given source
        tifs = glob.glob(os.path.join(path_to_tifs, str(sw)+"_*.tif"))
        if len(tifs) == 0:
            print(f"WARNING: No images for {sw}, continuing")
            preds[sw] = (None, gt_src, None, None)
            continue

        # if ground truth in exclude, exclude
        if gt_src in exclude:
            print(f"Excluding {sw}, source {gt_src}")
            preds[sw] = (None, gt_src, None, None)
            continue

        # if fast, get least blurry
        if fast:
            tifs = [get_least_blurry_img(tifs)]
            #print(f"FAST MODE: using {tifs[0]}")

            # if using alternates, substitute
            if alternates:
                if alts.get(sw, False):
                    print(f"Substituting image from alternates for {sw}")
                    tifs = [alts[sw]]

        pred_src_freq, save_img = predict_source_freq_from_tifs(tifs, cnn, (lat,long))
        if len(pred_src_freq) == 0: # make sure have some predictions
            print(f"WARNING: No predictions for source {sw}, {save_img}")
            fail += 1
            preds[sw] = (None, gt_src, save_img, False)
            continue

        pred_src = parse_pred_sources(pred_src_freq, id_to_label, gt_src)
        # print current stats and store prediction
        if pred_src == gt_src:
            success += 1
            preds[sw] = (pred_src, gt_src, save_img, True)
            print(f"{sw}: TRUE  -- model predicted {pred_src}, gt is {gt_src}")
        else:
            fail += 1
            preds[sw] = (pred_src, gt_src, save_img, False)
            print(f"{sw}: FALSE -- model predicted {pred_src}, gt is {gt_src}")

        print(f"Current score: {success} true, {fail} false, accuracy: {(success)/(success+fail)}")

    # save current info in pickle
    with open("compare.pickle", "wb") as f:
        pickle.dump(preds, f)

    # calculate confusion matrix and, if plot, visualize
    labels = []
    # put excluded labels at end
    for l in list(df["source_type"].unique()):
        if l in exclude:
            labels.append(l)
        else:
            labels.insert(0, l)
    cm = calc_comp_confusion_matrix_from_data(preds, labels)
    utils.pprint_cm(cm, labels + ["missed", "excluded"], labels)
    if plot:
        plot_confusion_matrix(cm, labels, labels + ["missed", "excluded"])
        barplot_scores(parse_results_by_label(preds))

if __name__=="__main__":
    compare_source_to_gt()
