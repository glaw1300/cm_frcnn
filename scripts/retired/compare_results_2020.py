import numpy as np
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
import torch
from PIL import Image
from detectron2.layers import batched_nms
from detectron2.structures.boxes import Boxes
from matplotlib.patches import Circle
from datetime import datetime
import tqdm

"""
finding source pixel
"""
def get_plume_info(plumes, tif):# , radius=10):
    # get all unique plumes (those outside of 25px radius of other plumes)
    pxs = []
    pids = []
    psizes = []
    src = "2020"

    # make sure plumes are in range
    w, h = Image.open(tif).size

    # iterate over plumes gorup
    for ind, row in plumes.iterrows():
        # get location
        lat = row["source_lat"]
        long = row["source_lon"]

        # get pixel location
        x, y = utils.coords_to_pix(tif, (lat,long))

        # if outside of pic, ignore
        if x >= w or x < 0 or y >= h or y < 0:
            continue

        # if close to one of the sources, do not add
        add = True
        #for p in pxs:
        #    if dist(p, x, y) < radius:
        #        add = False
        #        break

        if add:
            pxs.append((x,y))
            pids.append(row["plume_id"])
            psizes.append((row["EmissionRate_kgh_HRRR_10m"], row["EmissionUncertainty_kgh_HRRR_10m"]))
            #src = row["source_type"] # assign arbitrary source_type, b/c all same

    # return lat long of all unique plumes
    return pxs, pids, psizes, src


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

def min_dist_between_bb(bb1, bb2):
    """
    x1,y1,x2,y2
    """
    # unpack bbs
    x1, y1, x1b, y1b = bb1
    x2, y2, x2b, y2b = bb2

    # https://stackoverflow.com/questions/4978323/how-to-calculate-distance-between-two-rectangles-context-a-game-in-lua
    left = x2b < x1
    right = x1b < x2
    bottom = y2b < y1
    top = y1b < y2
    if top and left:
        return dist((x1, y1b), x2b, y2)
    elif left and bottom:
        return dist((x1, y1), x2b, y2b)
    elif bottom and right:
        return dist((x1b, y1), x2, y2b)
    elif right and top:
        return dist((x1b, y1b), x2, y2)
    elif left:
        return x1 - x2b
    elif right:
        return x2 - x1b
    elif bottom:
        return y1 - y2b
    elif top:
        return y2 - y1b
    else:# rectangles intersect
        return 0.

def get_distance_tensor(bboxes, p):
    ret = torch.Tensor([])
    # for each box, get min distance to the point
    for box in bboxes:
        d = calc_min_dist_to_pred(box.tolist(), p)
        ret = torch.cat((ret, torch.Tensor([d])))

    return ret

def area(x1, y1, x2, y2):
    return (x2 - x1) * (y2 - y1)

"""
decide which images are best
"""
def variance_of_laplacian(image):
    return cv2.Laplacian(image, cv2.CV_64F).var()

def get_least_blurry_img(tifs):
    max_lapl = float("-inf")
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
def get_prediction_outputs(tifs, cnn, score_thresh = .5, nms_thresh = .25, cent=None):
    # for storing relevant info
    bboxes = torch.Tensor()
    scores = torch.Tensor()
    classes = torch.Tensor()

    for tif in tifs:
        # load in image
        img = cv2.imread(tif)

        # make prediction
        preds = cnn.predict_image(img, view=True, fname=tif, threshold = score_thresh, verbose=True, crop=True, cent=cent)
        inst = preds["instances"]

        # save results
        bboxes = torch.cat((bboxes, inst.pred_boxes.tensor))
        scores = torch.cat((scores, inst.scores))
        classes = torch.cat((classes, inst.pred_classes))

    # run nms across all predictions for all tifs, treating all classes as same
    if len(tifs) > 1:
        keep = batched_nms(bboxes, scores, torch.ones(len(scores)), nms_thresh)
    else:
        keep = torch.ones(len(inst.pred_boxes), dtype=torch.bool)

    # return results
    return bboxes[keep], scores[keep], classes[keep]

def batched_masked_select(mask, *args):
    for a in args:
        yield a[mask]

def determine_source(bboxes, scores, classes, loc, labels):
    # if there is a large compressor, return large compressor
    """
    cind = labels.index("compressor")
    cs = (classes==cind).nonzero().flatten()
    if len(cs) > 0:
        for bb in bboxes[cs]:
            if area(*bb) > 1500:
                return "compressor"
    """
    # remove flares
    find = labels.index("flare")
    nfs = (classes!=find).nonzero().flatten()

    bboxes = bboxes[nfs]
    scores = scores[nfs]
    classes= classes[nfs]

    if len(bboxes) == 0:
        return "flare", 1000., 0.

    dists = get_distance_tensor(bboxes, loc)
    mind = torch.argmin(dists/scores)
    best_src = labels[int(classes[mind].item())], dists[mind].item(), scores[mind].item()

    return best_src
    # recursively iterate over score threshold and radius until get to whole image (~600px radius to get to corner)
    #return determine_source_recurse(bboxes, scores, classes, dists, labels=labels)


def determine_source_recurse(bboxes, scores, classes, dists, radius=100, score=.95,
                             s_decay=.925, r_alpha=1.55, labels=None, c_thresh=900):

    if radius > 600:
        # just take best score if outside radius
        return labels[int(classes[torch.argmax(scores)].item())]

    # mask by score and distance
    mask = torch.where((scores >= score)&(dists <= radius), True, False)

    # if no valid, recurse
    if not any(mask):
        return determine_source_recurse(bboxes, scores, classes, dists,
                                   radius=radius*r_alpha, score=score*s_decay,
                                   s_decay=s_decay, r_alpha=r_alpha, labels=labels)

    bbs, ss, cs, ds = batched_masked_select(mask, bboxes, scores, classes, dists)

    # parse sources in region, best is closest
    ind = torch.argmin(ds)

    # if tank or well or processing, return that
    # if compressor, return compressor unless under threshold, then well
    # if slugcatcher, return next best infrastructure in range or well
    best_src = labels[int(cs[ind].item())]
    return best_src

    # convert label from slugcatcher first
    """
    if best_src == "slugcatcher":
        min_d = float("inf")
        best_ind = ind
        # find closest tensor that is not self, else return well
        for i, bbox in enumerate(bbs):
            # not self
            if i != ind:
                min_d_btw = min_dist_between_bb(bbox.tolist(), bbs[ind].tolist())
                if min_d_btw < min_d:
                    best_ind = i
                    min_d = min_d_btw
        best_src = labels[int(cs[best_ind].item())]

    if best_src == "tank" or best_src == "well":
        return "production" #best_src
    elif best_src in ("pumpjack", "wellhead", "flare", "pond", "slugcatcher"):
        return "production"
    elif best_src == "processing":
        return "processing"
    elif best_src == "compressor":
        bb = bbs[ind]
        bb_a = area(*bb)
        return "compressor" if bb_a >= c_thresh else "production"
    # should be unreachable
    return "NA"
    """

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
        pred, gt, isvalid = row

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
        pred, gt, isvalid = tup
        # ground truth index
        if type(gt) == float or gt == None:
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
                preds = cnn.predict_image(img, view=False, fname=tif, crop=True)
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
def compare_source_to_gt(path_to_tifs="../DIMAC_Context_Images", fast=False, plot=True,
                         plume_list="../GAO_Summer2020_plume_source_list_multimodal.csv",
                         exclude = [],
                         alternates = False, score_thresh=.3,
                         iou_thresh=.35, cnn=None, iters=1000):
    # load in source list
    if plume_list.endswith(".csv"):
        df = pd.read_csv(plume_list)
    else:
        df = pd.read_excel(plume_list)
    df = df[(df["lat"] != -9999) & (df["lon"] != -9999) & (~df["source_id"].isna())]

    plumes = {}

    # load in model
    cnn = cnn or CNN()
    cnn.setup_model()
    cnn_preds = {}

    # keep metrics
    success = 0
    fail = 0

    if alternates:
        with open(alternates, "rb") as f:
            alts = pickle.load(f)

    # load in plume info (so don't have to run gdal every time)
    if os.path.exists("plumes2020.pickle"):
        print("Loading in plumes")
        with open("plumes2020.pickle", "rb") as f:
            plumes = pickle.load(f)
    else:
        print("Locating plumes")
        for name, group in tqdm.tqdm(df.groupby(["source_id"])):
            sid = int(name)
            tifs = glob.glob(os.path.join(path_to_tifs, str(sid)+"_*.tif"))

            if len(tifs) == 0:
                continue

            plumes[sid] = get_plume_info(group, tifs[0])

        with open("plumes2020.pickle", "wb") as f:
            pickle.dump(plumes, f)

    # load in predictions and save before doing attribution
    if os.path.exists("preds2020.pickle"):
        with open("preds2020.pickle", "rb") as f:
            cnn_preds = pickle.load(f)
    else:
        for name, group in df.groupby(["source_id"]):
            # get image info, if are none, exclude
            sid = int(name)

            tifs = glob.glob(os.path.join(path_to_tifs, str(sid)+"_*.tif"))

            if len(tifs) == 0:
                print(f"WARNING: No images for {sid}, continuing")
                #preds[f'{sid}'] = (None, None, None, None, None, gt_src)
                continue

            # if fast, get least blurry
            if fast:
                tifs = [get_least_blurry_img(tifs)]

                # if using alternates, substitute
                if alternates:
                    if alts.get(sid, False):
                        print(f"Substituting image from alternates for {sid}")
                        tifs = [alts[sid]]

            cents, pids, psizes, gt_src = plumes[sid]
            # get plume center of image
            ax = 0
            ay = 0
            for x, y in cents:
                ax += x
                ay += y
            ax /= len(cents)
            ay /= len(cents)

            cnn_preds[name] = get_prediction_outputs(tifs, cnn, score_thresh=score_thresh, nms_thresh=iou_thresh, cent=(ax, ay))
            if len(cnn_preds[name][0]) == 0:
                print(" ".join(tifs))

        # save predictions
        with open("preds2020.pickle", "wb") as f:
            pickle.dump(cnn_preds, f)

    for n in range(iters):
        preds = {}
        noimgct = 0
        noplumect = 0
        excludect = 0
        nopredct = 0
        # iterate over each source (b/c can group many of those plumes)
        for name, group in tqdm.tqdm(df.groupby(["source_id"])):
            # get image info, if are none, exclude
            sid = int(name)

            # from fixed reference point, determine which plumes are of different sources (default 25px radius)
            #start = datetime.now()

            tifs = glob.glob(os.path.join(path_to_tifs, str(sid)+"_*.tif"))
            if len(tifs) == 0:
                noimgct += 1
                #print(f"WARNING: No images for {sid}, continuing")
                #preds[f'{sid}'] = (None, None, None, None, None, gt_src)
                continue

            cents, pids, psizes, gt_src = plumes[sid]

            #print(f"Getting unique plumes: {datetime.now() - start}")

            # convert to production, processing, or compressor
            if gt_src == "tank" or gt_src == "well":
                gt_src = "production"

            if len(cents) == 0:
                noplumect += 1
                #print(f"WARNING: No valid plumes for {sid}, continuing")
                preds[f'{sid}'] = (None, None, None, tifs[0], None, gt_src)#(None, gt_src, None)
                continue

            # if ground truth in exclude, exclude
            if gt_src in exclude:
                #print(f"Excluding {sid}, source {gt_src}")
                excludect += 1
                preds[f'{sid}'] = (None, None, None, tifs[0], None, gt_src)#(None, gt_src, None)
                continue

            # make predictions across tifs (run nms across multiple images)
            bboxes, scores, classes = cnn_preds[name]

            if len(bboxes) == 0: # make sure have some predictions
                nopredct += 1
                #print(f"WARNING: No predictions for source {sid}")
                fail += 1
                preds[f'{sid}'] = (None, None, None, tifs[0], None, gt_src)#(None, gt_src, False)
                continue

            # only need to plot once
            if plot and n == 0:
                tmp = utils.TmpPrediction(bboxes, classes, scores)
                fig, ax = plotter.plot_pred_boxes(cv2.imread(tifs[0]), {"instances":tmp}, cnn.get_labels(), show=False, crop=True)
                ax.set_title(f"{sid} - {gt_src}")

            # for each plume, get the most likely prediction
            start = datetime.now()
            dups = set()
            for i, ((x,y), pid, psize) in enumerate(zip(cents, pids, psizes)):
                # add noise to x and y
                x += np.random.normal(scale=30)
                y += np.random.normal(scale=30)

                pred_src, d, s = determine_source(bboxes, scores, classes, (x, y), cnn.get_labels())

                # print current stats and store prediction
                store = (x, y, pid, tifs[0], pred_src, gt_src, d, s)
                preds[f"{sid}_{i}_{n}"] = store
                #print(f"{sid}_{i}_{n}", store)

                if plot and n == 0 and pred_src:
                    c = plotter.COLORS[cnn.get_labels().index(pred_src)]
                    ax.add_patch(Circle((x,y), max(psize[0]/20., 10.), fc=c, ec="black"))
                    #ax.errorbar(x-psize[0]/200.,y, xerr=psize[1]/200., ecolor="black", capsize=2)
            #print(f"Determining sources iteration: {datetime.now() - start}")
            if plot and n == 0:
                fig.savefig(f"preds_w_src_low/{sid}.png", dpi=100)
                plt.close(fig)
                """
                if gt_src == pred_src:
                    success += 1
                    preds[f"{sid}_{i}"] = (pred_src, gt_src, True)
                    print(f"{sid}: TRUE  -- model predicted {pred_src}, gt is {gt_src}")
                else:
                    fail += 1
                    preds[f"{sid}_{i}"] = (pred_src, gt_src, False)
                    print(f"{sid}: FALSE -- model predicted {pred_src}, gt is {gt_src}")

                    # if wrong prediction and plot, visualize
                    if plot:
                        tmp = utils.TmpPrediction(bboxes, classes, scores)
                        fig, ax = plotter.plot_pred_boxes(cv2.imread(tifs[0]), {"instances":tmp}, cnn.get_labels(), block=False)
                        ax.set_title(f"{sid}_{i} - pred:{pred_src}, gt:{gt_src}")
                        ax.add_patch(Circle((x,y), 5, fc="red", ec="red"))
                        fig.savefig(f"wrong_preds/{sid}_{i}.png", dpi=100)

                print(f"Current score: {success} true, {fail} false, accuracy: {(success)/(success+fail)}")
                """

        # show stats: no images, excluded, no predictions, no plumes, or success
        print(f"{n:6d} results:")
        print(f"{len(preds):6d} plumes predicted")
        print(f"{noimgct:6d} no images")
        print(f"{nopredct:6d} no predictions")
        print(f"{noplumect:6d} no plumes")
        print(f"{excludect:6d} excluded")

        # save current info in pickle
        with open(f"compares_low/compare_{n}.pickle", "wb") as f:
            pickle.dump(preds, f)

    # calculate confusion matrix and, if plot, visualize
    ticks = []
    # put excluded labels at end
    for l in list(df["source_type"].unique()):
        # 3 label segmentation again
        if l == "tank":
            continue
        elif l == "well":
            l = "production"

        # ordering
        if l in exclude:
            ticks.append(l)
        else:
            ticks.insert(0, l)

    cm = calc_comp_confusion_matrix_from_data(preds, ticks)
    utils.pprint_cm(cm, ticks + ["missed", "excluded"], ticks)
    if plot:
        plot_confusion_matrix(cm, ticks, ticks + ["missed", "excluded"])
        barplot_scores(parse_results_by_label(preds))

if __name__=="__main__":
    compare_source_to_gt(cnn= CNN(), plume_list="../GAO_Summer2020_plume_source_list_multimodal.csv")
