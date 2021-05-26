# methods for getting confusion matrix (and other metrics)
from detectron2.engine import DefaultPredictor
from tqdm import tqdm
import cv2
from detectron2.utils.visualizer import Visualizer
import numpy as np
from matplotlib.ticker import StrMethodFormatter
import matplotlib.pyplot as plt

def get_confusion_matrix(dataset, cfg, threshold=.5, labels={'pumpjack': 0, 'tank': 1, 'compressor': 2, 'flare': 3}, plot=True, hypervisualize=False):

    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = threshold # set a custom testing threshold
    predictor = DefaultPredictor(cfg)

    # PYTHON 3.7 dict keys are in insertion order !!!
    labels = list(labels.keys())

    cm = np.zeros((len(labels)+1, len(labels)+1))

    # iterate over all entries, accumulate score predictions
    for d in tqdm(dataset):
        im = cv2.imread(d["file_name"])
        prediction = predictor(im)
        cm += score_prediction(prediction, d, nlabels=len(labels), iou_threshold=threshold)

        if hypervisualize:
            v = Visualizer(im[:, :, ::-1],
                           scale=0.5
                           )
            # if have predictions
            if len(prediction["instances"]) > 0:
                out = v.draw_instance_predictions(prediction["instances"].to("cpu"))
                cv2.imshow(d["file_name"], out.get_image()[:, :, ::-1])
                cv2.waitKey(0)
                cv2.destroyWindow(d["file_name"])

    # if plot, plot windows
    if plot:
        # plot cm total
        fig, ax = plt.subplots()
        im, cbar = heatmap(cm, labels + ["background"], labels + ["missed"], ax=ax, cmap="YlOrRd", title=f"Confusion matrix total, IOU={threshold}")
        texts = annotate_heatmap(im, valfmt="{x:.0f}")
        # set fig size and save
        fig.set_size_inches(8, 6.5)
        fig.savefig(cfg.OUTPUT_DIR+"/cm.png", dpi=100)

        # plot cm normalized
        fig1, ax1 = plt.subplots()
        im1, cbar1 = heatmap(cm/np.sum(cm, axis=1)[:, np.newaxis], labels + ["background"], labels + ["missed"], ax=ax1, cmap="YlOrRd", title=f"Confusion matrix total, IOU={threshold}")
        texts = annotate_heatmap(im1, valfmt="{x:.2f}")
        # set fig size and save
        fig1.set_size_inches(8, 6.5)
        fig1.savefig(cfg.OUTPUT_DIR+"/cm_normalized.png", dpi=100)
        plt.show()

    # total number of predictions is all items except the last column b/c is missed detection
    return cm, cm[:, :-1].sum()



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
        # missed detection
        if np.sum(ious[a_ind])==0:
            cm[gt["annotations"][a_ind]["category_id"], bg] += 1
        # take best guess
        else:
            cm[gt["annotations"][a_ind]["category_id"], int(inst.pred_classes[np.argmax(ious[a_ind])])] += 1

    # for each detection, if it has not already been counted, count in cm
    for p_ind, p_class in enumerate(prediction["instances"].pred_classes):
        # annotation this class corresponds to
        a_ind = np.argmax(ious[:, p_ind])

        # if not an overlapping detection, record as labelling background
        if np.sum(ious[:, p_ind]) == 0:
            cm[bg, int(p_class)] += 1
        # if the annotation this class corresponds to has this valid detection as its
        # max iou, then we know we already recorded it
        #elif np.sum(ious[a_ind]) > 0  and np.argmax(ious[a_ind]) == p_ind:
        #    continue
        # finally, otherwise, it mislabeled an object
        #else:

        # otherwise, it is will have been selected if it was the best label for the annotation

    return cm

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

# From https://matplotlib.org/stable/gallery/images_contours_and_fields/image_annotated_heatmap.html
def heatmap(data, row_labels, col_labels, ax=None, xlabel="Detections", ylabel="Ground Truth",
            title="Confusion matrix", cbar_kw={}, cbarlabel="", cmap="YlOrRd", **kwargs):
    """
    Create a heatmap from a numpy array and two lists of labels.

    Parameters
    ----------
    data
        A 2D numpy array of shape (N, M).
    row_labels
        A list or array of length N with the labels for the rows.
    col_labels
        A list or array of length M with the labels for the columns.
    ax
        A `matplotlib.axes.Axes` instance to which the heatmap is plotted.  If
        not provided, use current axes or create a new one.  Optional.
    xlabel, ylabel
        x and y axis labels
    title
        title
    cbar_kw
        A dictionary with arguments to `matplotlib.Figure.colorbar`.  Optional.
    cbarlabel
        The label for the colorbar.  Optional.
    cmap
        color map, see https://matplotlib.org/stable/tutorials/colors/colormaps.html
    **kwargs
        All other arguments are forwarded to `imshow`.
    """

    if not ax:
        ax = plt.gca()

    # Plot the heatmap
    im = ax.imshow(data, **kwargs, cmap=cmap)

    # Create colorbar
    cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")

    # We want to show all ticks...
    ax.set_xticks(np.arange(data.shape[1]))
    ax.set_yticks(np.arange(data.shape[0]))
    # ... and label them with the respective list entries.
    ax.set_xticklabels(col_labels)
    ax.set_yticklabels(row_labels)

    # Let the horizontal axes labeling appear on top.
    ax.tick_params(top=True, bottom=False,
                   labeltop=True, labelbottom=False)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=-30, ha="right",
             rotation_mode="anchor")

    # set x and y axis labels
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)

    # Turn spines off and create white grid.
    ax.spines[:].set_visible(False)

    ax.set_xticks(np.arange(data.shape[1]+1)-.5, minor=True)
    ax.set_yticks(np.arange(data.shape[0]+1)-.5, minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)

    return im, cbar


def annotate_heatmap(im, data=None, valfmt="{x:.2f}",
                     textcolors=("black", "white"),
                     threshold=None, **textkw):
    """
    A function to annotate a heatmap.

    Parameters
    ----------
    im
        The AxesImage to be labeled.
    data
        Data used to annotate.  If None, the image's data is used.  Optional.
    valfmt
        The format of the annotations inside the heatmap.  This should either
        use the string format method, e.g. "$ {x:.2f}", or be a
        `matplotlib.ticker.Formatter`.  Optional.
    textcolors
        A pair of colors.  The first is used for values below a threshold,
        the second for those above.  Optional.
    threshold
        Value in data units according to which the colors from textcolors are
        applied.  If None (the default) uses the middle of the colormap as
        separation.  Optional.
    **kwargs
        All other arguments are forwarded to each call to `text` used to create
        the text labels.
    """

    if not isinstance(data, (list, np.ndarray)):
        data = im.get_array()

    # Normalize the threshold to the images color range.
    if threshold is not None:
        threshold = im.norm(threshold)
    else:
        threshold = im.norm(data.max())/2.

    # Set default alignment to center, but allow it to be
    # overwritten by textkw.
    kw = dict(horizontalalignment="center",
              verticalalignment="center")
    kw.update(textkw)

    # Get the formatter in case a string is supplied
    if isinstance(valfmt, str):
        valfmt = StrMethodFormatter(valfmt)

    # Loop over the data and create a `Text` for each "pixel".
    # Change the text's color depending on the data.
    texts = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            kw.update(color=textcolors[int(im.norm(data[i, j]) > threshold)])
            text = im.axes.text(j, i, valfmt(data[i, j], None), **kw)
            texts.append(text)

    return texts
