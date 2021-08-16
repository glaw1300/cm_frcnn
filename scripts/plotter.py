import json
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import StrMethodFormatter
from matplotlib import patches
from matplotlib import cm
from utils import load_json_arr
import os

COLORS = ['blue', 'orange', 'green', 'yellow', 'red', 'purple', 'pink', 'cyan', 'brown']

# from https://matplotlib.org/3.1.1/users/event_handling.html#draggable-rectangle-exercise
# merge draggable rectangle with arrow key shifts
class DraggableRectangle:
    def __init__(self, rect, fig):
        self.rect = rect
        self.press = None
        self.fig = fig

    def connect(self):
        'connect to all the events we need'
        self.cidpress = self.fig.canvas.mpl_connect(
            'button_press_event', self.on_press)
        self.cidrelease = self.fig.canvas.mpl_connect(
            'button_release_event', self.on_release)
        self.cidmotion = self.fig.canvas.mpl_connect(
            'motion_notify_event', self.on_motion)

    def on_press(self, event):
        'on button press we will see if the mouse is over us and store some data'
        if event.inaxes != self.rect.axes: return

        contains, attrd = self.rect.contains(event)
        if not contains: return
        print('event contains', self.rect.xy)
        x0, y0 = self.rect.xy
        self.press = x0, y0, event.xdata, event.ydata

    def on_motion(self, event):
        'on motion we will move the rect if the mouse is over us'
        if self.press is None: return
        if event.inaxes != self.rect.axes: return
        x0, y0, xpress, ypress = self.press
        dx = event.xdata - xpress
        dy = event.ydata - ypress
        #print('x0=%f, xpress=%f, event.xdata=%f, dx=%f, x0+dx=%f' %
        #      (x0, xpress, event.xdata, dx, x0+dx))
        self.rect.set_x(x0+dx)
        self.rect.set_y(y0+dy)

        self.fig.canvas.draw()


    def on_release(self, event):
        'on release we reset the press data'
        self.press = None
        self.fig.canvas.draw()

    def disconnect(self):
        'disconnect all the stored connection ids'
        self.fig.canvas.mpl_disconnect(self.cidpress)
        self.fig.canvas.mpl_disconnect(self.cidrelease)
        self.fig.canvas.mpl_disconnect(self.cidmotion)

def plot_pred_boxes(img, preds, labels, colors = COLORS, block=True, show=True, crop=False, cent = None):
    """
    preds: x1, y1, x2, y2
    """
    assert len(colors) >= len(labels), "Not enough colors to map to labels"

    # if crop, take 864 x 864 center
    if crop:
        if cent:
            x, y = cent
        else:
            y, x, _ = img.shape
            x /= 2
            y /= 2
        cut = 864
        img = img[int(y - cut / 2) : int(y + cut/2), int(x - cut/2 ): int(x+cut/2), :]

    # unpack predictions
    inst = preds["instances"]
    bboxes = inst.pred_boxes
    classes = inst.pred_classes
    scores = inst.scores

    # setup figure and plot image
    fig, ax = plt.subplots()
    ax.imshow(img[:, :, ::-1]) # cv2 loads in images in bgr

    # iterate over boxes and add rectangles
    labeled = set()
    for b, c, s in zip(bboxes, classes, scores):
        cind = int(c.item())
        x, y = b[0].item(), b[1].item()
        w, h = b[2].item() - x, b[3].item() - y
        l = labels[cind]

        ax.add_patch(patches.Rectangle((x,y),w,h, edgecolor=colors[cind], facecolor="none", label="" if l in labeled else l))
        ax.annotate("%.2f %s" %(s.item(), l), xy=(x,y), ha="left", va="bottom",
                    color='white', fontsize=5,
                    bbox=dict(boxstyle='square,pad=0', fc=colors[cind], ec='none', alpha=.5))

        labeled.add(l)

    # legend
    plt.legend()
    plt.axis('off')
    if show:
        plt.show(block=block)
    return fig, ax


def plot_val_with_total_loss(output_dir):
    metrics = load_json_arr(os.path.join(output_dir, "metrics.json"))
    tot = []
    val = []
    iters = []
    for line in metrics:
        if "validation_loss" in line and "total_loss" in line and "iteration" in line:
            tot.append(line["total_loss"])
            val.append(line["validation_loss"])
            iters.append(line["iteration"])

    plt.figure(999)
    plt.clf()

    plt.plot(iters, tot, "r-", label="total loss")
    plt.plot(iters, val, "b-", label="val loss")
    plt.title(f"Total and Validation Loss epochs {min(iters)} to {max(iters)}")
    plt.legend()

    plt.savefig(os.path.join(output_dir, "val_v_tot_loss.png"), dpi=100)

    plt.show(block=False)

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
    #ax.spines.set_visible(False)


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
