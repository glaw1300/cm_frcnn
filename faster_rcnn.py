from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import numpy as np
import os, json, random
from glob import glob
import pandas as pd
import cv2
from PIL import Image
from confusion import get_confusion_matrix
import logging
import matplotlib.pyplot as plt
from trainer import Trainer
import argparse
from utils import *
import plotter
from tqdm import tqdm

# import detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.data import MetadataCatalog, DatasetCatalog
import detectron2.data.datasets
from detectron2.structures import BoxMode
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.utils.visualizer import ColorMode
from detectron2.engine import DefaultPredictor


"""
General running information:
- To run this code you need the following files:
    labels.txt - list of all categories for data
    [directory] images - directory of all of your training/test images
    csv(s) - annotation csvs of form label,xmin,ymin,box_width,box_height,img_name,img_width,img_height
    excel spreadsheet - two columns, filename and files to use (denoted with an "x")

To specify the location of these files, simply change the constants below this comment
"""

class CNN():
    LABELS = "labels.txt" # location of labels file (format each line is label)
    EXCLUDE = ["well", "processing"] # make empty list if none
    IMG_PATH = "../DIMAC_jpeg/" # where images are located (WITH trailing /)
    OUTPUT_DIR = "output" # directory where checkpoints should be stored and loaded from

    @initializer # wrapper that unpacks each kwarg into self.<variable name>
    def __init__(self, label_path:str="labels.txt", exclude:list=["well", "processing"], resume:bool=False, config_path:str="cfg.yaml", model_config:str="COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml", splits:dict={"train":.7, "valid":.15, "test":.15}, dataname:str="dimac_", img_path:str="../DIMAC_jpeg/", csv_path:str="exports/csv/", excel_path:str="fnames.xlsx"):
        # config
        self._cfg = None
        # label dict of form label:id
        self._labels = {}
        # data to be filled
        self._data = []
        # segmentations
        self._splits_data = {key:[] for key in splits.keys()}
        # confusion matrix
        self._cm = None

        # to keep track of modules that need to be loaded
        self._unloaded = ["dataset", "labels", "cfg", "splits"]

    """
    getter methods
    """
    def get_dataset(self):
        return self._data

    def get_labels(self):
        return self._labels

    def get_splits(self):
        return self._splits_data

    def get_cfg(self):
        return self._cfg

    def get_confusion_matrix(self):
        if self._cm == None:
            raise MissingDataError(src="confusion matrix", message="Confusion matrix has not been run, call 'run_confusion matrix()' to get: ")
        return self._cm

    """
    load in data methods
    """
    def load_labels(self):
        # open label file
        with open(self.label_path, "r") as f:
            # assign index to each label
            idx = 0
            line = f.readline().strip("\n")
            while line:
                # omit labels
                if line in self.exclude:
                    line = f.readline().strip("\n")
                    continue
                self._labels[line] = idx
                line = f.readline().strip("\n")
                idx += 1
        # if no labels, raise error
        if len(self._labels) == 0:
            raise MissingDataError("labels")

        self._unloaded.remove("labels")


    def load_cfg(self):
        # configure model
        # get default config
        cfg = get_cfg()

        # merge with default config
        # https://github.com/facebookresearch/detectron2/blob/master/MODEL_ZOO.md
        cfg.merge_from_file(model_zoo.get_config_file(self.model_config))

        # merge from default config
        if self.config_path is not None:
            cfg.merge_from_file(self.config_path)

        # configure output directory
        os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

        # if resuming training, replace w preloaded model weights
        if self.resume:
            cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")

        # if labels haven't been loaded, warn user
        if "labels" in self._unloaded:
            logging.log(logging.WARNING, """
            IMPORTANT: config loaded before labels\n
            config will set number of classes to 0, and training and metadata cannot happen\n
            to train, run 'load_labels()' and then 'load_cfg()' again\n
            visualizations and predictions without metadata are still allowed
            """)
        else:
            cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(self._labels)

        self._cfg = cfg
        self._unloaded.remove("cfg")

    def load_dataset(self):
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

        if len(self._labels) == 0:
            print("Loading labels")
            self.load_labels()

        csvs = glob(self.csv_path + "*.csv")
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
                        tmp[img_name]["file_name"] = self.img_path+img_name
                        # source_id for later addition of unannotated files
                        tmp[img_name]["src_id"] = src_id
                        # mismatch in image sizes in annotatoin, just load in image :(
                        with Image.open(self.img_path+img_name) as i:
                            img_w, img_h = i.size

                        tmp[img_name]["width"] = img_w#min(864, int(vals[6]))
                        tmp[img_name]["height"] = img_h#min(864, int(vals[7]))
                        # image id can just be image name
                        tmp[img_name]["image_id"] = img_name
                        tmp[img_name]["annotations"] = []

                    # add bounding box
                    tmp[img_name]["annotations"].append({"bbox":[xmin, ymin, xw, yh],
                                                       "bbox_mode": BoxMode.XYWH_ABS,
                                                       "category_id": self._labels[vals[0]]})

                    # next line
                    line = f.readline().strip("\n")

        # for each row in the excel sheet (of form file name, use), confirm which images have
        # annotations and fill in those that don't by selecting another image of same source id
        use_files = pd.read_excel(self.excel_path, usecols = [1,2], names=["file", "use"])
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
                new_img_info["file_name"] = self.img_path + fname.lstrip("./")
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
        self._unloaded.remove("dataset")

    def load_splits(self):
        # make sure splits make sense
        assert sum(self.splits.values()) == 1, "Splits do not add to one"

        if "dataset" in self._unloaded:
            print("Loading dataset")
            self.load_dataset()

        # iterate over split_name, portion pairs
        splts = []
        cursplt = 0
        for key,val in self.splits.items():
            cursplt += int(val * len(self._data))
            splts.append(cursplt)

        # assign each chunk to a stage
        for chunk, stage in zip(np.split(self._data, splts[:-1]), self.splits.keys()):
            self._splits_data[stage] = chunk

        self._unloaded.remove("splits")

    def register_datasets(self):
        # ensure have labels and splits
        if "labels" in self._unloaded:
            print("Loading in labels")
            self.load_labels()
        if "splits" in self._unloaded:
            self.load_splits()

        # register datasets
        for stage in self.splits.keys():
            DatasetCatalog.register(self.dataname + stage, lambda x=stage: self._splits_data[x])
            MetadataCatalog.get(self.dataname + stage).thing_classes = list(self._labels.keys())
            # later, to access the data:
            #data: List[Dict] = DatasetCatalog.get("dimac_...")

    def setup_model(self):
        """
        wrapper for load dataset, labels, and config
        """
        # if labels haven't been loaded, load them
        if "labels" in self._unloaded:
            self.load_labels()

        # if config isnt loaded, load it
        if "cfg" in self._unloaded:
            self.load_cfg()

        # if data hasn't been loaded, load data
        if "dataset" in self._unloaded:
            self.load_dataset()

        # register datasets
        self.register_datasets()

        return True

    """
    model related methods (training, testing)
    """
    def train_model(self, plot=True):
        # assert modules loaded
        self._assert_loaded("cfg")
        self._assert_loaded("labels")
        self._assert_loaded("dataset")
        self._assert_loaded("splits")

        trainer = Trainer(self._cfg)
        trainer.resume_or_load(resume=self.resume)
        trainer.train()

        if plot:
            plotter.plot_val_with_total_loss(self._cfg.OUTPUT_DIR)

    def view_random_sample(self, n, stage, predict=False, threshold=None):
        """
        n: number of images t oview
        stage: which group to view from (train, test, valid)
        predict: make a prediction on the randomly selected images
        threshold: prediction threshold, will overwrite default
        """
        # assert modules loaded
        self._assert_loaded("cfg")
        self._assert_loaded("labels")
        self._assert_loaded("dataset")
        self._assert_loaded("splits")

        if threshold:
            self._cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = threshold # set a custom testing threshold

        metadata = MetadataCatalog.get(self.dataname + stage)

        # for random sample
        for d in np.random.choice(self._splits_data[stage], size=n, replace=False):
            # show data
            self.view_data_entry(d, predict=predict, metadata=metadata)


    def view_data_entry(self, fname, predict=False, metadata=None):
        # assert modules loaded
        self._assert_loaded("cfg")

        # show image
        img = cv2.imread(d["file_name"])
        v_d = Visualizer(img[:, :, ::-1], scale=0.5, metadata=metadata, instance_mode=ColorMode.SEGMENTATION)
        out = v_d.draw_dataset_dict(d)
        cv2.imshow(d["file_name"]+ " truth", out.get_image())

        # do prediction if asked
        if predict:
            self.predict_image(img, metadata=metadata, fname=d["file_name"])

        print("Press any key to clear windows")
        cv2.waitKey(0)
        cv2.destroyWindow(d["file_name"] + " predicted")
        cv2.destroyWindow(d["file_name"] + " truth")

    def predict_image(self, img, view=True, metadata=None, fname=""):
        # assert modules loaded
        self._assert_loaded("cfg")

        predictor = DefaultPredictor(self._cfg)
        preds = predictor(img)

        v = Visualizer(img[:,:,::-1], scale=0.5, metadata=metadata, instance_mode=ColorMode.SEGMENTATION)
        out = v.draw_instance_predictions(preds["instances"].to("cpu"))

        cv2.imshow(fname + " predicted", out.get_image()[:,:,::-1])
        print(f"Made {len(preds['instances'])} predictions for {fname}")
        print("Press any key to clear window")
        cv2.waitKey(0)
        cv2.destroyWindow(fname + " predicted")

        return preds

    def _assert_loaded(self, src):
        assert src not in self._unloaded, f"{src} not loaded, run 'load_{src}()' to load"

    def run_confusion_matrix(self, stage, threshold=None, iou_threshold=.5, plot=True, verbose=False, hypervisualize=False, slices=None):
        """
        if hypervisualize is True, verbose is automatically true
        """
        # assert modules loaded
        self._assert_loaded("cfg")
        self._assert_loaded("labels")
        self._assert_loaded("dataset")
        self._assert_loaded("splits")

        if threshold is not None:
            cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = threshold # set a custom testing threshold
        predictor = DefaultPredictor(self._cfg)

        # PYTHON 3.7 dict keys are in insertion order !!!
        labels = list(self._labels.keys())
        col_labels = labels + ["missed"]
        row_labels = labels + ["background"]

        cm = np.zeros((len(labels)+1, len(labels)+1))

        # get data
        dataset = self._splits_data[stage]
        if slices:
            dataset = dataset[slices]

        # get metadata
        metadata = MetadataCatalog.get(self.dataname + stage)

        # iterate over all entries, accumulate score predictions
        for d in tqdm(dataset):
            im = cv2.imread(d["file_name"])
            prediction = predictor(im)
            cm_tmp = score_prediction(prediction, d, nlabels=len(labels), iou_threshold=iou_threshold)
            cm += cm_tmp

            # show prediction and true
            if verbose or hypervisualize:
                # prediction for this attempt
                pprint_cm(cm_tmp, row_labels, col_labels)
                if hypervisualize:
                    # show predictions
                    v_p = Visualizer(im[:, :, ::-1], scale=0.5, metadata=metadata, instance_mode=ColorMode.SEGMENTATION)
                    out = v_p.draw_instance_predictions(prediction["instances"].to("cpu"))
                    cv2.imshow(d["file_name"] + " prediction", out.get_image()[:, :, ::-1])
                    # show truth
                    v_t = Visualizer(im[:, :, ::-1], scale=0.5, metadata=metadata, instance_mode=ColorMode.SEGMENTATION)
                    out = v_t.draw_dataset_dict(d)
                    cv2.imshow(d["file_name"] + " truth", out.get_image())

                    cv2.waitKey(0)
                    cv2.destroyWindow(d["file_name"] + " prediction")
                    cv2.destroyWindow(d["file_name"] + " truth")

        self._cm = cm

        # if plot, plot windows
        if plot:
            # plot cm total
            self.plot_confusion_matrix(iou_threshold, row=row_labels, col=col_labels)

        # print results
        pprint_cm(cm, row_labels, col_labels)

        # total number of predictions is all items except the last column b/c is missed detection
        return cm, cm[:, :-1].sum()

    """
    plotting tools (with plotter.py)
    """
    def plot_confusion_matrix(self, iou, row=[], col=[]):
        fig, ax = plt.subplots()
        im, cbar = plotter.heatmap(self._cm, row, col, ax=ax, cmap="YlOrRd", title=f"Confusion matrix total, IOU={iou}")
        texts = plotter.annotate_heatmap(im, valfmt="{x:.0f}")
        # set fig size and save
        fig.set_size_inches(8, 6.5)
        fig.savefig(self._cfg.OUTPUT_DIR+"/cm.png", dpi=100)

        # plot cm normalized
        fig1, ax1 = plt.subplots()
        im1, cbar1 = plotter.heatmap(self._cm/np.sum(self._cm, axis=1)[:, np.newaxis], row, col, ax=ax1, cmap="YlOrRd", title=f"Confusion matrix total, IOU={iou}")
        texts = plotter.annotate_heatmap(im1, valfmt="{x:.2f}")
        # set fig size and save
        fig1.set_size_inches(8, 6.5)
        fig1.savefig(self._cfg.OUTPUT_DIR+"/cm_normalized.png", dpi=100)
        plt.show()


parser = argparse.ArgumentParser()
parser.add_argument("-t", "--train", help="Train the network", action="store_true")
parser.add_argument("-c", "--confusion", help="Plot confusion matrix on test set", action="store_true")
parser.add_argument("-v", "--visualize", help="Visualize outputs. For train, this means visualizing sample training data.", type=int)
parser.add_argument("-p", "--predict", type=str, help="Run model for one image, specify path as argument.")
args = parser.parse_args()

if __name__ == "__main__":
    c = CNN()
    #print(DatasetCatalog.get("dimac_train"))
    if args.predict:
        c.load_cfg()
        print(c.predict_image(cv2.imread(args.predict)))
    if args.visualize:
        c.setup_model()
        c.view_random_sample(args.visualize, "train")
    if args.train:
        c.setup_model()
        c.train_model()
    if args.confusion:
        c.setup_model()
        c.run_confusion_matrix("valid", slices=range(3,6), hypervisualize=True)

#TODO: command line arguments
