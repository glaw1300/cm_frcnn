from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import numpy as np
import os
import glob
import cv2
from PIL import Image
import logging
import matplotlib.pyplot as plt
from trainer import Trainer
import argparse
from utils import *
import plotter
from tqdm import tqdm
from collections import OrderedDict
import sys
from evaluator import CustomCOCOEvaluator
from dataloader import DataLoader
from predictor import Predictor

# import detectron2 utilities
from detectron2 import model_zoo
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.data import MetadataCatalog, build_detection_test_loader
import detectron2.data.datasets
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.utils.visualizer import ColorMode
from detectron2.engine import DefaultPredictor
from detectron2.evaluation import COCOEvaluator, inference_on_dataset, print_csv_format
from detectron2.modeling import build_model

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
    dataloader = DataLoader
    predictor = Predictor
    evaluator = CustomCOCOEvaluator

    @initializer # wrapper that unpacks each kwarg into self.<variable name> (from utils)
    def __init__(self, dataloader=None):
        # instantiate dataloader
        self.dl = dataloader or self.dataloader()

        # confusion matrix
        self._cm = None


    """
    getter methods
    """
    def get_dataset(self):
        return self.dl.get_dataset()

    def get_labels(self):
        return self.dl.get_labels()

    def get_splits(self):
        return self.dl.get_splits()

    def get_cfg(self):
        return self.dl.get_cfg()

    def get_confusion_matrix(self):
        if self._cm == None:
            raise MissingDataError(src="confusion matrix", message="Confusion matrix has not been run, call 'run_confusion matrix()' to get: ")
        return self._cm

    def setup_model(self):
        """
        wrapper for load dataset, labels, and config
        """
        # load all data with dataloader
        logging.log(logging.INFO, "Loading in annotations, config with DataLoader")
        self.dl.load_all_data()

        # register datasets
        self.dl.register_datasets()

        return True

    """
    model related methods (training, testing)
    """
    def train_model(self, plot=False, resume=True):

        trainer = Trainer(self.dl._cfg)
        trainer.resume_or_load(resume=resume)
        trainer.train()

        if plot:
            plotter.plot_val_with_total_loss(self.dl._cfg.OUTPUT_DIR)

    def view_random_sample(self, n, stage, predict=False, threshold=None):
        """
        n: number of images t oview
        stage: which group to view from (train, test, valid)
        predict: make a prediction on the randomly selected images
        threshold: prediction threshold, will overwrite default
        """

        if threshold:
            self.dl._cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = threshold # set a custom testing threshold

        metadata = MetadataCatalog.get(self.dl.dataname + stage)

        # for random sample
        for d in np.random.choice(self.dl._splits_data[stage], size=n, replace=False):
            # show data
            self.view_data_entry(d, predict=predict, metadata=metadata)


    def view_data_entry(self, d, predict=False, metadata=None):
        # assert modules loaded

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

    def predict_image(self, img, view=True, metadata="train", fname="", threshold=.25, verbose=False, crop=False, cent=None):
        # if crop, take 864 x 864 center of image
        if crop:
            if cent:
                x, y = cent
            else:
                y, x, _ = img.shape
                x /= 2
                y /= 2
            cut = 864
            img = img[int(y - cut / 2) : int(y + cut/2), int(x - cut/2 ): int(x+cut/2), :]

        self.dl._cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = threshold

        # try to get metadata
        try:
            metadata = MetadataCatalog.get(self.dl.dataname + metadata)
        except:
            metadata = None

        p = self.predictor(self.dl._cfg)
        preds = p(img)

        v = Visualizer(img[:,:,::-1], scale=0.5, metadata=metadata, instance_mode=ColorMode.SEGMENTATION)
        out = v.draw_instance_predictions(preds["instances"].to("cpu"))

        if verbose:
            logging.log(logging.INFO, f"Made {len(preds['instances'])} predictions for {fname}")

        if view:
            plotter.plot_pred_boxes(img, preds, self.dl._labels)

        return preds

    def run_confusion_matrix(self, stage, nms_thresh=.25, threshold=None, iou_threshold=.35, plot=True, verbose=False, hypervisualize=False, slices=None):
        """
        if hypervisualize is True, verbose is automatically true
        """

        # default to .5, decrease in our case because overlappign infrastructure won't happen
        self.dl._cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = nms_thresh

        if threshold is not None:
            self.dl._cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = threshold # set a custom testing threshold
        p = self.predictor(self.dl._cfg)

        labels = list(self.dl._labels)
        col_labels = labels + ["missed"]
        row_labels = labels + ["background"]

        cm = np.zeros((len(labels)+1, len(labels)+1))

        # get data
        dataset = self.dl._splits_data[stage]
        if slices:
            dataset = dataset[slices]

        # get metadata
        metadata = MetadataCatalog.get(self.dl.dataname + stage)

        # iterate over all entries, accumulate score predictions
        for d in tqdm(dataset):
            im = cv2.imread(d["file_name"])
            prediction = p(im)
            cm_tmp = score_prediction(prediction, d, nlabels=len(labels), iou_threshold=iou_threshold)
            cm += cm_tmp

            # show prediction and true
            if verbose or hypervisualize:
                # prediction for this attempt
                logging.log(logging.INFO, d["file_name"])
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

    def do_test(self, stage):

        datasets = {
                    "test" : self.dl._cfg.DATASETS.TEST,
                    "train": self.dl._cfg.DATASETS.TRAIN
                    }
        model = build_model(self.dl._cfg)
        weights_path = os.path.join(
            self.dl.data_dir, self.dl.model_dir, self.dl.weights_fname
        )
        DetectionCheckpointer(model).load(weights_path)

        results = OrderedDict()
        for dataset_name in datasets[stage]:
            data_loader = build_detection_test_loader(self.dl._cfg, dataset_name)
            e = self.evaluator(dataset_name, tasks=("bbox",), distributed=True, output_dir=self.dl._cfg.OUTPUT_DIR)
            #evaluator = COCOEvaluator(dataset_name, tasks=("bbox",), distributed=True, output_dir=self.dl._cfg.OUTPUT_DIR)
            results_i = inference_on_dataset(model, data_loader, e)
            results[dataset_name] = results_i

            print("Evaluation results for {} in csv format:".format(dataset_name))
            print_csv_format(results_i)

        if len(results) == 1:
            results = list(results.values())[0]

        return results

    """
    plotting tools (with plotter.py)
    """
    def plot_confusion_matrix(self, iou, row=[], col=[]):

        fig, ax = plt.subplots()
        im, cbar = plotter.heatmap(self._cm, row, col, ax=ax, cmap="YlOrRd", title=f"Confusion matrix total, IOU={iou}")
        texts = plotter.annotate_heatmap(im, valfmt="{x:.0f}")
        # set fig size and save
        fig.set_size_inches(8, 6.5)
        fig.savefig(os.path.join(self.dl._cfg.OUTPUT_DIR, "cm.png"), dpi=100)

        # plot cm normalized
        fig1, ax1 = plt.subplots()
        im1, cbar1 = plotter.heatmap(self._cm/np.sum(self._cm, axis=1)[:, np.newaxis], row, col, ax=ax1, cmap="YlOrRd", title=f"Confusion matrix total, IOU={iou}")
        texts = plotter.annotate_heatmap(im1, valfmt="{x:.2f}")
        # set fig size and save
        fig1.set_size_inches(8, 6.5)
        fig1.savefig(os.path.join(self.dl._cfg.OUTPUT_DIR, "cm_normalized.png"), dpi=100)
        plt.show()


parser = argparse.ArgumentParser()
parser.add_argument("-t", "--train", help="Train the network", action="store_true")
parser.add_argument("-c", "--confusion", help="Plot confusion matrix on test set", action="store_true")
parser.add_argument("-v", "--visualize", help="Visualize outputs. For train, this means visualizing sample training data.", type=int)
parser.add_argument("-p", "--predict", type=str, help="Run model for one image, specify path as argument.")
parser.add_argument("-e", "--eval_all", help="Run COCOEvaluator on all 3 segments of datasets", action="store_true")
parser.add_argument("--edit", help="Adjust bounding boxes of dataset manually", action="store_true")
args = parser.parse_args()

if __name__ == "__main__":
    c = CNN()
    #print(DatasetCatalog.get("dimac_train"))
    if args.predict:
        c.setup_model()
        print(c.predict_image(cv2.imread(args.predict), fname=args.predict))
    if args.visualize:
        c.setup_model()
        c.view_random_sample(args.visualize, "train")
    if args.train:
        c.setup_model()
        c.train_model()
    if args.confusion:
        c.setup_model()
        c.run_confusion_matrix("test", verbose=True)
        #c.run_confusion_matrix("valid", verbose=True)
    if args.eval_all:
        c.setup_model()
        #for i in np.arange(0, 1, .05):
        #    c._cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = float(i)
        for stage in ["test"]:#, "valid", "train"]:
            #print(f"Evaluating on {stage} @ {i}")
            c.do_test(stage)
    if args.edit:
        c.load_dataset()
        c.adjust_bboxes()

#TODO: command line arguments
