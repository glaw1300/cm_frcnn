# allow imports from scripts
import sys
sys.path.append("scripts")
from faster_rcnn import CNN
import evaluator
from predictor import Predictor
from dataloader import DataLoader
import argparse
import utils
import cv2

# EVALUATOR SETTINGS
evaluator.NEW_IOU_MIN = .35
"""
# custom area ranges for examining performance at finer area resolutions
evaluator.CUSTOM_AREA_RANGES = [
    [0, 1e5 ** 2], # all
    [0, 256], # smallest
    [256, 512], # next smallest
    [512, 1024], # last smallest
    [1024, 2048], # medium
    [2048, 4096], # next medium
    [4096, 8192], # last medium
    [8192, 16384], # large
    [16384, 1e5 ** 2] # rest of large
]
evaluator.CUSTOM_AREA_LBLS = [
    "all", "0-2^8",
    *[f"2^{i}-2^{i+1}" for i in range(8,14)],
    ">2^14"
]
"""
# evaluator to use in CNN
e = evaluator.CustomCOCOEvaluator

# PREDICTOR SETTINGS
# predictor to use in CNN
p = Predictor

# DATALOADER SETTINGS (see README.md for data directory architecture)
# relevant directories
DataLoader.data_dir = "data"
DataLoader.annotation_dir = "annotations"
DataLoader.model_dir = "model"
DataLoader.csv_dir = "csvs"
DataLoader.img_path = "../DIMAC_jpeg"
# relevant file names
DataLoader.labels_fname = "labels.txt"
DataLoader.weights_fname = "model_final.pth"
DataLoader.cfg_fname = "cfg.yaml"
DataLoader.has_json = True
# overload method for loading in data
# TODO: IF YOU WANT TO CHANGE HOW YOU LOAD IN DATA, DO SO HERE
DataLoader.load_dataset = utils.load_dataset_from_csvs
DataLoader._make_splits = utils.make_splits_by_src_id

# Add all preceding classes to CNN
CNN.predictor = p
CNN.evaluator = e
CNN.dataloader = DataLoader

# RUNNING:
parser = argparse.ArgumentParser()
parser.add_argument("-t", "--train", help="Train the network", action="store_true")
parser.add_argument("-c", "--confusion", help="Plot confusion matrix on test set", action="store_true")
parser.add_argument("-v", "--visualize", help="Visualize outputs. For train, this means visualizing sample training data.", type=int)
parser.add_argument("-p", "--predict", type=str, help="Run model for one image, specify path as argument.")
parser.add_argument("-e", "--eval_all", help="Run COCOEvaluator on train and test datasets", action="store_true")
parser.add_argument("--edit", help="Adjust bounding boxes of dataset manually", action="store_true")
args = parser.parse_args()

c = CNN()
c.setup_model()

# parse command line arguments
if args.train:
    c.train_model()
if args.confusion:
    c.run_confusion_matrix("test", verbose=True)
if args.predict:
    print(c.predict_image(cv2.imread(args.predict), fname=args.predict, title=args.predict))
if args.visualize:
    c.view_random_sample(args.visualize, "train")
if args.eval_all:
    for stage in ["test", "train"]:
        c.do_test(stage)
if args.edit:
    c.dl.adjust_bboxes()
