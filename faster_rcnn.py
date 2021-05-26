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

# import detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
import detectron2.data.datasets
from detectron2.structures import BoxMode
from detectron2.engine import DefaultTrainer
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.utils.visualizer import ColorMode
from detectron2.engine import DefaultPredictor
import argparse

"""
General running information:
- To run this code you need the following files:
    labels.txt - list of all categories for data
    [directory] images - directory of all of your training/test images
    csv(s) - annotation csvs of form label,xmin,ymin,box_width,box_height,img_name,img_width,img_height
    excel spreadsheet - two columns, filename and files to use (denoted with an "x")

To specify the location of these files, simply change the constants below this comment
"""
parser = argparse.ArgumentParser()
parser.add_argument("-t", "--train", help="Train the network", action="store_true")
parser.add_argument("-c", "--confusion", help="Plot confusion matrix on test set", action="store_true")
parser.add_argument("-v", "--visualize", help="Visualize outputs. For train, this means visualizing sample training data.", action="store_true")
args = parser.parse_args()

LABELS = "labels.txt" # location of labels file (format each line is label)
EXCLUDE = ["well", "processing"] # make empty list if none
IMG_PATH = "../DIMAC_jpeg/" # where images are located (WITH trailing /)
CSV_PATH = "exports/csv/*.csv" # where annotation csvs are stored (format compatible w glob)
EXCEL_PATH = "fnames.xlsx" # location of excel file of images to include
OUTPUT_DIR = "output" # directory where checkpoints should be stored and loaded from

def load_labels(path=LABELS, exclude=EXCLUDE):
    # load in labels into dict and assign id values for each label
    print("Loading in labels...")
    label_ids = {}
    with open(path, "r") as f:
        idx = 0
        line = f.readline().strip("\n")
        while line:
            # omit wells and processing plants for now
            if line in exclude:
                line = f.readline().strip("\n")
                continue
            label_ids[line] = idx
            line = f.readline().strip("\n")
            idx += 1
    print("Labels loaded!")
    return label_ids

def load_dimac_dataset(labels, csv_path_glob=CSV_PATH, exclude=EXCLUDE, img_path=IMG_PATH,
                       excel_path=EXCEL_PATH):
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
    print("Loading data...")

    # load in boxes from csv and add to img list
    srcs = {}

    csvs = glob(csv_path_glob)
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
                    tmp[img_name]["file_name"] = img_path+img_name
                    # source_id for later addition of unannotated files
                    tmp[img_name]["src_id"] = src_id
                    # mismatch in image sizes in annotatoin, just load in image :(
                    with Image.open(img_path+img_name) as i:
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
            new_img_info["file_name"] = img_path + fname.lstrip("./")
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

    print("Data loaded!")
    return np.array(final)

def get_data_chunk(stage, data):
    # get train test or valid chunk from data
    ttv_inds = {"train": 0, "valid": 1, "test":2}
    ttv = np.split(data, [int(len(data)*.7), int(len(data)*.85)])
    return ttv[ttv_inds[stage]]

def load_and_register_data(imshow = False, imshow_num = 3, **kwargs):
    """
    imshow: display imshow_num test images to confirm annotation location
    **kwargs: keyword arguments for load_dimac_dataset and load_labels

    returns labels and data
    """
    # load labels
    label_ids = load_labels(**kwargs)

    # load data
    dataset_dicts = load_dimac_dataset(label_ids, **kwargs)

    print("Registering dataset...")
    for stage in ["train", "valid", "test"]:
        DatasetCatalog.register("dimac_" + stage, lambda x=stage: get_data_chunk(x, dataset_dicts))
        MetadataCatalog.get("dimac_" + stage).thing_classes = list(label_ids.keys())
        # later, to access the data:
        #data: List[Dict] = DatasetCatalog.get("dimac_...")

    metadata = MetadataCatalog.get("dimac_train")
    print("Dataset registered!")

    if imshow:
        print(f"Showing {imshow_num} random images with annotations. Press any key to continue.")
        for d in random.sample(list(dataset_dicts), imshow_num):
            img = cv2.imread(d["file_name"])
            visualizer = Visualizer(img[:, :, ::-1], metadata=metadata, scale=0.5)
            out = visualizer.draw_dataset_dict(d)
            cv2.imshow(d["file_name"], out.get_image())
            cv2.waitKey(0)
            cv2.destroyWindow(d["file_name"])

    return label_ids, dataset_dicts

def setup_config(output=OUTPUT_DIR, resume=True, nlabels=4, **kwargs):
    """

    """
    # configure model
    print("Loading config...")
    cfg = get_cfg()
    # configure output directory
    cfg.OUTPUT_DIR = output
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    # use faster_rcnn w resnext101
    cfg.merge_from_file(model_zoo.get_config_file(kwargs.get("model_config", "COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml")))
    cfg.MODEL.DEVICE=kwargs.get("runon", "cpu") # running on cpu
    cfg.DATASETS.TRAIN = ("dimac_train",)
    cfg.DATASETS.TEST = ("dimac_test",)
    cfg.DATASETS.VALID = ("dimac_valid",)

    # use number of cpus
    cfg.DATALOADER.NUM_WORKERS = kwargs.get("ncores", 4)
    # if resuming training, use preloaded model weights
    if resume:
        cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
    else:
        cfg.MODEL.WEIGHTS = kwargs.get("model_config", "COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml")

    # training params
    cfg.SOLVER.IMS_PER_BATCH = kwargs.get("ncores", 4)
    cfg.SOLVER.BASE_LR = kwargs.get("lr", .0001)  # LR of .00025 did not converge
    cfg.SOLVER.MAX_ITER = kwargs.get("max_iter", 250)
    cfg.SOLVER.STEPS = kwargs.get("lr_decay", [500, 1000])# decay learning rate at each iteration
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = kwargs.get("batch_size", 128)   # vary based on number of images
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = nlabels
    print("Config loaded!")

    return cfg

# train
def train_model(cfg, resume=True):
    print("Setting up trainer...")
    trainer = DefaultTrainer(cfg)
    trainer.resume_or_load(resume=resume)
    trainer.train()

# test
def visualize_test(cfg, dataset="dimac_train", stage="train", threshold = .4, nimages=5):
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = threshold # set a custom testing threshold
    predictor = DefaultPredictor(cfg) # predictor
    # get data and metadata
    test_dataset_dicts = get_data_chunk(stage, DatasetCatalog.get(dataset))
    metadata = MetadataCatalog.get(dataset)
    for d in random.sample(list(test_dataset_dicts), nimages):
        im = cv2.imread(d["file_name"])
        print(f"Predicting for {d['file_name']}")
        outputs = predictor(im)  # format is documented at https://detectron2.readthedocs.io/tutorials/models.html#model-output-format
        v = Visualizer(im[:, :, ::-1],
                       metadata=metadata,
                       scale=0.5
                       )
        # if have predictions
        if len(outputs["instances"]) > 0:
            out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
            cv2.imshow(d["file_name"], out.get_image()[:, :, ::-1])
            cv2.waitKey(0)
            cv2.destroyWindow(d["file_name"])
        else:
            print(f"No instances found for: {d['file_name']}")

if __name__ == "__main__":
    labels, data = load_and_register_data()
    cfg = setup_config(nlabels=len(labels), max_iter=2000)
    #print(DatasetCatalog.get("dimac_train"))
    if args.visualize:
        print("Displaying sample of training data")
        visualize_test(cfg)
    if args.train:
        print("Training model")
        train_model(cfg)
    if args.confusion:
        print("Running confusion matrix")
        get_confusion_matrix(DatasetCatalog.get("dimac_valid")[:1], cfg)
