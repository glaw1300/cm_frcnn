import torch
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import numpy as np
import os, json, random

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
import detectron2.data.datasets
from detectron2.structures import BoxMode
from glob import glob
import pandas as pd
import cv2
from PIL import Image

# load in labels into dict and assign id values for each label
print("Loading in labels...")
label_ids = {}
with open("labels.txt", "r") as f:
    idx = 0
    line = f.readline().strip("\n")
    while line:
        # omit wells and processing plants for now
        if line == "well" or line == "processing":
            line = f.readline().strip("\n")
            continue
        label_ids[line] = idx
        line = f.readline().strip("\n")
        idx += 1
print("Labels loaded!")

def load_dimac_dataset():
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

    csvs = glob("exports/csv/*.csv")
    # read in each annotation in each csv
    # CSV FORMAT: label,xmin,ymin,box_width,box_height,img_name,img_width,img_height
    for csv in csvs:
        with open(csv, "r") as f:
            line = f.readline().strip("\n")
            # for each box
            while line:
                vals = line.split(",")
                # omit well pads and processing for now
                if vals[0] == "processing" or vals[0] == "well":
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
                    tmp[img_name]["file_name"] = "../DIMAC_jpeg/"+img_name
                    # source_id for later addition of unannotated files
                    tmp[img_name]["src_id"] = src_id
                    # mismatch in image sizes in annotatoin, just load in image :(
                    with Image.open("../DIMAC_jpeg/"+img_name) as i:
                        img_w, img_h = i.size

                    tmp[img_name]["width"] = img_w#min(864, int(vals[6]))
                    tmp[img_name]["height"] = img_h#min(864, int(vals[7]))
                    # image id can just be image name
                    tmp[img_name]["image_id"] = img_name
                    tmp[img_name]["annotations"] = []

                """
                # make bounding boxes 864 x 864 square
                if xmin + xw > 864:
                    xw = 864. - xmin
                if ymin + yh > 864:
                    yh = 864. - ymin
                """

                # add bounding box
                tmp[img_name]["annotations"].append({"bbox":[xmin, ymin, xw, yh],
                                                   "bbox_mode": BoxMode.XYWH_ABS,
                                                   "category_id": label_ids[vals[0]]})

                # next line
                line = f.readline().strip("\n")

    # for each row in the excel sheet (of form file name, use), confirm which images have
    # annotations and fill in those that don't by selecting another image of same source id
    use_files = pd.read_excel("fnames.xlsx", usecols = [1,2], names=["file", "use"])
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
            new_img_info["file_name"] = "../DIMAC_jpeg/" + fname.lstrip("./")
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

def get_data_chunk(stage, data):
    # get train test or valid chunk from data
    ttv_inds = {"train": 0, "valid": 1, "test":2}
    ttv = np.split(data, [int(len(data)*.7), int(len(data)*.85)])
    return ttv[ttv_inds[stage]]

# register dataset for 3 stages
print("Loading dataset...")
dataset_dicts = load_dimac_dataset()
print("Dataset loaded!")

for stage in ["train", "valid", "test"]:
    DatasetCatalog.register("dimac_" + stage, lambda x=stage: get_data_chunk(x, dataset_dicts))
    MetadataCatalog.get("dimac_" + stage).thing_classes = list(label_ids.keys())
# later, to access the data:
#data: List[Dict] = DatasetCatalog.get("dimac")

metadata = MetadataCatalog.get("dimac_train")
for d in dataset_dicts[-10:]:#random.sample(list(dataset_dicts), 20):
    img = cv2.imread(d["file_name"])
    visualizer = Visualizer(img[:, :, ::-1], metadata=metadata, scale=0.5)
    out = visualizer.draw_dataset_dict(d)
    cv2.imshow(d["file_name"], out.get_image())
    cv2.waitKey(0)
    cv2.destroyWindow(d["file_name"])

from detectron2.engine import DefaultTrainer
from detectron2.checkpoint import DetectionCheckpointer

# configure model
print("Loading config...")
cfg = get_cfg()
# checkpoint!
cfg.OUTPUT_DIR = "output"
cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml"))
cfg.MODEL.DEVICE='cpu' # running on cpu
cfg.DATASETS.TRAIN = ("dimac_train",)
cfg.DATASETS.TEST = ("dimac_test",)
cfg.DATASETS.VALID = ("dimac_valid",)
cfg.DATALOADER.NUM_WORKERS = 1
cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")  # Let training initialize from model zoo
cfg.SOLVER.IMS_PER_BATCH = 2
cfg.SOLVER.BASE_LR = 0.0001  # pick a good LR
cfg.SOLVER.MAX_ITER = 1000    # 300 iterations seems good enough for this toy dataset; you will need to train longer for a practical dataset
cfg.SOLVER.STEPS = []        # do not decay learning rate
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128   # faster, and good enough for this toy dataset (default: 512)
cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(label_ids)
print("Config loaded!")

# train
print("Setting up trainer...")
os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
trainer = DefaultTrainer(cfg)
trainer.resume_or_load(resume=True)
#trainer.train()

# validate
from detectron2.utils.visualizer import ColorMode
from detectron2.engine import DefaultPredictor

cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.4   # set a custom testing threshold
predictor = DefaultPredictor(cfg)
test_dataset_dicts = get_data_chunk("test", dataset_dicts)
for d in test_dataset_dicts:
    im = cv2.imread(d["file_name"])
    outputs = predictor(im)  # format is documented at https://detectron2.readthedocs.io/tutorials/models.html#model-output-format
    v = Visualizer(im[:, :, ::-1],
                   metadata=metadata,
                   scale=0.5,
                   instance_mode=ColorMode.IMAGE_BW   # remove the colors of unsegmented pixels. This option is only available for segmentation models
    )
    if len(outputs["instances"]) > 0:
        out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        cv2.imshow(d["file_name"], out.get_image()[:, :, ::-1])
        cv2.waitKey(0)
        cv2.destroyAllWindows()
