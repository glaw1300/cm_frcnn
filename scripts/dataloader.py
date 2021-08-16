from detectron2.utils.logger import setup_logger
setup_logger()

import os
from utils import *
from detectron2.config import get_cfg, CfgNode
from detectron2 import model_zoo
import logging
from detectron2.data import DatasetCatalog, MetadataCatalog

class DataLoader:
    # relevant directories
    data_dir = "data"
    annotation_dir = "annotations"
    model_dir = "model"
    csv_dir = "csvs"
    img_path = "../DIMAC_jpeg"
    # relevant file names
    labels_fname = "labels.txt"
    weights_fname = "model_final.pth"
    cfg_fname = "cfg.yaml"
    has_json = True

    @initializer
    def __init__(self, exclude=["well"], dataname="dimac_",
                 splits={"train":.7, "valid":.15, "test":.15}, **kw):
        # unpack other kwargs (can overwrite all class attributes)
        for key, val in kw.items():
            setattr(self, key, val)

        # config
        self._cfg = None

        # labels list
        self._labels = []

        # for loading in data
        self._data = []
        self._splits_data = {key:[] for key in splits.keys()}

        # build relevant paths
        self.labels_path = os.path.join(
            self.data_dir, self.annotation_dir, self.labels_fname
        )
        self.cfg_path = os.path.join(
            self.data_dir, self.model_dir, self.cfg_fname
        )
        self.model_weights_path = os.path.join(
            self.data_dir, self.model_dir, self.weights_fname
        )
        self.annotation_path = os.path.join(
            self.data_dir, self.annotation_dir
        )
        self.csv_path = os.path.join(
            self.data_dir, self.annotation_dir, self.csv_dir
        )

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

    """
    loader methods
    """
    def load_labels(self):
        # open label file
        with open(self.labels_path, "r") as f:
            # load labels as list
            for line in f:
                # omit labels
                if line.strip("\n") not in self.exclude:
                    self._labels.append(line.strip("\n"))

        # if no labels, raise error
        if len(self._labels) == 0:
            raise MissingDataError("labels")

    def load_cfg(self):
        # if no labels, raise error
        # if no labels, raise error
        if len(self._labels) == 0:
            raise MissingDataError("labels")

        # configure model
        # get default config
        default_cfg = get_cfg()

        # load project cfg
        custom_cfg = CfgNode.load_cfg(open(self.cfg_path))
        # load model config and merge w default
        # https://github.com/facebookresearch/detectron2/blob/master/MODEL_ZOO.md <- model config options
        model_cfg = default_cfg.merge_from_file(
            model_zoo.get_config_file(custom_cfg.pop("MODEL_CFG"))
        )

        # merge with default config
        default_cfg.merge_from_other_cfg(custom_cfg)

        # configure output directory
        os.makedirs(default_cfg.OUTPUT_DIR, exist_ok=True)

        # if model weights exist, load model weights
        if os.path.exists(self.model_weights_path):
            default_cfg.MODEL.WEIGHTS = self.model_weights_path

        # set number of classes
        default_cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(self._labels)

        self._cfg = default_cfg

    def load_dataset(self):
        """
        must assign self._data a list of dictionaries, each dict is item in dataset
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

        default treats data file as pickle file
        """
        # labels must be loaded to load dataset
        if len(self._labels) == 0:
            logging.log(logging.WARNING, "Labels had not been loaded, loading now")
            self.load_labels()

        # if has json, load data from json
        if self.has_json:
            logging.log(logging.INFO, "Loading data from JSON")
            self._load_json(["data"])
            return

        # otherwise, load data from picle by default
        with open(os.path.join(self.annotation_path, "data.pickle"), "rb") as f:
            self._data = pickle.load(f)


    def load_splits(self):
        # if resuming with json data, do that
        if self.has_json:
            logging.log(logging.INFO, "Loading splits from JSON")
            self._load_json(self.splits.keys())
            return

        # make sure splits make sense
        assert sum(self.splits.values()) == 1, "Splits do not add to one"

        if len(self._data) == 0:
            logging.log(logging.INFO, "Data not loaded, loading")
            self.load_dataset()

        self._make_splits()

    def load_all_data(self):
        # load in labels
        if len(self._labels) == 0:
            self.load_labels()

        # load in dataset
        if len(self._data) == 0:
            self.load_dataset()

        # load in splits (if first split is empty, load in)
        if len(list(self._splits_data.values())[0]) == 0:
            self.load_splits()

        # if no config, load config
        if self._cfg == None:
            self.load_cfg()

    def register_datasets(self):
        # ensure have labels and splits
        if len(self._labels) == 0:
            logging.log(logging.INFO, "Loading labels")
            self.load_labels()
        if len(list(self._splits_data.values())[0]) == 0:
            logging.log(logging.INFO, "Loading splits")
            self.load_splits()

        # register datasets
        for stage in self.splits.keys():
            DatasetCatalog.register(self.dataname + stage, lambda x=stage: self._splits_data[x])
            MetadataCatalog.get(self.dataname + stage).thing_classes = list(self._labels)
            # later, to access the data:
            #data: List[Dict] = DatasetCatalog.get("dimac_...")


    """
    Additional helper methods for loading JSON data
    """
    def _load_json(self, srcs):
        # load each source
        for src in srcs:
            # get list for storage
            if src == "data":
                l = self._data
            else:
                l = self._splits_data[src]

            # load data
            with open(os.path.join(self.annotation_path, src+".json"), "r") as f:
                for line in f:
                    js = json.loads(line.strip())
                    # map annotations to id, keep all not in exclude
                    fin_annot = []
                    for annot in js["annotations"]:
                        # also make sure the annotation is in labels
                        if annot["category_id"] in self._labels:
                            annot["category_id"] = self._labels.index(annot["category_id"])
                            fin_annot.append(annot)
                    js["annotations"] = fin_annot
                    l.append(js)

    def _make_splits(self):
        # iterate over split_name, portion pairs
        splts = []
        cursplt = 0
        for key,val in self.splits.items():
            cursplt += int(val * len(self._data))
            splts.append(cursplt)

        # assign each chunk to a stage
        for chunk, stage in zip(np.split(self._data, splts[:-1]), self.splits.keys()):
            self._splits_data[stage] = chunk

    def _load_json(self, srcs):
        # load each source
        for src in srcs:
            # get list for storage
            if src == "data":
                l = self._data
            else:
                l = self._splits_data[src]

            # load data
            with open(os.path.join(self.annotation_path, src+".json"), "r") as f:
                for line in f:
                    js = json.loads(line.strip())
                    # map annotations to id, keep all not in exclude
                    fin_annot = []
                    for annot in js["annotations"]:
                        # also make sure the annotation is in labels
                        if annot["category_id"] in self._labels:
                            annot["category_id"] = self._labels.index(annot["category_id"])
                            fin_annot.append(annot)
                    js["annotations"] = fin_annot
                    l.append(js)


    def _assert_loaded(self, src, src_name):
        assert type(src) != type(None) and len(src) != 0, f"{src_name} not loaded, run 'load_{src_name}()' to load"

    """
    Dataset management (adjusting bounding boxes, saving)
    """

    def adjust_bboxes(self, verbose=True, data="dataset", save=False, add=True):
        """
        Overwrites self._data by default, can specify which split
        """
        self._assert_loaded(self._data, "dataset")

        if data == "dataset":
            dataset = self._data
        else:
            self._assert_loaded(list(self._splits_data.values())[0], "splits")
            dataset = self._splits_data[data]

        # control boxes with arrows
        def _on_press(event):
            sys.stdout.flush()
            # increment on keypress
            if event.key == "left":
                x_slider.set_val(max(x_slider.val - 1, x_slider.valmin))
            elif event.key == "right":
                x_slider.set_val(min(x_slider.val + 1, x_slider.valmax))
            elif event.key == "up":
                y_slider.set_val(max(y_slider.val - 1, y_slider.valmin))
            elif event.key == "down":
                y_slider.set_val(min(y_slider.val + 1, y_slider.valmax))
            # shift increases by 5
            if event.key == "shift+left":
                x_slider.set_val(max(x_slider.val - 5, x_slider.valmin))
            elif event.key == "shift+right":
                x_slider.set_val(min(x_slider.val + 5, x_slider.valmax))
            elif event.key == "shift+up":
                y_slider.set_val(max(y_slider.val - 5, y_slider.valmin))
            elif event.key == "shift+down":
                y_slider.set_val(min(y_slider.val + 5, y_slider.valmax))
            # terminate on ctrl+c
            elif event.key == "ctrl+c":
                logging.log(logging.INFO, "CTRL+C: ABORTING")
                sys.exit(0)

        # update functions for slider
        def _update_y(val):
            for r, coords in zip(rects, og_coords):
                r.rect.set_y(y_slider.val + coords[1])

        def _update_x(val):
            for r, coords in zip(rects, og_coords):
                r.rect.set_x(x_slider.val + coords[0])

        for i in tqdm(range(len(dataset))):
            # setup subplots
            fig, ax = plt.subplots()
            fig.set_size_inches(8, 8)
            fig.subplots_adjust(left=0.25, bottom=0.25)

            # add sliders
            x_slider_ax  = fig.add_axes([0.25, 0.15, 0.65, 0.03])
            x_slider = Slider(x_slider_ax, 'Horiz', -30, 30, valstep=1, valinit=0)

            # Draw another slider
            y_slider_ax  = fig.add_axes([0.25, 0.1, 0.65, 0.03])
            y_slider = Slider(y_slider_ax, 'Vert', -30, 30, valstep=1, valinit=0)

            # set sliders
            x_slider.on_changed(_update_x)
            y_slider.on_changed(_update_y)

            # make all rectangles
            rects = []
            og_coords = []
            for annot in dataset[i]["annotations"]:
                x, y, w, h = annot["bbox"]
                dr = plotter.DraggableRectangle(patches.Rectangle((x,y),w,h, edgecolor='r', facecolor="none"), fig)
                dr.connect()
                rects.append(dr)
                og_coords.append((x,y))

            # show rectangle
            for r in rects:
                ax.add_patch(r.rect)

            im = plt.imread(dataset[i]["file_name"])
            ax.imshow(im)

            fig.canvas.mpl_connect('key_press_event', _on_press)
            logging.log(logging.INFO, "Press (q) to close")
            plt.title(dataset[i]["file_name"])
            plt.show()

            # show final changes if verbose
            if verbose:
                for r, xy in zip(rects, og_coords):
                    logging.log(logging.INFO, f"x,y: {xy[0]},{xy[1]} ---> {r.rect.get_x()},{r.rect.get_y()}")

            # update dataset
            for j, r in enumerate(rects):
                if data == "dataset":
                    self._data[i]["annotations"][j]["bbox"][0] = r.rect.get_x()
                    self._data[i]["annotations"][j]["bbox"][1] = r.rect.get_y()
                else:
                    self._splits_data[data][i]["annotations"][j]["bbox"][0] = r.rect.get_x()
                    self._splits_data[data][i]["annotations"][j]["bbox"][1] = r.rect.get_y()

        # if modified data, modify splits
        if data == "dataset":
            self._make_splits()

        # if save datasets, save datasets
        if save:
            self.save_dataset(add=add)
            self.save_splits(add=add)#TODO should probably jsut save one

    def save_dataset(self, path=None, add=True):
        self._assert_loaded(self._data, "dataset")
        # if not path, do DataLoader data_dir
        if not path:
            path = self.dl.data_dir

        os.makedirs(path, exist_ok=True)

        # save as json
        self._save_json_list(self._data, os.path.join(path, "data.json"), add=add, sub_annots=True)

    def save_splits(self, path=None, add=True):
        self._assert_loaded(list(self._splits_data.values())[0], "splits")
        # priority: path parameter, cfg output_dir, exports
        # if not path, do DataLoader data_dir
        if not path:
            path = self.dl.data_dir

        os.makedirs(path, exist_ok=True)

        # for each split, save as json
        for key in self.splits.keys():
            self._save_json_list(self._splits_data[key], os.path.join(path, key+".json"), add=add, sub_annots=True)

    def _save_json_list(self, data, file, add=True, sub_annots=False):
        # either overwrite file or add to it
        if add:
            mode = "a+"
        else:
            mode = "w"

        # open file
        with open(file, mode) as f:
            # deepcopy data so don't change original
            for l in copy.deepcopy(data):
                # map category_ids to actual labels
                if sub_annots:
                    for annot in l.get("annotations", []):
                        annot["category_id"] = self._labels[annot["category_id"]]
                json.dump(l, f)
                f.write("\n")


if __name__ == '__main__':
    dl = DataLoader()
