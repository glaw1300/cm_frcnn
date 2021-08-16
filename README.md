# Faster R-CNN for Oil and Gas infrastructure Detection in the Permian Basin

This repository contains code to train, evaluate, and run a Faster R-CNN model. It currently has been trained on data from a GAO survey in the Permian Basin (dataset not included).

## Installation

Create conda Python 3.6 environment

```
conda create --name <name> python=3.6
conda activate <name>
```

Running `./setup.sh` should install all required packages.

Any missing packages should be installed using `conda-forge` if possible, then `conda`, then finally `pip`, ensuring your pip is within your conda environment (`which pip` should show pip being installed at something like `/path/to/anaconda3/envs/<name>/bin/pip`)

## Running

For simple evaluation and source attribution, running the scripts  `main.py` or `source_attribution.py` will work as the directory is currently configured. `main.py` accepts a host of flags that can be investigated by running `python main.py --help`.

Additionally, running `source_attribution.py` will require a plume list and data that is not provided.

## Repository architecture
As it stands, the repository runs the model with results listed in the **Results** section below. The default `DataLoader` class is currently configured to read information as it is organized in the directory hierarchy below. You can customize DataLoader and add your own data as you see fit, but the general hierarchy should be preserved.

```
cm-frcnn
│   README.md
│   main.py # for customizing and running the model
|   setup.sh # installs all relevant packages
|   
|   # any custom scripts you'd like to overwrite for the model (CustomTrainer, CustomEvaluator)
│
└─── scripts
│   │   faster_rcnn.py
│   │   trainer.py
|   |   evaluator.py
|   |   predictor.py
|   |   dataloader.py
│   │
│   └───utils
│       │   utils.py # miscellaneous utilities
│       │   plotter.py # plotting utilities
|   
└─── data # where the annotation and model data is stored (used by DataLoader)
|   └───annotations # annotations data
|   |   |   labels.txt # model labels
|   |   |   data.json # concise version of all annotations
|   |   |   train.json # training data annotations as json
|   |   |   test.json # test data annotations as json
|   |   |   valid.json # validation data annotations as json
|   |   |
|   |   └───csvs
|   |   |   |   ... all annotations from MakeSense.ai labeling in CSV files
|   |   |   # format:
|   |   |
|   |   └───yolo
|   |   |   ... all annotations from MakeSense.ai labeling in YOLO format, for loading in annotations and editing
|   |   
|   └───model # model config and
|   |   |   cfg.yaml # model config file
|   |   |   model_final.pth # model weights

```

## Results

To come...
