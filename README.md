## Faster R-CNN for Oil and Gas infrastructure Detection in the Permian Basin

This repository contains code to train, evaluate, and run a Faster R-CNN model. It currently has been trained on data from a GAO survey in the Permian Basin (dataset not included).

# Installation

Create conda Python 3.6 environment

```
conda create --name <name> python=3.6
conda activate <name>
```

Running `./setup.sh` should install all required packages.

Any missing packages should be installed using `conda-forge` if possible, then `conda`, then finally `pip`, ensuring your pip is within your conda environment (`which pip` should show pip being installed at something like `/path/to/anaconda3/envs/<name>/bin/pip`)

# Repository architecture
