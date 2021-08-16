# install python 3.6 (conda create -n cm python=3.6)
echo "installing pytorch and detectron2...";
conda install -y pytorch torchvision torchaudio -c pytorch;
conda install -y -c conda-forge detectron2;
echo "installing GDAL...";
conda install -y -c conda-forge libgdal;
conda install -y -c conda-forge gdal;
echo "installing remaining packages..."
conda install -y -c conda-forge yacs;
conda install -y -c iopath iopath;
conda install -y pandas;
yes | pip install opencv-python;
conda install -y -c conda-forge omegaconf;
conda install -y -c conda-forge geopandas;
conda install -y -c omnia termcolor;
conda install -y -c anaconda ipython;
conda install -y -c anaconda xlrd;
