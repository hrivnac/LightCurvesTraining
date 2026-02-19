FROM tensorflow/tensorflow
RUN python -m pip install --no-cache-dir pandas matplotlib
WORKDIR /home/hrivnac/work/LSST/LightCurvesTrainig
