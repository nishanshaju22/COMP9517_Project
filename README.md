# COMP9517_Project

To setup this project:
Download the EWS_DATASET from the following link: https://www.research-collection.ethz.ch/entities/researchdata/165d22fc-6b0f-4fc3-a441-20d8bdc50a70
Extract the ZIP file into a folder called "EWS-Dataset" and move this folder into the root folder of the project (same folder as this README)

To run the code, startup jupyter notebook and access the notebooks in the corresponding folder:
AdvancedSegmentation: watershed.ipynb, felzenszwalb.ipynb
DeepLabV3: Model.ipynb
XGBoost: hyperparam_xgb.ipynb, model_xgb.ipynb
randomForest: hyper_param_rf.ipynb, model_rf.ipynb
unet: unetTrain.ipynb

Then run each notebook to obtain results.


Utilised Libraries and Code:
numpy
cv2
albumentations
pathlib
matplotlib.pyplot
glob
os
time
skimage.segmentation
sys
random
PIL
tqdm
torch
torchvision
sklearn
json
itertools
typing
joblib


Code references:
https://docs.opencv.org/4.x/d3/db4/tutorial_py_watershed.html
https://www.geeksforgeeks.org/computer-vision/image-segmentation-with-watershed-algorithm-opencv-python/
https://www.geeksforgeeks.org/python/time-perf_counter-function-in-python/
https://albumentations.ai/docs/3-basic-usage/semantic-segmentation/
https://scikit-image.org/docs/stable/api/skimage.segmentation.html
https://www.geeksforgeeks.org/machine-learning/image-segmentation-using-pythons-scikit-image-module/
https://github.com/qubvel/segmentation_models.pytorch
https://arxiv.org/abs/1412.6980
https://github.com/mrdbourke/pytorch-deep-learning
https://xgboost.readthedocs.io/en/release_3.2.0/
