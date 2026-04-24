# COMP9517_Project

To setup this project:
Download the EWS_DATASET from the following link: https://www.research-collection.ethz.ch/entities/researchdata/165d22fc-6b0f-4fc3-a441-20d8bdc50a70
Create a new folder called "Data"  on the  root folder of the project (same folder as this README) and extract the ZIP file into a folder called "EWS-Dataset" in that "Data" folder.

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



Advanced Segmentation

Watershed algorithm (OpenCV): https://docs.opencv.org/4.x/d3/db4/tutorial_py_watershed.html
Watershed tutorial: https://www.geeksforgeeks.org/image-segmentation-with-watershed-algorithm-opencv-python/
Felzenszwalb & scikit-image segmentation: https://scikit-image.org/docs/stable/api/skimage.segmentation.html
scikit-image tutorial: https://www.geeksforgeeks.org/image-segmentation-using-pythons-scikit-image-module/

Machine Learning

Random Forest (scikit-learn): https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html
Random Forest user guide: https://scikit-learn.org/stable/modules/ensemble.html#forest
XGBoost classifier: https://xgboost.readthedocs.io/en/release_3.2.0/

Deep Learning

Segmentation models PyTorch (U-Net, DeepLabV3+): https://github.com/qubvel/segmentation_models.pytorch
PyTorch deep learning reference: https://github.com/mrdbourke/pytorch-deep-learning
Albumentations (data augmentation): https://albumentations.ai/docs/3-basic-usage/semantic-segmentation/
Adam optimiser: https://arxiv.org/abs/1412.6980

Utilities

Python timing (perf_counter): https://www.geeksforgeeks.org/python/time-perf_counter-function-in-python/
