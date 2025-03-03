import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from scipy.io import loadmat
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import pearsonr
import openpyxl
import os
# from keras.models import load_model
from my_package import MyUtil
from my_package import CustomCallback
# from flexible_robot.Self_Att.Self_Attention.self_attention_layers import Self_Attention
# from flexible_robot.Co_Att.Co_Attention.co_attention_layers import CoAttentionParallel


# Get the directory where the .py file is located
current_path = os.path.dirname(__file__)

# Set this directory as the working directory
os.chdir(current_path)

# Load data
dataMat = loadmat("data0-15.mat")

### Training set
trainSet = ["gujiqian0_1", "gujiqian0_2", "gujiqian5_1", "gujiqian5_2"]
### Validation set
validationSet = ["gujiqian10_1", "gujiqian10_2", "gujiqian10_3", "gujiqian10_4"]
### Test set
testSet = ["gujiqian15_1", "gujiqian15_2", "gujiqian15_3", "gujiqian15_4"]

### Basic parameter settings
block_length = 50
train_feature_stride = 10
train_label_stride = 10

validation_feature_stride = 10
validation_label_stride = 10
# For the test set, values for every time point need to be predicted, so the stride is set to 1
test_feature_stride = 1
test_label_stride = 1

# print(total_timepoints)#5000


# Columns 8 to 10 contain the x-displacement information (x1, x2, x3) of three markers in mm,
# and columns 11 to 13 contain the y-displacement information (y1, y2, y3) of three markers in mm.
# Column 14 is the actual bending angle information of the flexible joint in degrees.
# Column 17 is the theoretical input angle information in degrees.

# feature_column = [7, 8, 9, 10, 11, 12, 13]

"""Use columns 2 to 7 and column 17 as features"""
# Columns 2 to 4 are the three force measurements at the end-effector from the 6D force sensor, in N.
# Columns 5 to 7 are the three moment measurements at the end-effector from the 6D force sensor, in N·M.
# Column 17 is the theoretical input angle information.
feature_column = [1, 2, 3, 4, 5, 6, 16]

# Columns 8 to 10 are the x-displacement information (x1, x2, x3) of three markers in mm,
# and columns 11 to 13 are the y-displacement information (y1, y2, y3) of three markers in mm.
# Column 14 is the actual bending angle information of the flexible joint.
label_column = [7, 8, 9, 10, 11, 12, 13]


def getSamples(set_name=[], feature_stride=1, label_stride=1):
    features = []
    labels = []
    for matrixName in set_name:
        data = dataMat[matrixName]
        total_timepoints = len(data[:, 0])
        numberOfSamples = int((total_timepoints - block_length) / feature_stride) + 1
        # print("Each .csv file can produce {} samples".format(numberOfSamples))  # 496 samples
        # Construct the training sample features and labels
        for i in range(numberOfSamples):
            data_temp = data[(0 + i * feature_stride):(0 + i * feature_stride + block_length), feature_column]
            features.append(data_temp)

            data_temp = data[(0 + i * label_stride + block_length - 1), label_column]
            labels.append(data_temp)
    features = np.array(features)
    labels = np.array(labels)

    return features, labels


trainSampleFeature, trainSampleLabel = getSamples(set_name=trainSet, feature_stride=train_feature_stride,
                                                  label_stride=train_label_stride)
# print(trainSampleFeature.shape)  # Training set: 496*4=1984 samples (1984, 50, 7); 496*4 = 1984 samples
# print(trainSampleLabel.shape)    # Training set: 496*4=1984 samples (1984, 7)

# print("First sample:", trainSampleFeature[0])
# print("First label:", trainSampleLabel[0])

validationSampleFeature, validationSampleLabel = getSamples(set_name=validationSet,
                                                            feature_stride=validation_feature_stride,
                                                            label_stride=validation_label_stride)
# print(validationSampleFeature.shape)  # Validation set: 496*4=1984 samples (1984, 50, 7)
# print(validationSampleLabel.shape)    # Validation set: 496*4=1984 samples (1984, 7)

testSampleFeature, testSampleLabel = getSamples(set_name=testSet,
                                                feature_stride=test_feature_stride,
                                                label_stride=test_label_stride)

# print("Test set feature shape: {}".format(testSampleFeature.shape))  # Test set feature shape (19804, 50, 7)
# print("Test set label shape: {}".format(testSampleLabel.shape))      # Test set label shape (19804, 7)


## Data processing: Normalize the training sample features; the feature shape is flattened accordingly
# Flatten sample features to 1D arrays
trainSampleFeature = trainSampleFeature.reshape(trainSampleFeature.shape[0], 350)
validationSampleFeature = validationSampleFeature.reshape(validationSampleFeature.shape[0], 350)
testSampleFeature = testSampleFeature.reshape(testSampleFeature.shape[0], 350)

# Perform normalization
min_max_scaler = MinMaxScaler()
x_train = min_max_scaler.fit_transform(trainSampleFeature)
x_validation = min_max_scaler.transform(validationSampleFeature)
x_test = min_max_scaler.transform(testSampleFeature)
##

## Sample labels
# y_train = trainSampleLabel
# y_validation = validationSampleLabel

### Training set x1 coordinate
label_x1_train = trainSampleLabel[:, 0]
### Training set x2 coordinate
label_x2_train = trainSampleLabel[:, 1]
### Training set x3 coordinate
label_x3_train = trainSampleLabel[:, 2]

# Training set y1 coordinate
label_y1_train = trainSampleLabel[:, 3]
# Training set y2 coordinate
label_y2_train = trainSampleLabel[:, 4]
# Training set y3 coordinate
label_y3_train = trainSampleLabel[:, 5]
# Training set angle
label_angle_train = trainSampleLabel[:, 6]

### Validation set x1 coordinate
label_x1_validation = validationSampleLabel[:, 0]
### Validation set x2 coordinate
label_x2_validation = validationSampleLabel[:, 1]
### Validation set x3 coordinate
label_x3_validation = validationSampleLabel[:, 2]
### Validation set y1 coordinate
label_y1_validation = validationSampleLabel[:, 3]
### Validation set y2 coordinate
label_y2_validation = validationSampleLabel[:, 4]
### Validation set y3 coordinate
label_y3_validation = validationSampleLabel[:, 5]
### Validation set angle
label_angle_validation = validationSampleLabel[:, 6]

### Test set labels
label_x1_test = testSampleLabel[:, 0]
label_x2_test = testSampleLabel[:, 1]
label_x3_test = testSampleLabel[:, 2]
label_y1_test = testSampleLabel[:, 3]
label_y2_test = testSampleLabel[:, 4]
label_y3_test = testSampleLabel[:, 5]
label_angle_test = testSampleLabel[:, 6]
