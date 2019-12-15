import numpy as np
import os
import sys
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import torch.distributed as dist
import torchvision

from helperFunctions import getUCF101
from helperFunctions import loadFrame

import h5py
import cv2

from multiprocessing import Pool
###############################
IMAGE_SIZE = 224
NUM_CLASSES = 101
batch_size = 100
lr = 0.0001
num_of_epochs = 10

data_directory = '/projects/training/bayw/hdf5/'
class_list, train, test = getUCF101(base_directory = data_directory)
###############################
# load the confusion matrix
confusion_matrix = np.load('3d_resnet_matrix.npy')

results = np.diag(confusion_matrix)
indices = np.argsort(results)

number_of_examples = np.sum(confusion_matrix,axis=1)

sorted_list = np.asarray(class_list)
sorted_list = sorted_list[indices]
sorted_results = results[indices]

# sorted_list is the list of classes in order from lowest to highest probability of recognition
# sorted_results is the list of probabilities that correspond to the actual class being identified correctly
lowest_classes = sorted_list[0:10]
highest_classes = sorted_list[-10:]

print('highest classes: {}'.format(highest_classes))
print('lowest classes: {}'.format(lowest_classes))
# for most confused classes

confuse_array = np.zeros([3, 10])

for class_idx in range(len(class_list)):
    # get a list of all probs for each class
    class_probs = confusion_matrix[class_idx, :]

    class_probs[class_idx] = 0

    # get the max probability
    confused_class_prob = np.max(class_probs)
    confused_class_idx = np.argmax(class_probs)
    
    if class_idx != confused_class_idx:
        if confused_class_prob > np.min(confuse_array[2,:]):
            print(confused_class_prob)
            print(np.min(confuse_array[2,:]))
            replace_idx = np.argmin(confuse_array[2,:])
            confuse_array[0, replace_idx] = class_idx
            confuse_array[1, replace_idx] = confused_class_idx
            confuse_array[2, replace_idx] = confused_class_prob

print(confuse_array)
actual_classes = []
confuse_classes = []
confuse_class_prob = []

for i in range(10):
    actual_classes.append(class_list[int(confuse_array[0, i])])
    confuse_classes.append(class_list[int(confuse_array[1, i])])
    confuse_class_prob.append(confuse_array[2, i])

print(actual_classes)
print(confuse_classes)
