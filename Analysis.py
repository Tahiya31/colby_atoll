'''Analysis.py
Provides Summary Statistics for Segmented Images
Ray Wang
Atoll Project
Spring 2023
'''

import os
import numpy as np
from skimage.io import imread
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from skimage.transform import resize
import seaborn as sns
import cv2


class ImageAnalysis:

    def __init__(self, num_classes, mask_array_1, mask_array_2):
        self.num_classes = num_classes
        self.mask_array_1 = mask_array_1  # Now a list of numpy arrays
        self.mask_array_2 = mask_array_2  # Now a list of numpy arrays
        self.ious = []
        self.confusion_matrices = []
        self.accuracy = None
        self.F1 = None


    @staticmethod
    def iou_score(y_true, y_pred, num_classes):
        y_true = y_true.astype(np.int32)
        y_pred = y_pred.astype(np.int32)

        conf_matrix = confusion_matrix(y_true.flatten(), y_pred.flatten(), labels=np.arange(num_classes))

        intersection = np.diag(conf_matrix)
        ground_truth_set = conf_matrix.sum(axis=1)
        predicted_set = conf_matrix.sum(axis=0)
        union = ground_truth_set + predicted_set - intersection

        iou = intersection / (union.astype(np.float32) + np.finfo(np.float32).eps)
        return iou, conf_matrix

    
    def compute_ious(self):
        self.ious = []
        self.confusion_matrices = []

        # We no longer need to load images from files
        for mask_1, mask_2 in zip(self.mask_array_1, self.mask_array_2):

            iou, conf_matrix = self.iou_score(mask_1, mask_2, self.num_classes)

            self.ious.append(iou)
            self.confusion_matrices.append(conf_matrix)

    def get_mean_ious(self):
        mean_ious = np.mean(self.ious, axis=0)
        overall_mean_iou = np.mean(mean_ious)
        return mean_ious, overall_mean_iou


    def plot_confusion_matrix(self):
        sum_conf_matrix = np.sum(self.confusion_matrices, axis=0)
        plt.figure(figsize=(8, 6))

        # Compute the percentage matrix
        percentage_matrix = sum_conf_matrix / np.sum(sum_conf_matrix)

        # Combine the original matrix and the percentage matrix for annotation
        annot_matrix = [ ['{0} ({1:.2%})'.format(int(value), percentage_matrix[i, j]) 
                        for j, value in enumerate(row)] 
                        for i, row in enumerate(sum_conf_matrix) ]

        sns.heatmap(sum_conf_matrix, annot=annot_matrix, fmt='', cmap=plt.cm.Purples, cbar=False)
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.title('Confusion Matrix')
        plt.show()



    @staticmethod
    def compute_accuracy_score(y_true, y_pred):
        y_true = y_true.astype(np.int32).flatten()
        y_pred = y_pred.astype(np.int32).flatten()

        return accuracy_score(y_true,y_pred)

    def compute_accuracy(self):
        accuracies = []

        # We no longer need to load images from files
        for mask_1, mask_2 in zip(self.mask_array_1, self.mask_array_2):

            accuracy = self.compute_accuracy_score(mask_1, mask_2)
            accuracies.append(accuracy)

        mean_accuracy = np.mean(accuracies)
        self.accuracy = mean_accuracy

        return mean_accuracy
    @staticmethod
    def compute_f1_score(gt_mask, pred_mask):
   
        gt_mask = gt_mask.astype(np.int32).flatten()
        pred_mask = pred_mask.astype(np.int32).flatten()
      
        f1 = f1_score(gt_mask,pred_mask, average = None)
   
        return f1

    def compute_f1(self):
        f1_scores = []

        # We no longer need to load images from files
        for gt_array, pred_array in zip(self.mask_array_1, self.mask_array_2):

            f1 = self.compute_f1_score(gt_array, pred_array)
            f1_scores.append(f1)

        average_f1 = np.mean(f1_scores, axis = 0)

        if len(average_f1) == 4:
            average_f1 = average_f1[:3]

        average_f1 = np.mean(average_f1)

        self.F1 = average_f1

        return average_f1
