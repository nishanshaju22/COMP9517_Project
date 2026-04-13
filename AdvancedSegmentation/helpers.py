import numpy as np
import glob
import os
import cv2
import albumentations as A
import matplotlib.pyplot as plt

# Obtain wanted statistics for an image mask and its truth values
def get_stats(mask, label):
    tp = np.sum((mask == 0) & (label == 0)).astype(float)
    fp = np.sum((mask == 0) & (label == 255)).astype(float)
    fn = np.sum((mask == 255) & (label == 0)).astype(float)
    tn = np.sum((mask == 255) & (label == 255)).astype(float)
    acc = (tp + tn) / (tp + fp + fn + tn)
    recall = tp / (tp + fn)
    precision = tp / (tp + fp)
    if (tp + fp) == 0:
        precision = 0
    IoU = tp / (tp + fp + fn)
    f1 = 2 * precision * recall / (precision + recall)
    return acc, recall, precision, IoU, f1


# Function that outputs images and labels given an array of paths
def get_imgs(path):
    data = glob.glob(os.path.join(path, "*.png"))
    imgs = [x for x in data if "mask" not in x]
    labels = [x for x in data if "mask" in x]
    return imgs, labels

# Function that averages an array of stats
def average_stats(tst_stats):
    total_acc = np.mean(tst_stats[:,0])
    total_recall = np.mean(tst_stats[:,1])
    total_precision = np.mean(tst_stats[:,2])
    total_IoU = np.mean(tst_stats[:,3])
    total_f1 = np.mean(tst_stats[:,4])
    return [total_acc, total_recall, total_precision, total_IoU, total_f1]


# Function used to print results
def printer(string, stats):
    print(f"{string} Test Accuracy: {stats[0]}")
    print(f"{string} Test Recall: {stats[1]}")
    print(f"{string} Test Precision: {stats[2]}")
    print(f"{string} Test IoU: {stats[3]}")
    print(f"{string} Test F1 score: {stats[4]}")


# Optional Function used to display results for viewing.
def showResult(image_index, images, labels, masks, processed):
    plt.figure(figsize=(10, 2.5))
    plt.subplot(1, 4, 1)
    plt.title("Original")
    images[image_index] = cv2.cvtColor(images[image_index], cv2.COLOR_BGR2RGB)
    plt.imshow(images[image_index], cmap="grey")
    plt.subplot(1, 4, 2)
    plt.title("Labels")
    plt.imshow(labels[image_index], cmap="grey")
    plt.subplot(1, 4, 3)
    plt.title("Mask")
    plt.imshow(masks[image_index], cmap="grey")
    plt.subplot(1, 4, 4)
    plt.title("Post-Processed")
    plt.imshow(processed[image_index], cmap="grey")