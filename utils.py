import glob
import cv2
import imutils    
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from sklearn.metrics import confusion_matrix


def read_images():
    yes_dir = 'brain_tumor_dataset/yes'
    no_dir = 'brain_tumor_dataset/no'
    images = []
    labels = []
    
    for filename in glob.iglob(f'{yes_dir}/*'):
        img_array = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)

        images.append(img_array)
        labels.append(1)
    
    for filename in glob.iglob(f'{no_dir}/*'):
        img_array = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)

        images.append(img_array)
        labels.append(0)

    return images, np.array(labels)


def crop_images(images, IMG_SIZE=None):
    for i in range(len(images)):
        image = images[i]
        
        # blur the image slightly
        blurred = cv2.GaussianBlur(image, (5, 5), 0)

        thresh = cv2.threshold(blurred, 45, 255, cv2.THRESH_BINARY)[1]
        thresh = cv2.erode(thresh, None, iterations=2)
        thresh = cv2.dilate(thresh, None, iterations=2)

        # Find contours in thresholded image, then grab the largest one
        cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        c = max(cnts, key=cv2.contourArea)
        
        # extreme points
        extLeft = tuple(c[c[:, :, 0].argmin()][0])
        extRight = tuple(c[c[:, :, 0].argmax()][0])
        extTop = tuple(c[c[:, :, 1].argmin()][0])
        extBot = tuple(c[c[:, :, 1].argmax()][0])

        # crop new image out of the original image using the four extreme points (left, right, top, bottom)
        # if IMG_SIZE is provided, it will resize the image and assume the input was a np array
        if IMG_SIZE:
            images[i,:,:] = cv2.resize(image[extTop[1]:extBot[1], extLeft[0]:extRight[0]], (IMG_SIZE, IMG_SIZE))
        else:
            images[i] = image[extTop[1]:extBot[1], extLeft[0]:extRight[0]]
    
    return images


def resize_and_rescale(images, IMG_SIZE=150, rescale=True):
    for i in range(len(images)):
        #resize to be smaller to have less data
        images[i] = cv2.resize(images[i], (IMG_SIZE, IMG_SIZE))

        #normalize data
        if rescale:
            images[i] = images[i]/255.0
    
    # convert to numpy array now that all images have the same size
    return np.array(images)


def split_and_shuffle(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25)

    return X_train, y_train, X_val, y_val, X_test, y_test


def augment_data(X,y, IMG_SIZE=224):
    data_augmentation = tf.keras.Sequential([
      tf.keras.layers.experimental.preprocessing.RandomFlip("horizontal"),
      tf.keras.layers.experimental.preprocessing.RandomRotation(0.2),
    ])
    
    new_images = []
    new_labels = []
    
    for image, label in zip(X,y):
        new_images.append(image)
        new_labels.append(label)
        
        new_images.append(np.reshape(data_augmentation(tf.expand_dims(image, 0), training=True), image.shape))
        new_labels.append(label)
    
    return np.array(new_images), np.array(new_labels)


def plot_accuracy_comparison(accs, title, legend):
    epochs = len(accs[0])
    plt.figure(figsize = (10,5))
    for acc in accs:
        plt.plot(range(1, epochs+1), acc)

    plt.xticks(range(1, epochs+1))
    plt.title(title)
    plt.legend(legend)
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.show()


def plot_loss_comparison(losses, title, legend):
    epochs = len(losses[0])
    plt.figure(figsize = (10,5))
    for loss in losses:
        plt.plot(range(1, epochs+1), loss)

    plt.xticks(range(1, epochs+1))
    plt.title(title)
    plt.legend(legend)
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.show()


def plot_confusion_matrix(y_test, y_pred):
    plt.matshow(confusion_matrix(y_test, y_pred))
    plt.ylabel("Predicted Category", fontsize=14)
    plt.title("Category", fontsize=14)
    plt.show()
