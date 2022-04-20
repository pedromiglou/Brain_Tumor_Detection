import glob
import cv2
from keras.preprocessing.image import ImageDataGenerator
from os import listdir
import cv2
import imutils    
import matplotlib.pyplot as plt
import numpy as np
from sklearn.utils import shuffle #shuffling the data improves the model
from sklearn.model_selection import train_test_split
import tensorflow as tf


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

    return np.array(images), np.array(labels)


def crop_images(images):
    new_images = []
    
    for image in images:
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
        new_image = image[extTop[1]:extBot[1], extLeft[0]:extRight[0]]
    
        new_images.append(new_image)
    
    return np.array(new_images)


def resize_and_rescale(images, IMG_SIZE=224):    
    new_images = []

    for image in images:
        #resize to be smaller to have less data
        new_image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))

        #normalize data
        new_image = new_image/255.0
    
        new_images.append(new_image)
    
    return np.array(new_images)


def split_data(X, y, test_size=0.2):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)

    return X_train, y_train, X_test, y_test

def augment_data(X,y, IMG_SIZE=224):
    data_augmentation = tf.keras.Sequential([
      tf.keras.layers.experimental.preprocessing.RandomFlip("horizontal_and_vertical"),
      tf.keras.layers.experimental.preprocessing.RandomRotation(0.2),
    ])
    
    new_images = []
    new_labels = []
    
    for image, label in zip(X,y):
        new_images.append(image)
        new_labels.append(label)
        
        new_images.append(np.reshape(data_augmentation(tf.expand_dims(X[0], 0)), (IMG_SIZE,IMG_SIZE)))
        new_labels.append(label)
    
    return np.array(new_images), np.array(new_labels) 


"""
def augment_data(file_dir, n_generated_samples, save_to_dir):
    data_gen = ImageDataGenerator(rotation_range=10, 
                                  width_shift_range=0.1, 
                                  height_shift_range=0.1, 
                                  shear_range=0.1, 
                                  brightness_range=(0.3, 1.0),
                                  horizontal_flip=True, 
                                  vertical_flip=True, 
                                  fill_mode='nearest'
                                 )

    for filename in listdir(file_dir):
        image = cv2.imread(file_dir + '/' + filename)
        # reshape the image
        image = image.reshape((1,)+image.shape)
        save_prefix = 'aug_' + filename[:-4]
        i=0
        for batch in data_gen.flow(x=image, batch_size=1, save_to_dir=save_to_dir,save_prefix=save_prefix, save_format='jpg'):
                i += 1
                if i > n_generated_samples:
                    break





def load_data(dir_list, image_size):

    # load all images in a directory
    X = []
    y = []
    image_width, image_height = image_size
    
    for directory in dir_list:
        for filename in listdir(directory):
            image = cv2.imread(directory+'/'+filename)
            image = crop_brain_contour(image, plot=False)
            image = cv2.resize(image, dsize=(image_width, image_height), interpolation=cv2.INTER_CUBIC)
            # normalize values
            image = image / 255.
            # convert image to numpy array and append it to X
            X.append(image)
            # append a value of 1 to the target array if the image
            # is in the folder named 'yes', otherwise append 0.
            if directory[-3:] == 'yes':
                y.append([1])
            else:
                y.append([0])
                
    X = np.array(X)
    y = np.array(y)
    
    # Shuffle the data
    X, y = shuffle(X, y)
    
    print(f'Number of examples is: {len(X)}')
    print(f'X shape is: {X.shape}')
    print(f'y shape is: {y.shape}')
    
    return X, y


def plot_sample_images(X, y, n=40):
    for label in [0,1]:
        # grab the first n images with the corresponding y values equal to label
        images = X[np.argwhere(y == label)]
        n_images = images[:n]
        
        columns_n = 10
        rows_n = int(n/ columns_n)

        plt.figure(figsize=(10, 8))
        
        i = 1 # current plot        
        for image in n_images:
            plt.subplot(rows_n, columns_n, i)
            plt.imshow(image[0])
            
            # remove ticks
            plt.tick_params(axis='both', which='both', 
                            top=False, bottom=False, left=False, right=False,
                           labelbottom=False, labeltop=False, labelleft=False, labelright=False)
            
            i += 1
        
        label_to_str = lambda label: "Yes" if label == 1 else "No"
        plt.suptitle(f"Brain Tumor: {label_to_str(label)}")
        plt.show()





def plot_metrics(history):
    
    train_loss = history['loss']
    val_loss = history['val_loss']
    train_acc = history['accuracy']
    val_acc = history['val_accuracy']
    
    # Loss
    plt.figure()
    plt.plot(train_loss, label='Training Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.title('Loss')
    plt.legend()
    plt.show()
    
    # Accuracy
    plt.figure()
    plt.plot(train_acc, label='Training Accuracy')
    plt.plot(val_acc, label='Validation Accuracy')
    plt.title('Accuracy')
    plt.legend()
    plt.show()
"""