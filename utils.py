import glob
import cv2

def read_data(IMG_SIZE=224, color=False):
    yes_dir = 'brain_tumor_dataset/yes'
    no_dir = 'brain_tumor_dataset/no'
    data = []
    
    if color:
        flag = cv2.IMREAD_COLOR
    else:
        flag = cv2.IMREAD_GRAYSCALE
    
    for filename in glob.iglob(f'{yes_dir}/*'):
        img_array = cv2.imread(filename, flag)
        
        #resize to be smaller to have less data
        new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))

        #normalize data
        new_array = new_array/255.0

        data.append([new_array, 1])
    
    for filename in glob.iglob(f'{no_dir}/*'):
        img_array = cv2.imread(filename, flag)
        
        #resize to be smaller to have less data
        new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))

        #normalize data
        new_array = new_array/255.0

        data.append([new_array, 0])

    return data
