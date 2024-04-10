# Import the required libraries.
import os
import cv2
#import pafy
import math
import random
import numpy as np
import datetime as dt
from collections import deque
from tensorflow.keras.utils import to_categorical


import os
os.environ['PYTHONHASHSEED'] = '0'
os.environ['CUDA_VISIBLE_DEVICES'] = ''

seed_constant = 27
np.random.seed(seed_constant)
random.seed(seed_constant)

# Create a Matplotlib figure and specify the size of the figure.
#plt.figure(figsize = (20, 20))

# Get the names of all classes/categories in UCF50.
TRAIN_DIR = r'train'
VAL_DIR = r'val'
TEST_DIR = r'test'
all_classes_names = sorted(os.listdir(TRAIN_DIR))
print(all_classes_names)

# Specify the height and width to which each video frame will be resized in our dataset.
IMAGE_HEIGHT , IMAGE_WIDTH = 80, 80

# Specify the directory containing the UCF50 dataset. 
#DATASET_DIR = "/newdatasetMediapipe/kata/"

# Specify the list containing the names of the classes used for training. Feel free to choose any set of classes.
CLASSES_LIST =all_classes_names
#CLASSES_LIST = ['2 Tolong', '7 permisi', '1 Maaf']


# Specify the directory containing the UCF50 dataset. 
#DATASET_DIR = "/newdatasetMediapipe/kata/"

# Specify the list containing the names of the classes used for training. Feel free to choose any set of classes.
CLASSES_LIST =all_classes_names
#CLASSES_LIST = ['2 Tolong', '7 permisi', '1 Maaf']

def getAverageSequenceLength(DATASET_DIR):
    sum = 0
    c = 0
    for class_index, class_name in enumerate(CLASSES_LIST):    
        # Display the name of the class whose data is being extracted.
        print(f'Extracting Data of Class: {class_name}')
        
        # Get the list of video files present in the specific class name directory.
        files_list = os.listdir(os.path.join(DATASET_DIR, class_name))
        
        # Iterate through all the files present in the files list.
        for file_name in files_list:
            
            # Get the complete video path.
            video_file_path = os.path.join(DATASET_DIR, class_name, file_name)
            # Read the Video File using the VideoCapture object.
            video_reader = cv2.VideoCapture(video_file_path)
            # Get the total number of frames in the video.
            video_frames_count = int(video_reader.get(cv2.CAP_PROP_FRAME_COUNT))
            video_reader.release()
            sum+=video_frames_count
            c+=1
    
    return sum//c


def frames_extraction(video_path):
    '''
    This function will extract the required frames from a video after resizing and normalizing them.
    Args:
        video_path: The path of the video in the disk, whose frames are to be extracted.
    Returns:
        frames_list: A list containing the resized and normalized frames of the video.
    '''

    # Declare a list to store video frames.
    frames_list = []
    
    # Read the Video File using the VideoCapture object.
    video_reader = cv2.VideoCapture(video_path)

    # Get the total number of frames in the video.
    video_frames_count = int(video_reader.get(cv2.CAP_PROP_FRAME_COUNT))//2
    minFrame = video_frames_count-SEQUENCE_LENGTH//2
    maxFrame = video_frames_count+SEQUENCE_LENGTH//2

    #print("count="+str(video_frames_count))
    #print(video_frames_count)

    # Calculate the the interval after which frames will be added to the list.
    #skip_frames_window = max(int(video_frames_count/SEQUENCE_LENGTH), 1)
    #print("interval="+str(skip_frames_window))
    #print(skip_frames_window)
    
    # Iterate through the Video Frames.
    #for frame_counter in range(0,SEQUENCE_LENGTH,2):
    for frame_counter in range(minFrame,maxFrame):

        # Set the current frame position of the video.
        #video_reader.set(cv2.CAP_PROP_POS_FRAMES, frame_counter * skip_frames_window)
        video_reader.set(cv2.CAP_PROP_POS_FRAMES, frame_counter)

        
        # Reading the frame from the video. 
        success, frame = video_reader.read() 

        # Check if Video frame is not successfully read then break the loop
        if not success:
            break

        # Resize the Frame to fixed height and width.
        resized_frame = cv2.resize(frame, (IMAGE_HEIGHT, IMAGE_WIDTH))
        
        # Normalize the resized frame by dividing it with 255 so that each pixel value then lies between 0 and 1
        normalized_frame = resized_frame / 255
        # Append the normalized frame into the frames list
        frames_list.append(normalized_frame)
        # Convert to integer data type pixels.
        
        #print(frames_list[0].shape)
    #pad_array = np.frompyfunc(zeroMatrix.copy, 0, 1)(np.empty((7, 7, 7), object))
    pad_array = [np.zeros((IMAGE_HEIGHT, IMAGE_WIDTH, 3))]                            
    frames_list.extend(pad_array * (SEQUENCE_LENGTH - len(frames_list)))
    # Release the VideoCapture object. 
    video_reader.release()
    
    #for l in range(0,5):
    #    plt.subplot(6, 20, l+1)
    #    plt.imshow(frames_list[l])
    #    plt.axis('off')

    # Return the frames list.
    #print(frames_list)
    return frames_list

#frames_extraction(r"C:\Users\Sukasenyumm\Downloads\MIRACL-VC1\dataset\dataset\kata\1outfinal\data (145).mp4")

def create_dataset(DATASET_DIR):
    '''
    This function will extract the data of the selected classes and create the required dataset.
    Returns:
        features:          A list containing the extracted frames of the videos.
        labels:            A list containing the indexes of the classes associated with the videos.
        video_files_paths: A list containing the paths of the videos in the disk.
    '''

    # Declared Empty Lists to store the features, labels and video file path values.
    features = []
    labels = []
    #video_files_paths = []
    # Iterating through all the classes mentioned in the classes list
    for class_index, class_name in enumerate(CLASSES_LIST):
        
        # Display the name of the class whose data is being extracted.
        print(f'Extracting Data of Class: {class_name}')
        
        # Get the list of video files present in the specific class name directory.
        files_list = os.listdir(os.path.join(DATASET_DIR, class_name))
        
        # Iterate through all the files present in the files list.
        for file_name in files_list:
            
            # Get the complete video path.
            video_file_path = os.path.join(DATASET_DIR, class_name, file_name)

            # Extract the frames of the video file.
            frames = frames_extraction(video_file_path)
            #if len(frames) < SEQUENCE_LENGTH:
            #    print("ignore"+file_name+"::"+str(len(frames)))

            # Check if the extracted frames are equal to the SEQUENCE_LENGTH specified above.
            # So ignore the vides having frames less than the SEQUENCE_LENGTH.
            if len(frames) == SEQUENCE_LENGTH:

                # Append the data to their repective lists.
                features.append(frames)
                labels.append(class_index)
                #video_files_paths.append(video_file_path)
        

    # Converting the list to numpy arrays
    features = np.asarray(features)
    labels = np.array(labels)  
    assert len(features) == len(labels)
    p = np.random.permutation(len(features))
    
    
    # Return the frames, class index, and video file path.
    return features[p], labels[p]#, video_files_paths

# Create the dataset.
#features, labels, video_files_paths = create_dataset()
# Specify the number of frames of a video that will be fed to the model as one sequence.
SEQUENCE_LENGTH = 40#getAverageSequenceLength(TRAIN_DIR)
print(SEQUENCE_LENGTH)

if True:
    features_train,labels_train = create_dataset(TRAIN_DIR)
    features_test,labels_test = create_dataset(TEST_DIR)
    features_val,labels_val = create_dataset(VAL_DIR)

    labels_train = to_categorical(labels_train)
    labels_test = to_categorical(labels_test)
    labels_val = to_categorical(labels_val)
    np.savez('feature_train.npz', data=features_train,label=labels_train)
    np.savez('feature_test.npz', data=features_test,label=labels_test)
    np.savez('feature_val.npz', data=features_val,label=labels_val)