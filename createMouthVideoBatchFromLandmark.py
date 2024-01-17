import sys
import cv2 
import os
from pathlib import Path
import glob
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from moviepy.editor import *
import mediapipe
from mediapipe.python.solutions.drawing_utils import _normalized_to_pixel_coordinates
seed_constant = 27
np.random.seed(seed_constant)

def image_resize(image, width = None, height = None, inter = cv2.INTER_AREA):
    # initialize the dimensions of the image to be resized and
    # grab the image size
    dim = None
    (h, w) = image.shape[:2]

    # if both the width and height are None, then return the
    # original image
    if width is None and height is None:
        return image

    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the
        # dimensions
        r = height / float(h)
        dim = (int(w * r), height)

    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the
        # dimensions
        r = width / float(w)
        dim = (width, int(h * r))

    # resize the image
    resized = cv2.resize(image, dim, interpolation = inter)

    # return the resized image
    return resized

def image_paddingresize(image,new_image_width,new_image_height):
    old_image_height, old_image_width, channels = image.shape
    # create new image of desired size and color (blue) for padding
    color = (0,0,0)
    result = np.full((new_image_height,new_image_width, channels), color, dtype=np.uint8)
    # compute center offset
    x_center = (new_image_width - old_image_width) // 2
    y_center = (new_image_height - old_image_height) // 2

    # copy img image into center of result image
    result[y_center:y_center+old_image_height, x_center:x_center+old_image_width] = image
    return result


def getLandMarkpoints(face_lips):
    df = pd.DataFrame(list(face_lips), columns = ["p1", "p2"])
    p1 = df.iloc[1]["p1"]
    p2 = df.iloc[1]["p2"]

    routes_idx = []
    p1 = df.iloc[0]["p1"]
    p2 = df.iloc[0]["p2"]

    for i in range(0, df.shape[0]):
        p1 = df.iloc[i]["p1"]
        p2 = df.iloc[i]["p2"] 
        #print(p1, p2)
        #print(p2)
        #print(df["p1"])
        if df[df["p1"] == p2].size == 0:
            continue
        else:
            obj = df[df["p1"] == p2]
            p1 = obj["p1"].values[0]
            p2 = obj["p2"].values[0]
             
            route_idx = []
            route_idx.append(p1)
            route_idx.append(p2)
            routes_idx.append(route_idx)
    return routes_idx
 
# -------------------------------

def getBoundary(landmarks,img,routes_idx): 
    #for route_idx in routes_idx:
    #    print(f"Draw a line between {route_idx[0]}th landmark point to {route_idx[1]}th landmark point")
        
    routes = []
     
    for source_idx, target_idx in routes_idx:
         
        source = landmarks.landmark[source_idx]
        target = landmarks.landmark[target_idx]
             
        relative_source = (int(img.shape[1] * source.x), int(img.shape[0] * source.y))
        relative_target = (int(img.shape[1] * target.x), int(img.shape[0] * target.y))
     
        #cv2.line(img, relative_source, relative_target, (255, 255, 255), thickness = 1)
         
        routes.append(relative_source)
        routes.append(relative_target)
    #print(routes)
    resMax = [max(idx) for idx in zip(*routes)]
    resMin = [min(idx) for idx in zip(*routes)]
    #print(resMax[1])
    #print(routes[0][0])
    return routes,resMax,resMin

def PlayRecord(video_file_path,output_file_path):

    # Initialize the VideoCapture object to read from the video file.
    video_reader = cv2.VideoCapture(video_file_path)

    # Get the width and height of the video.
    original_video_width = int(video_reader.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_video_height = int(video_reader.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
        # Initialize the VideoWriter Object to store the output video in the disk.
    video_writer = cv2.VideoWriter(output_file_path, cv2.VideoWriter_fourcc('m', 'p', '4', 'v'), 
                                   video_reader.get(cv2.CAP_PROP_FPS), (224, 224)) #add ,0 for grayscaled video



    # load face detection model
    #mp_face = mp.solutions.face_detection.FaceDetection(
    #    model_selection=1, # model selection
    #    min_detection_confidence=0.5 # confidence threshold
    #)
    
    mp_face_mesh = mediapipe.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True)
    
    ds_factor = 1

    
    trigonce = False
    # Iterate until the video is accessed successfully.
    while video_reader.isOpened():
        # Read the frame.
        ok, frame = video_reader.read() 
        
        # Check if frame is not read properly then break the loop.
        if not ok:
            break

        #frame = cv2.resize(frame, None, fx=ds_factor, fy=ds_factor, interpolation=cv2.INTER_AREA)
        image_input = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        #print(image_input)
        results = face_mesh.process(image_input)
        #print("aaaaaaaaaaaaaaaa",results)
         # Print and draw face mesh landmarks on the image.
        if not results.multi_face_landmarks:
            print(video_file_path+"frame no detection ")
        landmarks = results.multi_face_landmarks[0]
        #print("sampai sini?")
        face_lips = mp_face_mesh.FACEMESH_LIPS
        #results = mp_face.process(image_input)
        #detection=results.detections[0]
        kui = cv2.waitKey(1)
        #frame_height, frame_width,c = frame.shape
        #trigonce = False
        routes_idx = getLandMarkpoints(face_lips)
        routes,resMax,resMin = getBoundary(landmarks,image_input,routes_idx)
        crop_img = image_input[resMin[1]:resMax[1], resMin[0]:resMax[0]] #untuk gambar
        crop_img = image_resize(crop_img,width=224,height=224)
        crop_img = image_paddingresize(crop_img,224,224)
        crop_img = cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB)
        #cv2.imwrite('result.jpg', cv2.cvtColor(crop_img, cv2.COLOR_RGB2BGR))
            # Write The frame into the disk using the VideoWriter Object.
        video_writer.write(crop_img)
        
        if kui==27:
            break
        
     # Release the VideoCapture and VideoWriter objects.
    video_reader.release()
    video_writer.release()
    cv2.destroyAllWindows()
    
def main():
    label = "outfinal"
    for i in range(10,11):
        videopath = str(i)
        N = len([entry for entry in os.listdir(videopath) if os.path.isfile(os.path.join(videopath, entry))])
        Path(videopath+label).mkdir(parents=True, exist_ok=True)
        for i in range(1,N+1):
            input_video_file_path = videopath+'/data ('+str(i)+').mp4'
            # Construct the output video path.
            output_video_file_path = videopath+label+'/data ('+str(i)+').mp4'
            # Perform Action Recognition on the Test Video.
            PlayRecord(input_video_file_path, output_video_file_path)

            # Display the output video.
            # VideoFileClip(output_video_file_path, audio=False, target_resolution=(300,None)).ipython_display()

if __name__ == "__main__":
   main()

