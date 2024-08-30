import threading 
from threading import Lock
import cv2
import time
from ultralytics import YOLO
from itertools import count
import torch


rtsp_link = "PUT YOUR LINK HERE"
vcap = cv2.VideoCapture(rtsp_link)
model = YOLO('yolov8m.pt')


conf_thresh = 0.6

i=0 # index for filenames

SaveTheImage = 0 #turns on capture of all screenshots to .png files
ShowTheImage = 0 # NEVER USE THIS, it's more efficient to just set show=True in "model.precict"


latest_frame = None
last_ret = None
lo = Lock()

def rtsp_cam_buffer(vcap):
    global latest_frame, lo, last_ret
    while True:
        with lo:
            last_ret, latest_frame = vcap.read()


t1 = threading.Thread(target=rtsp_cam_buffer,args=(vcap,),name="rtsp_read_thread")
t1.daemon=True
t1.start()

while True :
    if (last_ret is not None) and (latest_frame is not None):
        img = latest_frame.copy()
        

        #cv2.imshow('frame', img)
        #cv2.imwrite("Howdy.png", img)
        #results = model.predict(img, device='mps', show=True, save=False, save_txt=False, conf=conf_thresh, save_conf=False, stream=True) #stream t/f seems to make no difference..  This one uses MPS not CPU, see Powerpoint
        results = model.predict(img, show=True, save=False, save_txt=False, conf=conf_thresh, save_conf=False, stream=True) #stream t/f seems to make no difference..  

        # Process results list
        for result in results:
            boxes = result.boxes  # Boxes object for bounding box outputs
            masks = result.masks  # Masks object for segmentation masks outputs
            keypoints = result.keypoints  # Keypoints object for pose outputs
            probs = result.probs  # Probs object for classification outputs
            obb = result.obb  # Oriented boxes object for OBB outputs
            if ShowTheImage == 1:
                result.show()  # display to screen
            if SaveTheImage == 1:
                filename33 = f"{"Joe"}_{i+1}.png"  # Incrementing filename
                i=i+1 #increment index
                result.save(filename33)

        time.sleep(.2)
        if cv2.waitKey(20) & 0xFF == ord('q'):
            break

    else:
        print("unable to read the frame")
        time.sleep(0.1)
        continue
