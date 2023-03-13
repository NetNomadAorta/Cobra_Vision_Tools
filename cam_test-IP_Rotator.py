import time
import cv2
from datetime import datetime
import os

TO_PREDICT_PATH = "./Images/Prediction_Images/To_Predict/"
WAIT_TIME = 30


def deleteDirContents(dir):
    # Deletes photos in path "dir"
    # # Used for deleting previous cropped photos from last run
    for f in os.listdir(dir):
        full_path = os.path.join(dir, f)
        if os.path.isdir(full_path):
            shutil.rmtree(full_path)
        else:
            os.remove(full_path)

deleteDirContents(TO_PREDICT_PATH)


vc0 = cv2.VideoCapture('rtsp://admin:Orbital227@192.168.24.152:3002/0/profile2/media.smp')
vc1 = cv2.VideoCapture('rtsp://admin:Orbital227@192.168.24.152:3002/1/profile2/media.smp')
vc2 = cv2.VideoCapture('rtsp://admin:Orbital227@192.168.24.152:3002/2/profile2/media.smp')
vc3 = cv2.VideoCapture('rtsp://admin:Orbital227@192.168.24.152:3002/3/profile2/media.smp')

if vc0.isOpened():  # try to get the first frame
    rval0, frame0 = vc0.read()
else:
    rval0 = False

if vc1.isOpened():  # try to get the first frame
    rval1, frame1 = vc1.read()
else:
    rval1 = False

if vc2.isOpened():  # try to get the first frame
    rval2, frame1 = vc2.read()
else:
    rval2 = False

if vc3.isOpened():  # try to get the first frame
    rval3, frame1 = vc3.read()
else:
    rval3 = False

ii = 0
while True:
    if rval0:
        rval0, frame0 = vc0.read()

        now = datetime.now()
        now = now.strftime("%Y_%m_%d-%H_%M_%S")

        print("Saving image!")
        cv2.imwrite("./Images/Prediction_Images/To_Predict/{}-cam_0.jpg".format(now), frame0)
        time.sleep(WAIT_TIME)
    else:
        print("rval bad")

    if rval1:
        rval1, frame1 = vc1.read()

        now = datetime.now()
        now = now.strftime("%Y_%m_%d-%H_%M_%S")

        print("Saving image!")
        cv2.imwrite("./Images/Prediction_Images/To_Predict/{}-cam_1.jpg".format(now), frame1)
        time.sleep(WAIT_TIME)
    else:
        print("rval bad")

    if rval2:
        rval2, frame2 = vc2.read()

        now = datetime.now()
        now = now.strftime("%Y_%m_%d-%H_%M_%S")

        print("Saving image!")
        cv2.imwrite("./Images/Prediction_Images/To_Predict/{}-cam_2.jpg".format(now), frame2)
        time.sleep(WAIT_TIME)
    else:
        print("rval bad")

    if rval3:
        rval3, frame3 = vc3.read()

        now = datetime.now()
        now = now.strftime("%Y_%m_%d-%H_%M_%S")

        print("Saving image!")
        cv2.imwrite("./Images/Prediction_Images/To_Predict/{}-cam_3.jpg".format(now), frame3)
        time.sleep(WAIT_TIME)
    else:
        print("rval bad")



    ii += 1

    if ii == 50:
        break