import time

import cv2

vc = cv2.VideoCapture('rtsp://admin:Orbital227@192.168.24.152:3002/0/profile2/media.smp')

if vc.isOpened():  # try to get the first frame
    rval, frame = vc.read()
    print("It works!")
else:
    rval = False
    print("Not working")

ii = 0
while True:
    if rval:
        rval, frame = vc.read()

        print("Saving image!")
        cv2.imwrite("{}.jpg".format(ii), frame)
        time.sleep(1)
    else:
        print("rval bad")

    ii += 1

    if ii == 10:
        break