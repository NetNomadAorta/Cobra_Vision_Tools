import argparse
import os
import sys
import torch
from torchvision import models
import math
import re
import cv2
import albumentations as A  # our data augmentation library
import time
from torchvision.utils import draw_bounding_boxes
from albumentations.pytorch import ToTensorV2

from torchvision.utils import save_image
import shutil
import json


def time_convert(sec):
    mins = sec // 60
    sec = sec % 60
    hours = mins // 60
    mins = mins % 60
    print("Time Lapsed = {0}h:{1}m:{2}s".format(int(hours), int(mins), round(sec)))


def deleteDirContents(dir):
    # Deletes photos in path "dir"
    # # Used for deleting previous cropped photos from last run
    for f in os.listdir(dir):
        full_path = os.path.join(dir, f)
        if os.path.isdir(full_path):
            shutil.rmtree(full_path)
        else:
            os.remove(full_path)


# Creates class folder
def makeDir(dir, classes_1):
    for classIndex, className in enumerate(classes_1):
        os.makedirs(dir + className, exist_ok=True)


def detect():
    source, weights, imgsz, conf_thres = opt.source, opt.weights, opt.img_size, opt.conf_thres
    print(source, "-", weights, "-", imgsz, "-", conf_thres)
    dataset_path = "./Training_Data/" + weights.split("./Models/", 1)[1].split(".model", 1)[0] + "/"

    f = open(os.path.join(dataset_path, "train", "_annotations.coco.json"))
    data = json.load(f)
    n_classes_1 = len(data['categories'])
    classes_1 = [i['name'] for i in data["categories"]]

    # lets load the faster rcnn model
    model_1 = models.detection.fasterrcnn_resnet50_fpn(pretrained=True,
                                                       min_size=imgsz,
                                                       max_size=imgsz * 3,
                                                       box_score_thresh = conf_thres
                                                       )
    in_features = model_1.roi_heads.box_predictor.cls_score.in_features  # we need to change the head
    model_1.roi_heads.box_predictor = models.detection.faster_rcnn.FastRCNNPredictor(in_features, n_classes_1)

    # Loads last saved checkpoint
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if torch.cuda.is_available():
        map_location = lambda storage, loc: storage.cuda()
    else:
        map_location = 'cpu'

    if os.path.isfile(weights):
        checkpoint = torch.load(weights, map_location=map_location)
        model_1.load_state_dict(checkpoint)

    model_1 = model_1.to(device)

    model_1.eval()
    torch.cuda.empty_cache()

    transforms_1 = A.Compose([ToTensorV2()])

    # Start FPS timer
    fps_start_time = time.time()

    color_list = ['green', 'red', 'blue', 'magenta', 'orange', 'cyan', 'lime', 'turquoise', 'yellow']
    
    
    image_path = source

    image_b4_color = cv2.imread(image_path)
    orig_image = image_b4_color
    image = cv2.cvtColor(image_b4_color, cv2.COLOR_BGR2RGB)

    transformed_image = transforms_1(image=image)
    transformed_image = transformed_image["image"]

    line_width = max(round(transformed_image.shape[1] * 0.004), 1)

    with torch.no_grad():
        prediction_1 = model_1([(transformed_image / 255).to(device)])
        pred_1 = prediction_1[0]

    coordinates = pred_1['boxes'][pred_1['scores'] > conf_thres]
    class_indexes = pred_1['labels'][pred_1['scores'] > conf_thres]
    # BELOW SHOWS SCORES - COMMENT OUT IF NEEDED
    scores = pred_1['scores'][pred_1['scores'] > conf_thres]

    print(pred_1)

    predicted_image = draw_bounding_boxes(transformed_image,
                                          boxes=coordinates,
                                          # labels = [classes_1[i] for i in class_indexes],
                                          # labels = [str(round(i,2)) for i in scores], # SHOWS SCORE IN LABEL
                                          width=line_width,
                                          colors=[color_list[i] for i in class_indexes],
                                          font="arial.ttf",
                                          font_size=20
                                          )

    # Saves full image with bounding boxes
    if len(class_indexes) != 0:
        save_image((predicted_image / 255), image_path.replace(".jpg", "") + "-Annot.jpg")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='"./Models/Construction.model"', help='model.model path(s)')
    parser.add_argument('--source', type=str, default='robotest.jpg', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--img-size', type=int, default=800, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.60, help='object confidence threshold')
    opt = parser.parse_args()
    print(opt)
    #check_requirements(exclude=('pycocotools', 'thop')).6

    # Starting stopwatch to see how long process takes
    start_time = time.time()

    detect()

    print()  # Since above print tries to write in last line used, this one clears it up
    print("Done!")

    # Stopping stopwatch to see how long process takes
    end_time = time.time()
    time_lapsed = end_time - start_time
    time_convert(time_lapsed)