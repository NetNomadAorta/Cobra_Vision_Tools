import os
import sys
import torch
import torchvision.transforms
from torchvision import models
import math
import re
import cv2
# import albumentations as A  # our data augmentation library
import time
from torchvision.utils import draw_bounding_boxes
# from albumentations.pytorch import ToTensorV2

from torchvision.utils import save_image
import shutil
import json
from datetime import datetime

# User parameters
SAVE_NAME_OD = "./Models/Construction.model"
DATASET_PATH = "./Training_Data/" + SAVE_NAME_OD.split("./Models/", 1)[1].split(".model", 1)[0] + "/"
MIN_IMAGE_SIZE = 1000  # Minimum size of image (ASPECT RATIO IS KEPT SO DONT WORRY). So for 1600x2400 -> 800x1200
TO_PREDICT_PATH = "./Images/Prediction_Images/To_Predict/"
PREDICTED_PATH = "./Images/Prediction_Images/Predicted_Images/"
SAVE_ANNOTATED_IMAGES = True
SAVE_ORIGINAL_IMAGE = False
SAVE_CROPPED_IMAGES = False
MIN_SCORE = 0.7  # Default 0.6
CAMERA_ADDRESS = 'rtsp://admin:Orbital227@192.168.24.152:3002/0/profile2/media.smp'


def time_convert(sec):
    mins = sec // 60
    sec = sec % 60
    hours = mins // 60
    mins = mins % 60
    print("Time Lapsed = {0}h:{1}m:{2}s".format(int(hours), int(mins), round(sec)))


def put_text(image, label, start_point, font, fontScale, color, thickness):
    cv2.putText(image, label, start_point, font, fontScale, (0, 0, 0), thickness + 2)
    cv2.putText(image, label, start_point, font, fontScale, color, thickness)


def deleteDirContents(dir):
    # Deletes photos in path "dir"
    # # Used for deleting previous cropped photos from last run
    for f in os.listdir(dir):
        full_path = os.path.join(dir, f)
        if os.path.isdir(full_path):
            shutil.rmtree(full_path)
        else:
            os.remove(full_path)


# Starting stopwatch to see how long process takes
start_time = time.time()

# Deletes images already in "Predicted_Images" folder
deleteDirContents(PREDICTED_PATH)

dataset_path = DATASET_PATH

f = open(os.path.join(dataset_path, "train", "_annotations.coco.json"))
data = json.load(f)
n_classes_1 = len(data['categories'])
classes_1 = [i['name'] for i in data["categories"]]

# lets load the faster rcnn model
model_1 = models.detection.fasterrcnn_resnet50_fpn(pretrained=True,
                                                   min_size=MIN_IMAGE_SIZE,
                                                   max_size=MIN_IMAGE_SIZE * 3
                                                   )
in_features = model_1.roi_heads.box_predictor.cls_score.in_features  # we need to change the head
model_1.roi_heads.box_predictor = models.detection.faster_rcnn.FastRCNNPredictor(in_features, n_classes_1)

# Loads last saved checkpoint
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if torch.cuda.is_available():
    map_location = lambda storage, loc: storage.cuda()
else:
    map_location = 'cpu'

if os.path.isfile(SAVE_NAME_OD):
    checkpoint = torch.load(SAVE_NAME_OD, map_location=map_location)
    model_1.load_state_dict(checkpoint)
else:
    print("MODEL NOT FOUND! Maybe typo?")

model_1 = model_1.to(device)

model_1.eval()
torch.cuda.empty_cache()

# transforms_1 = A.Compose([ToTensorV2()])
transforms_1 = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])

# Start FPS timer
fps_start_time = time.time()

color_list = ['green', 'red', 'blue', 'magenta', 'orange', 'cyan', 'lime', 'turquoise', 'yellow']

vc = cv2.VideoCapture(CAMERA_ADDRESS)

if vc.isOpened():  # try to get the first frame
    rval, frame = vc.read()
    print("It works!")
    if rval:
        print("Got an image!")
    else:
        print("rval bad")
else:
    rval = False
    print("Not working")

ii = 0
while True:
    if rval:
        rval, frame = vc.read()
    else:
        break

    # Getting time
    now = datetime.now()
    current_time = now.strftime("%H_%M_%S")
    image_name = current_time+".jpg"

    image_b4_color = frame
    orig_image = image_b4_color
    image = cv2.cvtColor(image_b4_color, cv2.COLOR_BGR2RGB)

    transformed_image = transforms_1(image)
    # print(transformed_image.shape)
    # transformed_image = transformed_image["image"]

    if ii == 0:
        line_width = max(round(transformed_image.shape[1] * 0.004), 1)

    with torch.no_grad():
        prediction_1 = model_1([(transformed_image).to(device)])
        pred_1 = prediction_1[0]

    coordinates = pred_1['boxes'][pred_1['scores'] > MIN_SCORE].numpy().tolist()
    class_indexes = pred_1['labels'][pred_1['scores'] > MIN_SCORE].numpy().tolist()
    # BELOW SHOWS SCORES - COMMENT OUT IF NEEDED
    scores = pred_1['scores'][pred_1['scores'] > MIN_SCORE].numpy().tolist()

    labels_found = [str(int(scores[index] * 100)) + "% - " + str(classes_1[class_index])
                    for index, class_index in enumerate(class_indexes)]

    if SAVE_ANNOTATED_IMAGES:

        predicted_image_cv2 = transformed_image.permute(1, 2, 0).contiguous().numpy()
        predicted_image_cv2 = cv2.cvtColor(predicted_image_cv2, cv2.COLOR_RGB2BGR)

        coords_person = []
        scores_person = []
        texts_person = []
        coords_helmet = []
        coords_safety_vest = []
        coords_glasses = []
        # Places text and bounding boxes around objects
        for coordinate_index, coordinate in enumerate(coordinates):
            start_point = (int(coordinate[0]), int(coordinate[1]))
            end_point = (int(coordinate[2]), int(coordinate[3]))
            color = (255, 255, 255)
            # thickness = 3
            # cv2.rectangle(predicted_image_cv2, start_point, end_point, color, thickness)

            start_point_text = (start_point[0], max(start_point[1] - 5, 0))
            font = cv2.FONT_HERSHEY_SIMPLEX
            fontScale = 0.5
            thickness = 1
            # put_text(image, label, start_point, font, fontScale, color, thickness)
            # put_text(predicted_image_cv2, labels_found[coordinate_index],
            #          start_point_text, font, fontScale, color, thickness)

            if "Person" in classes_1[class_indexes[coordinate_index]]:
                color_box = (255, 0, 255)
                coords_person.append(coordinate)
                scores_person.append(scores[coordinate_index])

            elif "Safety_Vest" in classes_1[class_indexes[coordinate_index]]:
                color_box = (255, 255, 0)
                coords_safety_vest.append([((coordinate[0] + coordinate[2]) / 2), coordinate[1]])
            elif "Helmet" in classes_1[class_indexes[coordinate_index]]:
                color_box = (0, 255, 0)
                coords_helmet.append([((coordinate[0] + coordinate[2]) / 2), coordinate[3]])
            elif "Glasses" in classes_1[class_indexes[coordinate_index]]:
                color_box = (0, 255, 255)
                coords_glasses.append([((coordinate[0] + coordinate[2]) / 2), coordinate[3]])
            else:
                color_box = (255, 0, 255)
            # cv2.rectangle(predicted_image_cv2, start_point, end_point, color_box, thickness)

        # Checks if person has proper PPE
        for index, coord_person in enumerate(coords_person):
            label = ""
            has_helmet = False
            has_highVis = False
            has_glasses = False

            # Checks if helmet present
            for coord_helmet in coords_helmet:
                if (coord_person[0] < coord_helmet[0] < coord_person[2]
                        and coord_person[1] < coord_helmet[1] < coord_person[3]):
                    has_helmet = True

            # Checks if safety_vest present
            for coord_safety_vest in coords_safety_vest:
                if (coord_person[0] < coord_safety_vest[0] < coord_person[2]
                        and coord_person[1] < coord_safety_vest[1] < coord_person[3]):
                    has_highVis = True

            # Checks if glasses present
            for coord_glasses in coords_glasses:
                if (coord_person[0] < coord_glasses[0] < coord_person[2]
                        and coord_person[1] < coord_glasses[1] < coord_person[3]):
                    has_glasses = True

            # Helmet labeler
            if has_helmet:
                label = label + "Helmet, "
                put_text(predicted_image_cv2, "Helmet",
                         (int(coord_person[0]), max(int(coord_person[1]) - 10 * 5 + 5, 0)),
                         font, fontScale, (0, 255, 0), thickness)
            else:
                label = label + "Head, "
                put_text(predicted_image_cv2, "Head",
                         (int(coord_person[0]), max(int(coord_person[1]) - 10 * 5 + 5, 0)),
                         font, fontScale, (0, 0, 255), thickness)

            # Safety_Vest labeler
            if has_highVis:
                label = label + "HighViz, "
                put_text(predicted_image_cv2, "HighViz",
                         (int(coord_person[0]), max(int(coord_person[1]) - 10 * 3 + 5, 0)),
                         font, fontScale, (0, 255, 0), thickness)
            else:
                label = label + "Clothing, "
                put_text(predicted_image_cv2, "Clothing",
                         (int(coord_person[0]), max(int(coord_person[1]) - 10 * 3 + 5, 0)),
                         font, fontScale, (0, 0, 255), thickness)

            # Glasses labeler
            if has_glasses:
                label = label + "Eyewear"
                put_text(predicted_image_cv2, "Eyewear",
                         (int(coord_person[0]), max(int(coord_person[1]) - 10 * 1 + 5, 0)),
                         font, fontScale, (0, 255, 0), thickness)
            else:
                label = label + "UncertainEyewear"
                put_text(predicted_image_cv2, "UncertainEyewear",
                         (int(coord_person[0]), max(int(coord_person[1]) - 10 * 1 + 5, 0)),
                         font, fontScale, (0, 255, 255), thickness)

            texts_person.append(label)

            if has_helmet == True and has_highVis == True and has_glasses == True:
                cv2.rectangle(predicted_image_cv2, (int(coord_person[0]), int(coord_person[1])),
                              (int(coord_person[2]), int(coord_person[3])), (0, 255, 0), thickness)
            elif has_helmet == False or has_highVis == False:
                cv2.rectangle(predicted_image_cv2, (int(coord_person[0]), int(coord_person[1])),
                              (int(coord_person[2]), int(coord_person[3])), (0, 0, 255), thickness)
            elif has_helmet == True and has_highVis == True and has_glasses == False:
                cv2.rectangle(predicted_image_cv2, (int(coord_person[0]), int(coord_person[1])),
                              (int(coord_person[2]), int(coord_person[3])), (0, 255, 255), thickness)

        # # Saves full image with bounding boxes
        # if len(class_indexes) != 0:
        #     cv2.imwrite(PREDICTED_PATH + image_name, predicted_image_cv2*255)
        #     # save_image((predicted_image/255), PREDICTED_PATH + image_name)
        cv2.imwrite(PREDICTED_PATH + image_name, predicted_image_cv2 * 255)


        # # Saves full image with bounding boxes
        # if len(class_indexes) != 0:
        #     save_image((predicted_image), PREDICTED_PATH
        #                + image_name.replace(".jpg", "") + "-Annot.jpg")

        # save_image((predicted_image/255), PREDICTED_PATH + image_name)

    if (SAVE_ORIGINAL_IMAGE and len(class_indexes) != 0):
        cv2.imwrite(PREDICTED_PATH + image_name, orig_image)

    if SAVE_CROPPED_IMAGES:

        for box_index in range(len(coordinates)):
            xmin = int(coordinates[box_index][0])
            ymin = int(coordinates[box_index][1])
            xmax = int(coordinates[box_index][2])
            ymax = int(coordinates[box_index][3])

            save_image(transformed_image[:, ymin:ymax, xmin:xmax],
                       PREDICTED_PATH + image_name.replace(".jpg", "") + "-{}-Cropped.jpg".format(box_index))

    ten_scale = 1


    ii += 1
    if ii % ten_scale == 0:
        fps_end_time = time.time()
        fps_time_lapsed = fps_end_time - fps_start_time
        fps = round(ten_scale / fps_time_lapsed, 2)
        percent_progress = round(ii / len(os.listdir(TO_PREDICT_PATH)) * 100)
        images_left = len(os.listdir(TO_PREDICT_PATH)) - ii

        time_left = images_left / (fps)  # in seconds
        mins = time_left // 60
        sec = time_left % 60

        sys.stdout.write('\033[2K\033[1G')
        print("  " + str(percent_progress) + "%",
              "-", fps, "FPS -",
              "Time Left: {0}m:{1}s".format(int(mins), round(sec)),
              end="\r"
              )
        fps_start_time = time.time()

print()  # Since above print tries to write in last line used, this one clears it up
print("Done!")

# Stopping stopwatch to see how long process takes
end_time = time.time()
time_lapsed = end_time - start_time
time_convert(time_lapsed)