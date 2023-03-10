import os
import sys
import torch
import torchvision.transforms
from torchvision import models
import cv2
import json
import argparse

# User parameters
SAVE_NAME_OD = "./Models/Construction.model"
DATASET_PATH = "./Training_Data/" + SAVE_NAME_OD.split("./Models/", 1)[1].split(".model", 1)[0] + "/"
MIN_IMAGE_SIZE = 1000  # Minimum size of image (ASPECT RATIO IS KEPT SO DONT WORRY). So for 1600x2400 -> 800x1200
TO_PREDICT_PATH = "./Images/Prediction_Images/To_Predict/"
PREDICTED_PATH = "./Images/Prediction_Images/Predicted_Images/"
SAVE_ANNOTATED_IMAGES = True
SAVE_ORIGINAL_IMAGE = False
SAVE_CROPPED_IMAGES = False
MIN_SCORE = 0.6  # Default 0.6


def put_text(image, label, start_point, font, fontScale, color, thickness):
    cv2.putText(image, label, start_point, font, fontScale, (0, 0, 0), thickness + 2)
    cv2.putText(image, label, start_point, font, fontScale, color, thickness)

dataset_path = DATASET_PATH

f = open(os.path.join(dataset_path, "train", "_annotations.coco.json"))
#f = open(dataset_path + "train/_annotations.coco.json")
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
color_list = ['green', 'red', 'blue', 'magenta', 'orange', 'cyan', 'lime', 'turquoise', 'yellow']

def detect(image):
    image_b4_color = image
    orig_image = image_b4_color
    image = cv2.cvtColor(image_b4_color, cv2.COLOR_BGR2RGB)

    transformed_image = transforms_1(image)

    line_width = 2

    with torch.no_grad():
        prediction_1 = model_1([(transformed_image).to(device)])
        pred_1 = prediction_1[0]

    coordinates = pred_1['boxes'][pred_1['scores'] > MIN_SCORE].numpy().tolist()
    class_indexes = pred_1['labels'][pred_1['scores'] > MIN_SCORE].numpy().tolist()
    # BELOW SHOWS SCORES - COMMENT OUT IF NEEDED
    scores = pred_1['scores'][pred_1['scores'] > MIN_SCORE].numpy().tolist()

    labels_found = [str(int(scores[index] * 100)) + "% - " + str(classes_1[class_index])
                    for index, class_index in enumerate(class_indexes)]

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
        end_point = ( int(coordinate[2]), int(coordinate[3]) )
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
            color_box = (255,0,255)
            coords_person.append(coordinate)
            scores_person.append(scores[coordinate_index])

        elif "Safety_Vest" in classes_1[class_indexes[coordinate_index]]:
            color_box = (255, 255, 0)
            coords_safety_vest.append( [((coordinate[0]+coordinate[2])/2), coordinate[1] ] )
        elif "Helmet" in classes_1[class_indexes[coordinate_index]]:
            color_box = (0, 255, 0)
            coords_helmet.append( [((coordinate[0]+coordinate[2])/2), coordinate[3] ] )
        elif "Glasses" in classes_1[class_indexes[coordinate_index]]:
            color_box = (0, 255, 255)
            coords_glasses.append( [((coordinate[0]+coordinate[2])/2), coordinate[3] ] )
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
                     (int(coord_person[0]), max(int(coord_person[1]) - 10*5+5, 0)),
                     font, fontScale, (0,255,0), thickness)
        else:
            label = label + "Head, "
            put_text(predicted_image_cv2, "Head",
                     (int(coord_person[0]), max(int(coord_person[1]) - 10*5+5, 0)),
                     font, fontScale, (0,0,255), thickness)

        # Safety_Vest labeler
        if has_highVis:
            label = label + "HighViz, "
            put_text(predicted_image_cv2, "HighViz",
                     (int(coord_person[0]), max(int(coord_person[1]) - 10*3+5, 0)),
                     font, fontScale, (0,255,0), thickness)
        else:
            label = label + "Clothing, "
            put_text(predicted_image_cv2, "Clothing",
                     (int(coord_person[0]), max(int(coord_person[1]) - 10*3+5, 0)),
                     font, fontScale, (0,0,255), thickness)

        # Glasses labeler
        if has_glasses:
            label = label + "Eyewear"
            put_text(predicted_image_cv2, "Eyewear",
                     (int(coord_person[0]), max(int(coord_person[1]) - 10*1+5, 0)),
                     font, fontScale, (0,255,0), thickness)
        else:
            label = label + "UncertainEyewear"
            put_text(predicted_image_cv2, "UncertainEyewear",
                     (int(coord_person[0]), max(int(coord_person[1]) - 10*1+5, 0)),
                     font, fontScale, (0,255,255), thickness)

        texts_person.append(label)

        if has_helmet == True and has_highVis == True and has_glasses == True:
            cv2.rectangle(predicted_image_cv2, (int(coord_person[0]), int(coord_person[1])),
                          (int(coord_person[2]), int(coord_person[3])), (0,255,0), thickness)
        elif has_helmet == False or has_highVis == False:
            cv2.rectangle(predicted_image_cv2, (int(coord_person[0]), int(coord_person[1])),
                          (int(coord_person[2]), int(coord_person[3])), (0,0,255), thickness)
        elif has_helmet == True and has_highVis == True and has_glasses == False:
            cv2.rectangle(predicted_image_cv2, (int(coord_person[0]), int(coord_person[1])),
                          (int(coord_person[2]), int(coord_person[3])), (0,255,255), thickness)

    output_image = predicted_image_cv2*255

    # if len(coords_person) ==0:



    # Creating JSON section
    # ==================================================================================
    data = []

    for index, text_person in enumerate(texts_person):
        data.append({
            "coordinate": coords_person[index],
            "score": scores_person[index],
            "label": text_person
        })

    return data, output_image

#arguments
parser = argparse.ArgumentParser()
parser.add_argument('--image', type=str, required=True)

args = parser.parse_args()

#process single image passed as argument
if __name__ == "__main__":
    image = cv2.imread(args.image)
    
    data, image_out = detect(image)

    jsonpath = args.image.replace(".jpg", "_result.json")
    imgpath = args.image.replace(".jpg", "_result.jpg")

    with open(jsonpath, 'w') as f:
        json.dump(data, f)

    print(jsonpath)

    cv2.imwrite(imgpath, image_out)

    print(imgpath)