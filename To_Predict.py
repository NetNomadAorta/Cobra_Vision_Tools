
import torch
from torchvision import models
import cv2
import albumentations as A  # our data augmentation library
import time
from torchvision.utils import draw_bounding_boxes
# Now, we will define our transforms
from albumentations.pytorch import ToTensorV2

from torchvision.utils import save_image


# User parameters
IMAGE_SIZE              = 800 # Minimum size of image (ASPECT RATIO IS KEPT SO DONT WORRY). So for 1600x2400 -> 800x1200
IMAGE_PATH              = "robotest.jpg"
MIN_SCORE               = 0.6 # Default 0.5


def time_convert(sec):
    mins = sec // 60
    sec = sec % 60
    hours = mins // 60
    mins = mins % 60
    print("Time Lapsed = {0}h:{1}m:{2}s".format(int(hours), int(mins), round(sec) ) )



# Starting stopwatch to see how long process takes
start_time = time.time()

# lets load the faster rcnn model
model_1 = models.detection.fasterrcnn_resnet50_fpn(pretrained=True)

# Loads last saved checkpoint
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if torch.cuda.is_available():
    map_location=lambda storage, loc: storage.cuda()
else:
    map_location='cpu'

model_1 = model_1.to(device)

model_1.eval()
torch.cuda.empty_cache()

transforms_1 = A.Compose([
    ToTensorV2()
])

# Start FPS timer
fps_start_time = time.time()

image_path = IMAGE_PATH

image_b4_color = cv2.imread(image_path)
orig_image = image_b4_color
image = cv2.cvtColor(image_b4_color, cv2.COLOR_BGR2RGB)

transformed_image = transforms_1(image=image)
transformed_image = transformed_image["image"]

with torch.no_grad():
    prediction_1 = model_1([(transformed_image/255).to(device)])
    pred_1 = prediction_1[0]

coordinates = pred_1['boxes'][pred_1['scores'] > MIN_SCORE]
class_indexes = pred_1['labels'][pred_1['scores'] > MIN_SCORE]
# BELOW SHOWS SCORES - COMMENT OUT IF NEEDED
scores = pred_1['scores'][pred_1['scores'] > MIN_SCORE]

predicted_image = draw_bounding_boxes(transformed_image,
    boxes = coordinates,
    width = 2,
    font = "arial.ttf",
    font_size = 20
    )

# Saves full image with bounding boxes
if len(class_indexes) != 0:
    save_image((predicted_image/255), "robotest-annotated.jpg")


print() # Since above print tries to write in last line used, this one clears it up
print("Done!")

# Stopping stopwatch to see how long process takes
end_time = time.time()
time_lapsed = end_time - start_time
time_convert(time_lapsed)