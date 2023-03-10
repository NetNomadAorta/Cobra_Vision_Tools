import glob
from PIL import Image



im = Image.open("./Images/0.jpg")

for i in range(50):
  im=im.rotate(1-(i/100), expand=False)
  im.save('./Images/{}.jpg'.format(i))