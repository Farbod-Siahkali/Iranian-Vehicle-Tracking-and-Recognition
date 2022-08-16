import os
from glob import glob
import numpy as np
from random import shuffle



directory = r'carScrape\SVID'

folders = glob(directory+ r'\*')

images = glob(directory + r'\*' + r'\*.jpg')


for i,image in enumerate(images):
    os.rename(image, os.path.join(os.path.join(image.split('\\')[0], image.split('\\')[1], image.split('\\')[2]), f"{i}.jpg"))



def train_test_split(directory):
    folders = glob(directory + r'\*')
    for folder in folders:
        images = glob(os.path.join(folder, '*.jpg'))
        





from PIL import Image
import glob, os
paths = glob.glob('./*/*.jpg')

data = {}

for i, path in enumerate(paths):
    label = path.split("\\")[1]
    filename = path.split("\\")[1] + '-' + path.split("\\")[2]
    data.update({filename:label})

    #os.rename(path, os.path.join('./All', str(i)+'.jpg'))
    #os.remove(path)
    #im = Image.open(path).convert("RGB")
    #im.save(path.split(".webp")[0]+".jpg","jpeg")

classes = list(data.values())

categories = list(dict.fromkeys(classes))

import numpy as np

labels = np.zeros((len(classes), len(categories)))

for i, element in enumerate(classes):
    idx = categories.index(element)
    labels[i][idx] = 1

print()


