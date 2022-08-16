import glob
import os
paths = glob.glob('./all/train/*/*.jpg')

data = {}

paths2 = []

for i in paths:
      ii = (i.split('\\'))[-1].split('.jpg')[0]
      label = i.split('\\')[-2]
      paths2.append((ii, label))

paths = sorted(paths2,key=lambda x:float(x[0]))

paths = [os.path.join('.\\train\\', label , s+'.jpg') for s, label in paths]

for i, path in enumerate(paths):
    label = path.split("\\")[2]
    filename = path.split("\\")[3]
    data.update({filename:label})

    #os.rename(path, os.path.join('./All', str(i)+'.jpg'))
    #os.remove(path)
    #im = Image.open(path).convert("RGB")
    #im.save(path.split(".webp")[0]+".jpg","jpeg"

classes = list(data.values())

categories = glob.glob('./all/train/*')
categories = [i.split('\\')[-1] for i in categories]

import numpy as np

labels = np.zeros((len(classes), len(categories)))

for i, element in enumerate(classes):
    idx = categories.index(element)
    labels[i][idx] = 1

np.save('train.npy', labels)

print()