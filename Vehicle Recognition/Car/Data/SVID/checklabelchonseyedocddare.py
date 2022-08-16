import numpy as np
import glob 
import matplotlib.pyplot as plt
from PIL import Image
import cv2, os

names = ['206','207i','405','Arisun','Dena','HcCross','JackS5','KaraMazdaPickup','L90','MVM315H',
                            'MVMX22','NeissanVanet','Pars','PeykanSavari','PeykanVanet','Pride131nasimsaba',
                            'Pride132and111','Pride141','PrideVanet151','Quik','RenaultPK','RioSD','Runna','Saina',
                            'Samand','SamandSoren','Shahin','Tiba','Xantia']

paths = glob.glob('.\\train\\*\\*.jpg')
paths2 = []

for i in paths:
      ii = (i.split('\\'))[-1].split('.jpg')[0]
      label = i.split('\\')[-2]
      paths2.append((ii, label))

paths = sorted(paths2,key=lambda x:float(x[0]))

paths = [os.path.join('.\\train\\', label , s+'.jpg') for s, label in paths]

train = np.load('train.npy')
i = 29362

print(train[i])
print(names[np.argmax(train[i])])

im = cv2.imread(paths[i])

cv2.imshow('kir', im)

cv2.waitKey(0)
print()