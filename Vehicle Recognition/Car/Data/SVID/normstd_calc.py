import glob
import torchvision.transforms as T
import numpy as np
from PIL import Image

'''m1 = [0.47365347, 0.4684487, 0.46542493]
std1 = [0.2530077, 0.24997568, 0.25495428] 
m2 = [0.47200933, 0.46324202, 0.45690387]
std2 = [0.25115883, 0.25053665, 0.25555712]'''

transforms = []
transforms += [T.Resize((256, 256))]
transforms += [T.ToTensor()]
preprocess = T.Compose(transforms)

path = glob.glob('.\\train\\*.jpg')

p1 = path[:len(path)//2]
p2 = path[len(path)//2:]

images = []
images = [preprocess(Image.open(i)) for i in p1] # generator comprehension
images = np.stack(images)  # this takes time 
mean1 = [np.mean(images[:,0,:,:]),np.mean(images[:,1,:,:]),np.mean(images[:,2,:,:])]
std1 = [np.std(images[:,0,:,:]),np.std(images[:,1,:,:]),np.std(images[:,2,:,:])]

images = []
images = [preprocess(Image.open(i)) for i in p2] # generator comprehension
images = np.stack(images)  # this takes time 
mean2 = [np.mean(images[:,0,:,:]),np.mean(images[:,1,:,:]),np.mean(images[:,2,:,:])]
std2 = [np.std(images[:,0,:,:]),np.std(images[:,1,:,:]),np.std(images[:,2,:,:])]

print(mean1)
print(std1)
print(mean2)
print(std2)
