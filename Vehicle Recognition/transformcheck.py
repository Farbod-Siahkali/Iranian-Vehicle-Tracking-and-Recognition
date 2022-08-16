from torchvision import transforms as T
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np


transforms = []
transforms += [T.ToTensor()]
transforms += [T.RandomPosterize(bits=2)]
preprocess = T.Compose(transforms)

image = Image.open('./aaa.jpg').convert('RGB')

image = preprocess(image).permute((1,2,0))

image = image.cpu().detach().numpy()


plt.imshow(image)
plt.show()

print()