from torchreid.models import build_model
import torch
from PIL import Image
import torchvision.transforms as T
from glob import glob
import os
import numpy as np
import pickle
with open('errors.pkl', 'rb') as f:
    errors = pickle.load(f)
    
npy = np.load('./Car/Data/SVID/test.npy')
images = glob('./Car/Data/SVID/test/*/*.jpg')

paths2 = []

for i in images:
      ii = (i.split('\\'))[-1].split('.jpg')[0]
      label = i.split('\\')[-2]
      paths2.append((ii, label))

paths = sorted(paths2,key=lambda x:float(x[0]))

paths = [os.path.join('.\\Car\\Data\\SVID\\test\\', label , s+'.jpg') for s, label in paths]

model = build_model(
                    name='resnet50',
                    num_classes=29,
                    loss='softmax',
                    pretrained=False
                    )

trained_net = torch.load("./best_attr_net.pth")
model.load_state_dict(trained_net)

model.eval()

model.to('cuda')
os.makedirs('errors', exist_ok=True)
errors = dict()
for i, img in enumerate(paths):
    image = Image.open(img).convert('RGB')

    transforms = []
    transforms += [T.Resize((256, 256))]
    transforms += [T.ToTensor()]
    transforms += [T.Normalize(mean=[0.4611, 0.4658, 0.4728], std=[0.2552, 0.2502, 0.2520])]
    preprocess = T.Compose(transforms)

    image = preprocess(image)

    image = image.unsqueeze(0).to('cuda')

    res = model.forward_attr_eval(image)

    softmax = torch.nn.Softmax(dim=-1)
    out_data = softmax(res)
    conf, preds = torch.max(out_data, 1)
    if preds != np.argmax(npy[i]):  #### change this to match test images and npy 
        errors[img] = (preds.item(), npy[i], conf.item())   #### I'm not sure that my dataset is ok
    del image

print()