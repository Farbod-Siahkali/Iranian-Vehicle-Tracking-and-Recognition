from torchreid.models import build_model
import torch
from PIL import Image
import torchvision.transforms as T

names = ['206','207i','405','Arisun','Dena','HcCross','JackS5','KaraMazdaPickup','L90','MVM315H',
                            'MVMX22','NeissanVanet','Pars','PeykanSavari','PeykanVanet','Pride131nasimsaba',
                            'Pride132and111','Pride141','PrideVanet151','Quik','RenaultPK','RioSD','Runna','Saina',
                            'Samand','SamandSoren','Shahin','Tiba','Xantia']

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

image = Image.open("./mm.jpg").convert('RGB')

transforms = []
transforms += [T.Resize((256, 256))]
transforms += [T.ToTensor()]
transforms += [T.Normalize(mean=[0.4611, 0.4658, 0.4728], std=[0.2552, 0.2502, 0.2520])]
preprocess = T.Compose(transforms)

image = preprocess(image)

images = image.unsqueeze(0).to('cuda')

res = model.forward_attr_eval(images)

softmax = torch.nn.Softmax(dim=-1)
out_data = softmax(res)
_, preds = torch.max(out_data, 1)

print(names[preds.item()])

print()