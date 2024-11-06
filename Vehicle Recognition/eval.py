from torchreid.models import build_model
from torchreid import utils
import torch, os
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data import Dataset 
from torchvision import transforms
from PIL import Image
import numpy as np
import glob

softmax = torch.nn.Softmax(dim=-1)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def f1_loss(y_true, y_pred):        #should change
    
    tp = (y_true * y_pred).sum().to(torch.float32)
    tn = ((1 - y_true) * (1 - y_pred)).sum().to(torch.float32)
    fp = ((1 - y_true) * y_pred).sum().to(torch.float32)
    fn = (y_true * (1 - y_pred)).sum().to(torch.float32)
    
    epsilon = 1e-7
    
    precision = tp / (tp + fp + epsilon)
    recall = tp / (tp + fn + epsilon)
    
    f1 = 2* (precision*recall) / (precision + recall + epsilon)
    return f1

class CA_Loader(Dataset):
    def __init__(self,img_path,
                 attr,
                 resolution,
                 transform=None):
        
        # images variables:
        self.img_path = img_path
        self.img_names = attr['img_names']
        self.resolution = resolution
      
        self.attr = attr['attributes']         
        
        if transform:
            self.transform = transform
        else:
            self.transform = None
        
        transform_list = []
        transform_list += [transforms.ToTensor()]
        transform_list += [transforms.Resize(resolution)] #should change
        transform_list += [transforms.Normalize(mean=[0.4611, 0.4658, 0.4728], std=[0.2552, 0.2502, 0.2520])]
        preprocess = transforms.Compose(transform_list)
        self.preprocess = preprocess
            
    def __len__(self):
        return len(self.img_names)
    
    def __getitem__(self,idx):
        
        img = Image.open(self.img_names[idx]).convert('RGB')
        if self.transform:
            img = self.transform(img) 
        img = self.preprocess(img)
        out = {'img' : img}
        out.update({'attributes':self.attr[idx]})
         
        return out

def load_attributes(path_attr):
    attr_vec_np = np.load(path_attr)# loading attributes
        # attributes
    attr_vec_np = attr_vec_np.astype(np.int32)
    return torch.from_numpy(attr_vec_np)

def load_image_names(main_path):
    paths = glob.glob(main_path+'/*/*.jpg')
    paths2 = []
    for i in paths:
        ii = (i.split('\\'))[-1].split('.jpg')[0]
        label = i.split('\\')[-2]
        paths2.append((ii, label))
    paths = sorted(paths2,key=lambda x:float(x[0]))
    paths = [os.path.join(main_path, label , s+'.jpg') for s, label in paths]
    return np.array(paths)

def data_delivery(main_path, path_attr=None):
    output = {}
    attr_vec = load_attributes(path_attr) # numpy array
    output.update({'attributes':attr_vec})
    img_names = load_image_names(main_path)
    output.update({'img_names':img_names})   

    output.update({'names' : ['206','207i','405','Arisun','Dena','HcCross','JackS5','KaraMazdaPickup','L90','MVM315H',
                            'MVMX22','NeissanVanet','Pars','PeykanSavari','PeykanVanet','Pride131nasimsaba',
                            'Pride132and111','Pride141','PrideVanet151','Quik','RenaultPK','RioSD','Runna','Saina',
                            'Samand','SamandSoren','Shahin','Tiba','Xantia']}) #should change

    return output

test_img_path = './Car/Data/SIVD/test'
path_attr_test = './Car/Data/SIVD/test.npy'

attr_test = data_delivery(test_img_path, path_attr_test)                

test_data = CA_Loader(img_path=test_img_path,
                            attr=attr_test,
                            resolution=(256,256),
                            transform=None) 

loss = nn.CrossEntropyLoss().to(device)

test_loader = DataLoader(test_data,batch_size=150,shuffle=False)

model = build_model(name='resnet34',
                    num_classes = 29, #check this for new dataset
                    loss='softmax', 
                    pretrained=False
                    )

trained = './29classes/RESNET34/best_attr_net.pth' #set the best pretrained
utils.load_pretrained_weights(model, trained)

attr_net = model.to(device)

test_loss = []
F1_test = []
Acc_test = []
recall_test = []
avg_precision_test = []
times = []
loss_test = []

from sklearn.metrics import precision_score, recall_score, f1_score, precision_recall_curve, accuracy_score
from sklearn.metrics import average_precision_score

attr_net.eval()

test_array = np.load(r"Car\Data\SIVD\test.npy")
pred_vec = np.array([])
test_vec = np.array([])


import time

for i in test_array:
    index = np.where(i==1.)
    test_vec = np.concatenate((test_vec, index[0]))

with torch.no_grad():
    for idx, data in enumerate(test_loader):
        for key, _ in data.items():
            data[key] = data[key].to(device)
        
        start_time = time.time()
        out_data = attr_net.forward_attr_eval(data['img'])
        tt = time.time() - start_time
        loss_part = loss(out_data, data['attributes'].float())
        out_data = softmax(out_data)
        _, preds = torch.max(out_data, 1)
        pred_vec = np.concatenate((pred_vec, preds.cpu()))
        out_data = torch.zeros(out_data.shape[0], out_data.shape[1]).to(device)
        for i, pred in enumerate(preds):
            out_data[i][pred] = 1

        #precision = precision_score(data['attributes'].cpu(), out_data.cpu(), pos_label='positive', average='samples')
        avg_precision = average_precision_score(data['attributes'].cpu().reshape((-1)), out_data.cpu().reshape((-1)))
        #logloss = log_loss(data['attributes'].cpu(), out_data.cpu())
        #recall = recall_score(data['attributes'].cpu(), out_data.cpu(), pos_label='positive', average='samples')
        #f1 = f1_score(data['attributes'].cpu(), out_data.cpu(), pos_label='positive', average='samples')
        acc = accuracy_score(data['attributes'].cpu(), out_data.cpu())

        print(idx*150)
        

        #recall_test.append(recall)
        #precision_test.append(precision)
        avg_precision_test.append(avg_precision)
        if idx > 1:
            times.append(tt) 
        Acc_test.append(acc) 
        loss_test.append(loss_part.item())
    
#F1 = sum(F1_test) / len(F1_test)
#Recall = sum(recall_test) / len(recall_test)
#Presicion = sum(precision_test) / len(precision_test)
mAP = sum(avg_precision_test) / len(avg_precision_test)
Accuary = sum(Acc_test) / len(Acc_test)
Lss = sum(loss_test) / len(loss_test)
avgtime = sum(times) / len(times)

print()