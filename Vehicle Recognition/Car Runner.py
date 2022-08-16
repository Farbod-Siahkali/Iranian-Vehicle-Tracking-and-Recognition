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

#IranianCarsDataset bama
#CarDataset camera
train_img_path = './Car/Data/SVID/train'
test_img_path = './Car/Data/SVID/test'
path_attr_train = './Car/Data/SVID/train.npy'
path_attr_test = './Car/Data/SVID/test.npy'
saving_path = './Car/Results/'

attr_train = data_delivery(train_img_path, path_attr_train) #should check for new dataset
attr_test = data_delivery(test_img_path, path_attr_test)

train_transform =  transforms.Compose([transforms.RandomRotation(degrees=(0,45)),
                        transforms.RandomHorizontalFlip(),
                        transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.2, 2)),
                        transforms.ColorJitter(saturation=[0.7,1.4], brightness = (0.8, 1.2), contrast = (0.8, 1.2)),
                        transforms.RandomPerspective(distortion_scale=0.3, p=0.4),
                        transforms.RandomAffine(degrees=(1, 15), scale=(0.92, 0.99)),
                        #transforms.RandomInvert(),
                        #transforms.RandomPosterize(bits=2),
                        #transforms.RandomSolarize(threshold=192.0)
                        ])                    

train_data = CA_Loader(img_path=train_img_path, #check for new dataset
                            attr=attr_train,
                            resolution=(256,256),
                            transform=train_transform) 

test_data = CA_Loader(img_path=test_img_path,
                            attr=attr_test,
                            resolution=(256,256),
                            transform=None) 

loss = nn.CrossEntropyLoss().to(device)

batch_size = 32 #can increase
train_loader = DataLoader(train_data,batch_size=batch_size,shuffle=True)
test_loader = DataLoader(test_data,batch_size=100,shuffle=False)

model = build_model(name='resnet50',
                    num_classes = attr_train['attributes'].shape[1], #check this for new dataset
                    loss='softmax', 
                    pretrained=True
                    )

'''def get_n_params(model):
    pp=0
    for p in list(model.parameters()):
        nn=1
        for s in list(p.size()):
            nn = nn*s
        pp += nn
    return pp
print(get_n_params(model))'''

#pretrained = './checkpoints/osnet_x1_0_msmt17.pth'
trained = './best_attr_net.pth' #set the best pretrained
utils.load_pretrained_weights(model, trained)

'''params = model.parameters()
for idx, param in enumerate(params):
    param.requires_grad = False'''

#attr_net = attributes_model(model, feature_dim = 512, attr_dim = attr_train['attributes'].shape[1])
attr_net = model.to(device)

params = attr_net.parameters()

optimizer = torch.optim.AdamW(params, lr=3e-5, betas=(0.9, 0.99), eps=1e-08) #check the lr and milestones
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[20, 60, 100, 140], gamma=0.7)

train_loss = []
test_loss = []
F1_train = []
F1_test = []
Acc_train = []
Acc_test = []
recall_test = []
precision_test = []

num_epoch = 200 #check this

attr_loss_train = torch.zeros((num_epoch))
attr_loss_test = torch.zeros((num_epoch))

loss_min = 10000
f1_best = 0

from sklearn.metrics import f1_score, precision_recall_curve   

for epoch in range(1, num_epoch + 1):
    attr_net = attr_net.to(device)
    attr_net.train()

    loss_e = []
    loss_t = []
    ft_train = []
    ft_test = []
    acc_train = []
    acc_test = []
    ret_test = []
    prt_test = []
    
    for idx, data in enumerate(train_loader):
        for key, _ in data.items():
            data[key] = data[key].to(device)
        # forward step
        optimizer.zero_grad()
        out_data = attr_net.forward(data['img'])

        loss_part = loss(out_data, data['attributes'].float())

        loss_e.append(loss_part.item())
        print(loss_part.item())
        loss_part.backward()

        optimizer.step()

    train_loss.append(np.mean(loss_e))
    attr_net.eval()

    with torch.no_grad():
        for idx, data in enumerate(test_loader):
            for key, _ in data.items():
                data[key] = data[key].to(device)
            
            out_data = attr_net.forward_attr_eval(data['img'])

            loss_part = loss(out_data, data['attributes'].float())

            out_data = softmax(out_data)
            _, preds = torch.max(out_data, 1)
            out_data = torch.zeros(out_data.shape[0], out_data.shape[1]).to(device)

            for i, pred in enumerate(preds):
                out_data[i][pred] = 1
            
            f1_sc = f1_loss(data['attributes'], out_data)
            #for i in range(data['attributes'].shape[1]): lst.append(f1_score(data['attributes'][i].cpu().data, out_data[i].cpu()))
            ft_test.append(f1_sc.item())
            #ret_test.append(recall_sc.item())
            #prt_test.append(precision_sc.item())
            #acc_test.append(test_attr_metrics[-3]) 
            loss_t.append(loss_part.item())
        
    test_loss.append(np.mean(loss_t))
    F1_test.append(np.mean(ft_test))
    #recall_test.append(np.mean(ret_test))
    #precision_test.append(np.mean(prt_test))
    #Acc_test.append(np.mean(acc_test))

    print('Epoch: {}\ntrain loss: {:.6f}\ntest loss: {:.6f}\nF1:{:.6f}\n'.format(
                        epoch,train_loss[-1],test_loss[-1], F1_test[-1]))

    #print('Epoch: {}\ntrain loss: {:.6f}\ntest loss: {:.6f}\n\nF1 train: {:.4f}\nF1 test: {:.4f}\n\nacc_train: {:.4f}\nacc_test: {:.4f}\n'.format(
    #                    epoch,train_loss[-1],test_loss[-1],F1_train[-1],F1_test[-1],Acc_train[-1],Acc_test[-1]))
                        
    scheduler.step()

    d = 0
    if test_loss[-1] < loss_min: 
        loss_min = min(test_loss)
        d += 1
        torch.save(attr_net.state_dict(), os.path.join(saving_path, 'best_attr_net.pth'))
        torch.save(epoch, os.path.join(saving_path, 'best_epoch.pth'))
        print('test loss improved')