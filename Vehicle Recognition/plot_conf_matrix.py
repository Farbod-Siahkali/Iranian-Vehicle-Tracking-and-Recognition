from distutils.log import error
from attr import attr
import torch
from PIL import Image
import numpy as np
import torchvision.transforms as T
import os
import glob
from sklearn.metrics import precision_score, recall_score, f1_score

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

    output.update({'names' : ['206','207i','405','Arisun','Dena','HcCross','JackS5','Kara or MazdaPickup','L90','MVM315H',
                            'MVMX22','NeissanVanet','Pars','PeykanSavari','PeykanVanet','Pride131nasimsaba',
                            'Pride132and111','Pride141','PrideVanet151','Quik','RenaultPK','RioSD','Runna','Saina',
                            'Samand','SamandSoren','Shahin','Tiba','Xantia']}) #should change

    return output

def predict_images(model, image):
    image = Image.open(image)
    softmax = torch.nn.Softmax(dim=-1)
    image = preprocess(image)
    image = image.unsqueeze(0).to('cuda')
    pred = model.forward_attr_eval(image)
    soft_pred = softmax(pred.to('cpu'))
    conf, label = torch.max(soft_pred, dim=1)
    return conf, label



test_img_path = './Car/Data/SIVD/test'
attr_test = './Car/Data/SIVD/test.npy'

names =  ['206','207i','405','Arisun','Dena','HcCross','JackS5','Kara or Mazda Pickup','L90','MVM315H',
                            'MVMX22','Neissan Pickup','Pars','Peykan','Peykan Pickup','Pride 131 nasim or saba',
                            'Pride 132 or 111','Pride 141','Pride Pickup 151','Quik','Renault PK','RioSD','Runna','Saina',
                            'Samand','Samand Soren','Shahin','Tiba','Xantia']

'''model = build_model(
                    name='resnet50',
                    num_classes=29,
                    loss='softmax',
                    pretrained=False
                    )'''

'''trained_net = torch.load("./best_attr_net.pth")
model.load_state_dict(trained_net)

model.eval()

model.to('cuda')'''


transforms = []
transforms += [T.Resize((256, 256))]
transforms += [T.ToTensor()]
transforms += [T.Normalize(mean=[0.4611, 0.4658, 0.4728], std=[0.2552, 0.2502, 0.2520])]
preprocess = T.Compose(transforms)


test_images = glob.glob(test_img_path + '/*' + '/*')
test_attrs = np.load(attr_test) 

data_dict = dict()
result_dict = dict()


# label_mat = np.zeros((test_attrs.shape[1], test_attrs.shape[1]))

# y_pred = np.zeros((2, len(test_images)))


# for i, img in enumerate(test_images):
#     conf, pred_index = predict_images(model, img)
#     print(i,f"/{len(test_images)}")
#     true_index = np.where(test_attrs[i]==1)
#     y_pred[0][i] = pred_index
#     y_pred[1][i] = true_index[0]

#     label_mat[pred_index, true_index] += 1

import matplotlib.pyplot as plt


def plot_confusion_matrix(df_confusion, title='Confusion matrix', cmap=plt.cm.gray_r):
    plt.matshow(df_confusion, cmap=cmap) # imshow
    #plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(df_confusion.columns))
    plt.xticks(tick_marks, df_confusion.columns, rotation=45)
    plt.yticks(tick_marks, df_confusion.index)
    #plt.tight_layout()
    plt.ylabel(df_confusion.index.name)
    plt.xlabel(df_confusion.columns.name)

import pandas as pd
mm = pd.read_csv('ConfusionMatrix.csv')
#mm = mm.values
import seaborn as sns
ax= plt.subplot()
sns.heatmap(mm, annot=True, fmt='g',cmap="Blues", ax=ax, xticklabels=names, yticklabels=names);  #annot=True to annotate cells, ftm='g' to disable scientific notation
# labels, title and ticks
ax.set_xlabel('Predicted labels', font='Times New Roman', fontsize=13)
ax.set_ylabel('True labels', font='Times New Roman', fontsize=13)
ax.set_title('Confusion Matrix', font='Times New Roman', fontsize=15); 
#ax.xaxis.set_ticklabels(names, rotation=90); ax.yaxis.set_ticklabels(names, rotation=0);

plt.xticks(font='Times New Roman', fontsize=13)
plt.yticks(font='Times New Roman', fontsize=13)

plot_confusion_matrix(mm)
plt.show()

precision = precision_score(label_mat)
recall = precision_score(label_mat)
f1 = f1_score(label_mat)


for i , name in enumerate(names):
    result_dict[name] = {'precision': precision[i], 'recall': recall[i], 'f1':f1[i]}

print()