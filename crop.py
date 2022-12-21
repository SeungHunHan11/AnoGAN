import cv2
from keras.utils import normalize
import os
import tensorflow as tf
import numpy as np
from glob import glob
import warnings
from tqdm import tqdm
warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt

train_images=[]

def load_image(img_path,pic_size):
    SIZE_X, SIZE_Y = pic_size
    for img_path in tqdm(glob(os.path.join(img_path, "*.jpeg"))):
        img = cv2.imread(img_path, 0)       
        img = cv2.resize(img, (SIZE_Y, SIZE_X))
        train_images.append(img)
    return train_images

def save_image(data,name,path):
    height = float(data.shape[0])
    width = float(data.shape[1])
     
    fig = plt.figure()
    fig.set_size_inches(width/height, 1, forward=False)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
 
    ax.imshow(data)
    plt.savefig(path+name+'.png', dpi = height) 
    plt.close(fig)

def crop(im_vec,is_normal=True,model=None):

    label='normal' if is_normal else 'abnormal'

    train_images = im_vec

    im_vec=None

    train_images2 = np.expand_dims(train_images, axis=3)
    train_images2 = normalize(train_images, axis=1)
    print('Fitting on Images for optimal crop')
    prediction = (model.predict(train_images2))

    train_images2=None

    predicted_img2 = np.argmax(prediction, axis=3)

    prediction=None

    xl=list(map(lambda x: x[0][1],np.array(list(map(np.nonzero,predicted_img2)))))
    xr=list(map(lambda x: max(x[0]),np.array(list(map(np.nonzero,predicted_img2)))))
    yl=list(map(lambda x: min(x[1]),np.array(list(map(np.nonzero,predicted_img2)))))
    yr=list(map(lambda x: max(x[1]),np.array(list(map(np.nonzero,predicted_img2)))))
    new_img=[x[xl[idx]:xr[idx], yl[idx]:yr[idx]] for idx,x in enumerate(train_images)]

    os.makedirs('./cropped_images/normal/',exist_ok=True)
    os.makedirs('./cropped_images/abnormal/',exist_ok=True)

    for idx, x in enumerate(new_img):
        name=label+'_'+str(idx)
        path= './cropped_images/normal/' if is_normal else './cropped_images/abnormal/'
        save_image(x,name,path)

    return new_img