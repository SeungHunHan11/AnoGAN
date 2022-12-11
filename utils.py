import torch
from torch.utils.data import Dataset
import numpy as np
import matplotlib.pyplot as plt
import os
from torchvision.utils import make_grid
from torchvision import transforms
from torch.autograd import Variable

def transform_data(image_size):
    transform=transforms.Compose(
            [
            transforms.ToPILImage(),
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
    return transform

class rawimage_dataset(Dataset):
    def __init__(self,path_list,transform=None):
        super(custom_data,self).__init__()

        self.path_list=path_list
        self.transform=transform

    def __len__(self):
        return(len(self.path_list))
    
    def __getitem__(self, idx):
        img=self.path_list[idx]
        img_vec=np.transpose(plt.imread(img),(2,0,1))

        if self.transform:
            img_vec=self.transform(img_vec)

        return img_vec

class custom_data(Dataset):
    def __init__(self,data,labels,transform=None):
        self.data=data
        self.labels=labels
        self.transform=transform
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self,idx):
        data=np.transpose(self.data[idx],(2,0,1))
        labels=self.labels[idx]

        if self.transform:
            data=self.transform(data)

        return data, labels

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.normal_(m.weight, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        torch.nn.init.normal_(m.weight, 1.0, 0.02)
        torch.nn.init.zeros_(m.bias)

def display_image(real_img,generated_img,epoch=None,g_loss=None,d_loss=None,
                    batch_size=128,sample_size=10,Dis_acc=None,fake_acc=None):

    os.makedirs('./train_img/',exist_ok=True)

    realimg=real_img

    generatedimg=generated_img

    fig=plt.figure(figsize=(sample_size,5))
    subfigs = fig.subfigures(2, 1, wspace=0.07)

    axesup=subfigs[0].subplots(1,sample_size,sharey=True)
    axesdown=subfigs[1].subplots(1,sample_size,sharey=True)

    subfigs[0].suptitle(f'Training Result at {epoch} Epoch Epoch Generator Loss: {g_loss:.4f} Discriminator Loss: {d_loss:.4f} \n Real Image')
    subfigs[1].suptitle('Generated Image')

    for i in range(sample_size):
        
        axesup[i].imshow(make_grid(realimg[i],normalize=True,padding=5).permute((1,2,0)))
        #axesup[i].get_xaxis().set_visible(False)
        axesup[i].get_yaxis().set_visible(False)
        axesup[i].set_title(f"{i+1} \n")
        axesup[i].set_xlabel(f'Acc: {Dis_acc[i]:.3f}')
        #axesup[i].axis('off')

        axesdown[i].imshow(np.transpose(make_grid(generatedimg[i],normalize=True,padding=2).numpy(),(1,2,0)))
        #axesdown[i].get_xaxis().set_visible(False)
        axesdown[i].get_yaxis().set_visible(False)
        axesdown[i].set_title("{}".format(i+1))
        axesdown[i].set_xlabel(f'Acc: {fake_acc[i]:.3f}')

        #axesdown[i].axis('off')
    
    plt.tight_layout()
    plt.savefig('./train_img/epoch_{}_train.png'.format(epoch), dpi=300,facecolor='white')
    plt.close(fig)
    #plt cookbook https://matplotlib.org/stable/gallery/subplots_axes_and_figures/subfigures.html


def residual_loss(real_images, generated_images):
    subtract = real_images - generated_images

    return torch.sum(torch.abs(subtract))

def discriminator_loss(Discrimination_fake, real_images_dis):

    subtract = Discrimination_fake - real_images_dis

    return torch.sum(torch.abs(subtract))

def estimate_anomaly_score(real_images,Discrimination_fake,real_images_dis, generated_images,alpha):
    real_images = real_images
    generated_images = generated_images

    resi_loss = residual_loss(real_images, generated_images)
    disc_loss = discriminator_loss(Discrimination_fake, real_images_dis)
    ano_loss = (1 - alpha) * resi_loss + alpha * disc_loss

    return torch.sqrt(ano_loss)

def plot_test(input,fake_img_vec,idx=None):

    fig, axs = plt.subplots(1, 2)

    axs[0].title.set_text(f'Batch {idx+1} Original')
    axs[1].title.set_text(f'Batch {idx+1} Generated')

    axs[0].imshow(make_grid(input,normalize=True,padding=3).permute((1,2,0)))
    axs[1].imshow(make_grid(fake_img_vec.detach().cpu(),normalize=True,padding=3).permute((1,2,0)))

    plt.tight_layout()
    plt.savefig(f'./test_imgs/test_img_{idx+1}.png', dpi=300,facecolor='white')
    plt.close(fig)