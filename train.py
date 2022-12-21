import torch
import torch.nn as nn
from torch.utils.data import DataLoader,Dataset
from torchvision.datasets import CIFAR10
import sys
import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
from torchvision.utils import make_grid
import argparse
from pathlib import Path
from glob import glob
from utils import *
import torch
from torch.utils.tensorboard import SummaryWriter
from torch.autograd import Variable

sys.path.append('./AnoGAN')

from model import *
from utils import custom_data,weights_init,display_image,transform_data,rawimage_dataset

__file__='train.py'
ROOT = Path(os.path.dirname(os.path.realpath('__file__'))).absolute()
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
ROOT = Path(os.path.abspath(os.path.join('/',ROOT))) 


class train(nn.Module):
    def __init__(self,device,latent_dim,is_color,img_size):
        super(train,self).__init__()

        '''
        'Define Class'
            Args:
                device: cuda or cpu
                latent_dim: Dimension of Latent Vector Z which is computed into generator to produce fake image
                img_size: Size of image being computed
                is_color: Configures number of channel 3 for colors 1 for grey
        '''
        self.device=device
        self.latent_dim=latent_dim
        self.is_color=is_color
        self.img_size=img_size
        self.D=None
        self.G=None

    def train_G_D(self,train_loader,lr=1E-3,bias=False,epochs=50,
                g_epochs=5,sample_size=10,verbose=False,save_logs=True
                ):

        '''
        'Fit Generator and Discriminator only using normal dataset'
            Args:
                Train_loader: DataLoader which should cosist of normal (with no anomaly) dataset
                lr: Learning rate for discriminator and generator
                latent_dim: Dimension of Latent Vector Z which is computed into generator to produce fake image
                img_size: Size of image being computed
                is_color: Configures number of channel 3 for colors 1 for grey
                bias: add bias
                Epochs: Epochs for training
                g_epochs: Number of generator backpropagation per one epoch. This setup is to prevent underfitting of generator compared to discriminator.
                sample_size: Number of images which will be sampled to create training progress image 
                Verbose: Print iteration progress
            Returns:
                Trained Generator
                Trained Discriminator
        '''
        if save_logs:
            x=0
            while True:
                increment='' if x==0 else str(x)
                dir_name='./train_logs'+increment+'/'
                if not os.path.exists(dir_name):
                    os.makedirs(dir_name,exist_ok=True)
                    break
                x+=1
            
            os.makedirs(dir_name+'/model_saves')
            print('Training Logs will be saved to'+dir_name)

        device=self.device
        latent_dim=self.latent_dim
        is_color=self.is_color
        img_size=self.img_size
        

        self.G = generator(latent_dim=latent_dim, last_dim=img_size, is_color=is_color, bias=bias).to(device)
        self.G=self.G.apply(weights_init)

        self.D = Discriminator(imgsize=img_size, is_color=is_color, bias=bias).to(device)
        self.D=self.D.apply(weights_init)
        criterion = nn.BCELoss()

        optimizer_G = torch.optim.Adam(self.G.parameters(), lr=lr, weight_decay=1e-5, betas=(0.5, 0.999))
        optimizer_D = torch.optim.Adam(self.D.parameters(), lr=lr, weight_decay=1e-5, betas=(0.5, 0.999))
        scheduler_G = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer_G, T_0=15, T_mult=2)
        scheduler_D = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer_D, T_0=15, T_mult=2)

        #writer = SummaryWriter()

        test_z = torch.randn(sample_size,latent_dim,1,1,device=device)

        best_loss=100000

        if verbose:
            train_loader=tqdm(train_loader)

        sample_loader=next(iter(train_loader))[0]
        batch=sample_loader.shape[0]
        idx=torch.randint(0,batch,(sample_size,),)
        real_sampled=sample_loader[idx]

        real_sampled=real_sampled.to(device)

        self.D.train()
        self.G.train()
        steps_per_epoch=len(train_loader)


        for epoch in range(epochs):
            best_renewed=False
            for idx, (xx,_) in enumerate(train_loader):
                real_images=xx.to(device)
                batch_size=real_images.shape[0]

                #Train Discriminator
                optimizer_D.zero_grad()

                real_output=self.D(real_images)
                real_labels=torch.ones_like(real_output,device=device)

                z = torch.randn(batch_size,latent_dim,1,1,device=device)

                fake_img=self.G(z).detach()    
                fake_output=self.D(fake_img)
                fake_labels=torch.zeros_like(fake_output,device=device)

                real_loss_D=criterion(real_output,real_labels)
                fake_loss_D=criterion(fake_output,fake_labels)

                loss_D= fake_loss_D+real_loss_D

                loss_D.backward()
                optimizer_D.step()

                
                real_prob=real_output.mean().item()
                fake_prob=fake_output.mean().item()

                #Train Generator

                for _ in range(g_epochs):
                    optimizer_G.zero_grad()
                    fake_img=self.G(z) # It is paramount that generator and discriminator are trained 'seperately'.
                    fake_output=self.D(fake_img)
                    fake_labels_G=torch.ones_like(fake_output,device=device)
                    loss_G=criterion(fake_output,fake_labels_G)
                    loss_G.backward(retain_graph=True)
                    optimizer_G.step()


            with torch.no_grad():
                test_img=self.G(z)
                dis_acc=self.D(real_sampled).flatten().detach().cpu()
                fake_img_dis=self.D(test_img).flatten().detach().cpu()
                

            scheduler_D.step(epoch + 1 + idx * steps_per_epoch)
            scheduler_G.step(epoch + 1 + idx * steps_per_epoch)

            if save_logs:
                display_image(real_img=real_sampled.detach().cpu(),
                            generated_img=test_img.detach().cpu(),epoch=epoch,
                            batch_size=batch_size,sample_size=sample_size,
                            g_loss=loss_G.item(),d_loss=loss_D.item(),
                            Dis_acc=dis_acc,fake_acc=fake_img_dis,save_path=dir_name)

            if best_loss>loss_G.item():
                best_loss=loss_G.item()
                if save_logs:
                    torch.save(self.D.state_dict(),dir_name+'/model_saves'+'/Discriminator_best.pt')
                    torch.save(self.G.state_dict(),dir_name+'/model_saves'+'/Generator_best.pt')
                best_renewed=True

            prompt=f'''
                    Training In Progress

                    Now at {epoch+1} / {epochs} Epoch 
                    D_Loss: {loss_D.item():.4f} G_Loss: {loss_G.item():.4f}

                    '''

            if best_renewed:
                prompt=prompt+'\nBest Renewed.'

            if epoch%5==0 and save_logs:
                torch.save(self.D.state_dict(),dir_name+'/model_saves'+'/Discriminator_epoch_{}.pt'.format(epoch))
                torch.save(self.G.state_dict(),dir_name+'/model_saves'+'/Generator_epoch_{}.pt'.format(epoch))
                print('Latest weight saved')

            print(prompt)
            
            #writer.add_scalar("Discriminator_Loss/train", loss_D.item(), epoch+1)
            #writer.add_scalar("Generator_Loss/train", loss_G.item(), epoch+1)
        
        #writer.close()

        return self.G, self.D

    def latent_mapping(self,test_loader,latent_dim=None,alpha=0.1,D=None,G=None,lr=1E-3,iteration=400,save_logs=True):
        '''
        'Train a latent vector that can be mapped to the computed image and calculate anomaly score'
            Args:
                test_loader: DataLoader which should cosist of normal and abnormal dataset
                lr: Learning rate for latent vector optimizer
                alpha: Weight used to calculate anomaly score.
                D: Trained discriminator. If undefined, discriminator trained in train_G_D will be used, if ever been called before.
                D: Trained generator. If undefined, generator trained in train_G_D will be used, if ever been called before.
                Iteration: Number of backpropagation to optimize one single image

            Returns:

                Original image vector in the test loader
                Original label vector in the test loader
                Fully trained latent vector collections
                Anomaly score for each images
        '''

        if save_logs:
            x=0
            while True:
                increment='' if x==0 else str(x)
                dir_name='./test_logs'+increment+'/'
                if not os.path.exists(dir_name):
                    os.makedirs(dir_name,exist_ok=True)
                    break
                x+=1
            print('Inference Logs will be saved to'+dir_name)

        
        if D !=None:
            Dis=D
        else:
            if self.D!=None:
                Dis=self.D
            else:
                print('Please compute Pretrained Discriminator')

        if D !=None:
            Gen=G
        else:
            if self.G!=None:
                Gen=self.G
            else:
                print('Please compute latent_dim')
        
        if latent_dim !=None:
            latent_dim=latent_dim
        else:
            if self.latent_dim!=None:
                latent_dim=self.latent_dim
            else:
                print('Please compute latent_dim')

        if len(test_loader)!=1:
            batch_size=next(iter(test_loader))[0].shape[0]
        else:
            batch_size=1

        channels=3 if self.is_color else 1

        channel=test_loader.dataset[0][0].shape[0]

        if channel!=channels:
            print('Input image channel not compatible with pretrained Discriminator')
        
        z=torch.randn(batch_size,latent_dim,1,1,device=self.device,requires_grad=True)
        optimizer_z = torch.optim.Adam([z], lr=lr)


        Gen.eval()
        Dis.eval()

        loss=[]
        latent_collection=[]
        xx_original=[]
        yy_original=[]


        for idx, (xx,yy) in enumerate(test_loader):
            real_img=xx.to(self.device)
            for i in range(iteration):

                optimizer_z.zero_grad()

                fake_img=Gen(z)
                fake_img_dis=Dis.forward_features(fake_img)
                real_img_dis=Dis.forward_features(real_img)

                anomaly_loss=estimate_anomaly_score(real_img,fake_img_dis,
                                    real_img_dis,fake_img,alpha
                                    )
                anomaly_loss.backward(retain_graph=True)
                optimizer_z.step()


                if i%200==0:
                    print(f'Loss for {idx+1} Batch / {len(test_loader)} at {i} iteration is {anomaly_loss.item():.4f}')

            fitted_z=z
            latent_collection.append(fitted_z.detach().cpu().data.numpy())
            xx_original.append(xx.numpy())
            yy_original.append(yy.numpy())

            with torch.no_grad():
                fake_img=Gen(fitted_z)
                fake_img_dis=Dis.forward_features(fake_img)
                real_img_dis=Dis.forward_features(real_img)

                final_anomaly=estimate_anomaly_score(real_img,fake_img_dis,
                                    real_img_dis,fake_img,alpha)

                loss.append(final_anomaly.detach().cpu().numpy())

            if save_logs:
                plot_test(xx,fake_img,idx,label=yy.item(),save_path=dir_name)

            print(f'Final Anomaly Loss for Batch {idx+1} / {len(test_loader)} is {final_anomaly.item():.4f}')


        latent_space = np.array(latent_collection)

        if save_logs:
            np.save(dir_name+'/loss_saves',loss)
            np.save(dir_name+'/z_saves',latent_space)
            np.save(dir_name+'/xx_saves',xx_original)
            np.save(dir_name+'/yy_saves',yy_original)

            print('Data Saved')

        print('Optimization Completed')

        return xx_original,yy_original,latent_space,loss


if __name__=='__main__':
    
    parser = argparse.ArgumentParser()

    parser.add_argument(
                    '--img_size',
                    type=int,
                    default=64,
                    help='Input image size'
                    )

    parser.add_argument(
        '--device',
        type=str,
        default='cuda',
        help='cuda for gpu else cpu'
        )

    parser.add_argument(
        '--latent_dim',
        type=int,
        default=128,
        help='Dimension of Latent Vector Z which is computed into generator to produce fake image'
        )

    parser.add_argument(
        '--trainimgs',
        type=str,
        help='Path for normal images')

    parser.add_argument(
        '--testimgs',
        type=str,
        help='Path for abnormal images')

    parser.add_argument(
        '--epochs',
        type=int,
        default=50,
        help='Epochs for training Generator and Discriminator'
        )

    parser.add_argument(
        '--verbose',
        type=bool,
        default=False,
        help='Print Training Log?'
        )

    parser.add_argument(
        '--iscolor',
        type=bool,
        default=True,
        help='Are images colored?'
        )

    parser.add_argument(
        '--samplesize',
        type=int,
        default=10,
        help='Number of images which will be sampled to track training progress'
        )

    parser.add_argument(
        '--trainbatchsize',
        type=int,
        default=128,
        help='Train Batchsize'
        )

    parser.add_argument(
        '--testbatchsize',
        type=int,
        default=1,
        help='Test Batchsize'
        )

    parser.add_argument(
                    '--alpha',
                    type=float,
                    default=0.1,
                    help=''
                    )
    parser.add_argument(
                    '--iteration',
                    type=int,
                    default=400,
                    help='Iteration for image optimization'
                    )

    parser.add_argument(
                    '--trainmode',
                    type=str,
                    default='True',
                    help='Train DCGAN? Or make an inference'
                    )

    parser.add_argument(
                    '--generator',
                    type=str,
                    default=None,
                    help='path for pretrained generator weights'
                    )
    parser.add_argument(
                '--discriminator',
                type=str,
                default=None,
                help='path for pretrained discriminator weights'
                )

    parser.add_argument(
        '--savelogs',
        type=str,
        default='True',
        help='Save imgs and data'
        )

    args=vars(parser.parse_args())

    train_path=os.path.join(ROOT/args['trainimgs'])
    test_path=os.path.join(ROOT/args['testimgs'])


    img_size=args['img_size']
    device= torch.device('cuda') if args['device']=='cuda' and torch.cuda.is_available() else torch.device('cpu')
    epochs=args['epochs']
    latent_dim=args['latent_dim']
    verbose=args['verbose']
    is_color=args['iscolor']
    sample_size=args['samplesize']
    trainbatchsize=args['trainbatchsize']
    testbatchsize=args['testbatchsize']
    train_mode=True if args['trainmode'].lower()=='true' else False
    iteration=args['iteration']
    alpha=args['alpha']
    savelogs=True if args['savelogs'].lower()=='true' else False
    lr=1E-3

    transform=transform_data(img_size)

    trainset_path=glob(os.path.join(train_path+'/*.png'))

    train_cut=round(len(trainset_path)*0.7)
    train_label=np.zeros((len(trainset_path[:train_cut]),))

    train_dataset2=rawimage_dataset(trainset_path[:train_cut],train_label,transform=transform)
    train_loader = DataLoader(train_dataset2, batch_size=trainbatchsize, shuffle=True)
    

    test_path=glob(os.path.join(test_path+'/*.png'))

    test_labels=np.append(np.zeros((len(trainset_path[train_cut:]),)),np.ones((len(test_path),1)))

    test_path=test_path+trainset_path[train_cut:]

    test_dataset2=rawimage_dataset(test_path,test_labels,transform=transform)
    test_loader = DataLoader(test_dataset2, batch_size=testbatchsize, shuffle=True)
    
    C=train(device=device,latent_dim=latent_dim,is_color=is_color,img_size=img_size)

    if train_mode:
        G,D=C.train_G_D(train_loader=train_loader,g_epochs=10,epochs=epochs,
                    verbose=verbose,sample_size=sample_size,
                    lr=lr,save_logs=savelogs)
    else:
        G = generator(latent_dim=latent_dim, last_dim=img_size, is_color=is_color, bias=False).to(device)
        D = Discriminator(imgsize=img_size, is_color=is_color, bias=False).to(device)
        G.load_state_dict(torch.load(args['generator']))
        D.load_state_dict(torch.load(args['discriminator']))
    
    xx,yy,z,loss=C.latent_mapping(test_loader,alpha=alpha,D=D,G=G,
                                    iteration=iteration,save_logs=savelogs)



  ###################################################