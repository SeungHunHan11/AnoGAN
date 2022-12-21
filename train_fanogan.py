from model import *
import os
from utils import weights_init
import argparse
from utils import *
from glob import glob
import numpy as np
from torch.utils.data import DataLoader,Dataset


def train_f_encoder(G=None,D=None,epochs=100,kappa=1.0,
                    lr=0.001,train_loader=None,device=None,
                    latent_dim=128,channels=3,img_size=64,save_logs=True): 
    
    if save_logs:
        x=0
        while True:
            increment='' if x==0 else str(x)
            dir_name='./f-anogan/logs'+increment+'/'
            if not os.path.exists(dir_name):
                os.makedirs(dir_name,exist_ok=True)
                break
            x+=1
        os.makedirs(dir_name+'/model_saves',exist_ok=True)

    G.to(device)
    D.to(device)
    G.eval()
    D.eval()

    E = Encoder(latent_dim, img_size, 3, bias=False).to(device)
    E=E.apply(weights_init)
    E.train()

    criterion = nn.MSELoss()
    optimizer_E = torch.optim.Adam(E.parameters(), lr=lr, betas=(0.0, 0.999))

    record_loss=[]
    best_loss=1000

    for epoch in range(epochs):
        eloss_epoch=0

        for idx, (img,_) in enumerate(train_loader):
            real_imgs = img.to(device)
        
            optimizer_E.zero_grad() 

            z = E(real_imgs)

            fake_imgs = G(z)
            real_features = D.forward_features(real_imgs)
            fake_features = D.forward_features(fake_imgs)

            
            # izif architecture
            loss_imgs = criterion(fake_imgs, real_imgs)
            loss_features = criterion(fake_features, real_features)
            e_loss = loss_imgs + kappa * loss_features

            e_loss.backward()
            optimizer_E.step()

            eloss_epoch+=e_loss.item()
            
            if idx % 5 ==0:
                print(f'Encoder Loss at {epoch+1} epoch iteration {idx} is {e_loss.item():.4f}')
        
        epochloss=eloss_epoch/len(train_loader)
        record_loss.append(epochloss)

        if best_loss>epochloss:
            best_loss=epochloss
            if save_logs:
                torch.save(E.state_dict(),dir_name+'/model_saves'+'/Encoder.pt')
            print(f"Best Model Renewed at {epoch+1} with {best_loss:.4f}")

    return E, record_loss


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    
    parser.add_argument(
                        '--generator',
                        type=str,
                        default=None,
                        help='Set pretrained generator weight path'
                        )
    
    parser.add_argument(
                        '--discriminator',
                        type=str,
                        default=None,
                        help='Set pretrained discriminator weight path'
                        )
    parser.add_argument(
                    '--latent_dim',
                    type=int,
                    default=128,
                    help='Dimension of latent vector Z'
                    )
    parser.add_argument(
                '--channels',
                type=int,
                default=3,
                help='Channel of an image'
                )
    parser.add_argument(
                '--imgsize',
                type=int,
                default=64,
                help='Image size'
                )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda',
        help='cuda for gpu else cpu'
        )
    parser.add_argument(
        '--trainimgs',
        type=str,
        help='Path for normal images')
    parser.add_argument(
        '--epochs',
        type=int,
        default=50,
        help='Epochs for training Generator and Discriminator'
    )
    parser.add_argument(
    '--savelogs',
    type=str,
    default='True',
    help='Save Encoder weight'
    )
    parser.add_argument(
        '--kappa',
        type=float,
        default=1.0,
        help='Ratio for anomaly score'
    )
    parser.add_argument(
    '--trainbatchsize',
    type=int,
    default=128,
    help='Train Dataset batchsize'
    )
    
    args=vars(parser.parse_args())
    latent_dim=args['latent_dim']
    channels=args['channels']
    is_color=True if channels==3 else False
    img_size=args['imgsize']
    device=torch.device('cuda') if args['device']=='cuda' else torch.device('cpu')
    epochs=args['epochs']
    savelogs=True if args['savelogs'].lower()=='true' else False
    kappa=args['kappa']
    
    train_batch_size=args['trainbatchsize']
    train_path=args['trainimgs']
    
    
    transform=transform_data(img_size)
    trainset_path=glob(os.path.join(train_path+'/*.png'))

    train_cut=round(len(trainset_path)*0.7)
    train_label=np.zeros((len(trainset_path[:train_cut]),))

    train_dataset2=rawimage_dataset(trainset_path[:train_cut],train_label,transform=transform)
    train_loader = DataLoader(train_dataset2, batch_size=train_batch_size, shuffle=True)
    
    G = generator(latent_dim=latent_dim, last_dim=img_size, is_color=is_color, bias=False).to(device)
    D = Discriminator(imgsize=img_size, is_color=is_color, bias=False).to(device)
    G.load_state_dict(torch.load(args['generator']))
    D.load_state_dict(torch.load(args['discriminator']))
    
    E,record_loss=train_f_encoder(G=G,D=D,epochs=epochs,kappa=kappa,
                    lr=0.001,train_loader=train_loader,device=device,
                    latent_dim=latent_dim,channels=channels,img_size=img_size,save_logs=savelogs)
