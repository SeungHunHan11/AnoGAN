from model import *
import os
from utils import weights_init

def train_f_encoder(G=None,D=None,epochs=100,kappa=1.0,
                    lr=0.001,train_loader=None,save_dir=None,device=None,
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
            torch.save(E.state_dict(),dir_name+'/model_saves'+'/Encoder.pt')
            print(f"Best Model Renewed at {epoch+1} with {best_loss:.4f}")

    return E, record_loss