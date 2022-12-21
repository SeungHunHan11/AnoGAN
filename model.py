import torch
import torch.nn as nn

class generator(nn.Module):
    def __init__(self,latent_dim=100,last_dim=32,is_color=True,bias=True):
        super(generator,self).__init__()

        self.latent_dim=latent_dim
        self.is_color=True
        self.bias=bias
        self.last_dim=last_dim
        self.channels= 3 if self.is_color else 1

        self.model=nn.Sequential(
                                nn.ConvTranspose2d(self.latent_dim,self.last_dim*8,4,1,0,bias=self.bias),
                                nn.BatchNorm2d(self.last_dim*8),
                                nn.ReLU(),

                                nn.ConvTranspose2d(self.last_dim*8, self.last_dim*4, 4, 2, 1, bias=self.bias),
                                nn.BatchNorm2d(self.last_dim*4),
                                nn.ReLU(),
                                
                                nn.ConvTranspose2d(self.last_dim*4, self.last_dim*2, 4, 2, 1, bias=self.bias),
                                nn.BatchNorm2d(self.last_dim*2),
                                nn.ReLU(),
                                
                                nn.ConvTranspose2d(self.last_dim*2, self.last_dim, 4, 2, 1, bias=self.bias),
                                nn.BatchNorm2d(self.last_dim),
                                nn.ReLU(),

                                nn.ConvTranspose2d(self.last_dim, self.channels, 4, 2, 1, bias=self.bias),
                                nn.Tanh()
                                )
    def forward(self,x):
        output=self.model(x)
        return output

class Discriminator(nn.Module):
    def __init__(self, imgsize=32, is_color=True, bias=True):
        super(Discriminator, self).__init__()

        channels= 3 if is_color else 1

        self.feature_layer = nn.Sequential(
                                    nn.Conv2d(channels, imgsize, 4, 2, 1, bias=bias),
                                    nn.BatchNorm2d(imgsize),
                                    nn.LeakyReLU(0.2, inplace=True),

                                    nn.Conv2d(imgsize, imgsize*2, 4, 2, 1, bias=bias),
                                    nn.BatchNorm2d(imgsize*2),
                                    nn.LeakyReLU(0.2, inplace=True),
                                    
                                    nn.Conv2d(imgsize*2, imgsize*4, 4, 2, 1, bias=bias),
                                    nn.BatchNorm2d(imgsize*4),
                                    nn.LeakyReLU(0.2, inplace=True),
                                    
                                    nn.Conv2d(imgsize*4, imgsize*8, 4, 2, 1, bias=bias),
                                    nn.BatchNorm2d(imgsize*8),
                                    nn.LeakyReLU(0.2, inplace=True),
    
                                    )
        

        self.output_layer=nn.Sequential(
                                        nn.Conv2d(imgsize*8, 1, 4, 1, 0, bias=bias),
                                        nn.Sigmoid()
                                        )
    
    def forward_features(self,x):
        features=self.feature_layer(x)
        return features


    def forward(self, x):
        features= self.forward_features(x)
        
        discrimination = self.output_layer(features)
        return discrimination

class Encoder(nn.Module):
    def __init__(self,latent_dim=128,img_size=64,channels=3,bias=True):
        super(Encoder,self).__init__()

        def encoder_block(in_features, out_features, bn=True):
            if bn:
                block = [
                         nn.Conv2d(in_features, out_features, 4, 2, 1, bias=bias),
                         nn.BatchNorm2d(out_features),
                         nn.LeakyReLU(0.2, inplace=True)
                ]
            else:
                block = [
                         nn.Conv2d(in_features, out_features, 4, 2, 1, bias=bias),
                         nn.LeakyReLU(0.2, inplace=True)
                ]
            return block

        self.features = nn.Sequential(
            *encoder_block(channels, img_size, bn=False),
            *encoder_block(img_size, img_size*2, bn=True),
            *encoder_block(img_size*2, img_size*4, bn=True),
            *encoder_block(img_size*4, img_size*8, bn=True),
            nn.Conv2d(img_size*8, latent_dim, 4, 1, 0, bias=bias),
            nn.Tanh()
        )

    def forward(self, x):
        validity = self.features(x)
        return validity