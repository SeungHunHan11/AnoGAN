# ***AnoGAN Pytorch Implementation***

# Written By Seung Hun Han 
# Written In Dec/11/2022 
# Email: andrewhan@korea.ac.kr

## For AnoGAN train and inference

### Necessary Notices:


- Weight of a pretrained DCGAN, which used mnist dataset(where car, label=1, was treated as normal and others as abnormal) is available in [**Mnist Weights**] 
- Weight of a pretrained DCGAN which used Retina Abnormal dataset is available in [**Retina Weights**]
- Latent Vector optimization result for Retina Abnormal dataset is available in [**Latent Optimizaiton**]
- Retina Abnormal dataset is available in [**Retina Dataset**]

### Steps:

1. Install packages listed in [**requirements.txt**]
2. Divide dataset into normal and abnormal image folders
3. Run prompt given below

```python
python [train.py]
--trainimgs cropped_images/normal #Normal Image folder. Image should be in png format
--testimgs cropped_images/abnormal #Abnormal Image folder. Image should be in png format
--verbose True # Print Training process 
--generator model_saves/Generator_epoch_195.pt # Pretrained Generator weights. 
--discriminator model_saves/Discriminator_epoch_195.pt # Pretrained Discriminator weights (if available)
--trainmode True # Obtain trained GAN. By setting it as false, one will have to provide pretrained GAN in the command above
--epochs 300 # Epochs for training GAN
--iteration 800 # Iterations for individual image optimization
--savelogs true # Option whether to save images and weights or not.
```
- Upon finishing optimization, anomaly loss for each individual images will be saved into a .npy file.
- Beware that test image dataset, which will be used in latent vector optimization step, will be comprised of both normal (Randomly selected from folder provided in  ‘trainimgs’ option) and abnormal images.


## For f-AnoGAN train and inference

-

## Further guidance

For training and inference, refer to [guide.ipynb] 

For comparison of AnoGAN and F-AnoGAN results, refer to [**fanoganguide.ipynb**]

## Original Paper:
[**Unsupervised Anomaly Detection with Generative Adversarial Networks to Guide Marker Discovery**] (https://arxiv.org/abs/1703.05921) (Schlegl et al., 2017)

## Credit:
* https://github.com/mulkong/f-AnoGAN_with_Pytorch
* https://wsshin.tistory.com/m/4

