# ***AnoGAN Pytorch Implementation***

# Written By Seung Hun Han 
# Written In Dec/11/2022 
# Email: andrewhan@korea.ac.kr

## For AnoGAN train and inference

### Necessary Notices:


- Weight of a pretrained DCGAN, which used mnist dataset(where car, label=1, was treated as normal and others as abnormal) is available in [**Mnist Weights**](https://github.com/SeungHunHan11/AnoGAN/tree/master/Mnist_weights/permanent)
- Weight of a pretrained DCGAN which used Retina Abnormal dataset is available in [**Retina Weights**](https://github.com/SeungHunHan11/AnoGAN/tree/master/retina_train/model_saves)
- Latent Vector optimization result for Retina Abnormal dataset is available in [**Latent Optimizaiton**](https://github.com/SeungHunHan11/AnoGAN/tree/master/retina_inference_result)
- Retina Abnormal dataset is available in [**Retina Dataset**](https://github.com/SeungHunHan11/AnoGAN/tree/master/cropped_images)

### Steps:

1. Install packages listed in [**requirements.txt**](https://github.com/SeungHunHan11/AnoGAN/blob/master/requirements.txt)
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

```python
python train_fanogan.py
--trainimgs cropped_images/normal # Normal image set for encoder training. 70% of the dataset will be utilized for training. Rest will comprise test image set
--generator retina_train/model_saves/Generator_epoch_295.pt # Pretrained Generator for Retina Dataset
--discriminator retina_train/model_saves/Discriminator_epoch_295.pt # Pretrained Discriminator for Retina Dataset
--trainbatchsize 16 # Train dataset batch size
```

- Use trained encoder to produce generated image (i.e. G(E(x)), where x is an original image, E(x)=z).
- Add up discriminator loss and generation loss using the image generated via encoder.
- Trained encoder weight is available in [**Encoder Weight**](https://github.com/SeungHunHan11/AnoGAN/blob/master/f-anogan/logs/model_saves/Encoder.pt)
- Refer to [**f-AnoGAN Guide**](https://github.com/SeungHunHan11/AnoGAN/blob/master/fanoganguid.ipynb) for Inference 

## Further guidance

For training and inference, refer to [**guide.ipynb**](https://github.com/SeungHunHan11/AnoGAN/blob/master/guide.ipynb)

For comparison of AnoGAN and F-AnoGAN results, refer to [**fanoganguide.ipynb**](https://github.com/SeungHunHan11/AnoGAN/blob/master/fanoganguid.ipynb)

## Original Paper:
[**Unsupervised Anomaly Detection with Generative Adversarial Networks to Guide Marker Discovery**](https://arxiv.org/abs/1703.05921) (Schlegl et al., 2017)

## Credit:
* https://github.com/mulkong/f-AnoGAN_with_Pytorch
* https://wsshin.tistory.com/m/4

