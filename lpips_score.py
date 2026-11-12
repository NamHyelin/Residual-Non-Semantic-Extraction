# !pip install torchvision lpips

import torch
import numpy as np
import lpips
from PIL import Image
import torchvision
import torchvision.transforms as transforms
from skimage.metrics import structural_similarity as ssim
from skimage import io
from torchmetrics.image.fid import FrechetInceptionDistance
import pytorch_fid_wrapper as pfw
import cv2

import tensorflow as tf
import numpy as np
# from tensorflow.keras.applications.inception_v3 import InceptionV3
# from tensorflow.keras.applications.inception_v3 import preprocess_input
from torchmetrics.image.kid import KernelInceptionDistance




def LPIPS(image1, image2):
  lpips_model = lpips.LPIPS(net='alex')

  # # Load the images
  # image1 = Image.open('img/save/input_ref/content/input_ref/18.jpg').convert('RGB')
  # image2 = Image.open('/img/dog.png').convert('RGB')

  image1_np=np.array(image1)
  image2_np=np.array(image2)
  # image1_np=np.concatenate((np.expand_dims(image1_np[-1],0),image1_np),0)

  # Preprocess the images
  image1_tensor = [torch.from_numpy(image1_np).permute(2, 0, 1).float() / 255.]
  image2_tensor = [torch.from_numpy(image2_np).permute(2, 0, 1).float() / 255.]

  # Transform the images to tensors and normalize
  transform = transforms.Compose([
      transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
      transforms.Resize((256,256))
  ])

  # Convert the images to tensors and add a batch dimension
  image1_tensor = transform(image1_tensor[0])
  image2_tensor = transform(image2_tensor[0])
  lpips_score = lpips_model.forward(image1_tensor, image2_tensor).item()

  return lpips_score




def LPIPS_MNIST(image1, image2):
  # lpips_model = lpips.LPIPS(net='alex')

  image1_np = np.array(image1)
  image2_np = np.array(image2)

  if image1_np.shape[1] == 1:  # If grayscale with shape (batchsize, 1, H, W)
      image1 = image1.repeat(1, 3, 1, 1)
  if image2_np.shape[1] == 1:  # If grayscale with shape (batchsize, 1, H, W)
      image2 = image2.repeat(1, 3, 1, 1)


  # transform = transforms.Compose([
  #     transforms.Resize((256, 256)),
  #     transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
  # ])
  #
  # image1_tensor = torch.stack([transform(img.float() / 255.0) for img in image1])
  # image2_tensor = torch.stack([transform(img.float() / 255.0) for img in image2])
  #
  # lpips_scores = lpips_model.forward(image1_tensor, image2_tensor)
  # lpips_score = lpips_scores.view(-1).mean()
  lpips_score=LPIPS(image1, image2)

  return lpips_score






def SSIM(image1, image2):
  transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    transforms.Resize((256, 256)),
    transforms.ToPILImage()
  ])

  image1 = transform(image1)
  image2 = transform(image2)

  image1_np = np.array(image1)
  image2_np = np.array(image2)

  grayA = cv2.cvtColor(image1_np, cv2.COLOR_BGR2GRAY)
  grayB = cv2.cvtColor(image2_np, cv2.COLOR_BGR2GRAY)

  (ssim_score, diff) = ssim(grayA, grayB, full=True)
  diff = (diff * 255).astype("uint8")

  # # show difference
  # cv2.imshow('diff',diff)
  # cv2.waitKey(0)


  return ssim_score




def FID(image1, image2):
  transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((256, 256))
  ])
  image1_tensor = transform(image1).unsqueeze(0)
  image2_tensor = transform(image2).unsqueeze(0)

  pfw.set_config(batch_size=1, dims=192, device='cpu')  #64 192 768 2048
  '''BLOCK_INDEX_BY_DIM = {
    64: 0,  # First max pooling features, corresponds to output of first max pooling
    192: 1,  # Second max pooling featurs, corresponds to output of second max pooling
    768: 2,  # Pre-aux classifier features, corresponds to output which is fed to aux classifier
    2048: 3  # Final average pooling features, corresponds to output of final average pooling
  }'''
  fid_score= pfw.fid(image1_tensor, image2_tensor) #.expand(5,-1,-1,-1)
  return fid_score




'''
def calculate_inception_score(images, model=InceptionV3(weights='imagenet')):
  images_resized = tf.image.resize(images, (299, 299))
  images_preprocessed = (images_resized - 0.5) * 2.0
  preds = model.predict(images_preprocessed)

  # Calculate the class probabilities
  p_yx = np.exp(preds) / np.sum(np.exp(preds), axis=1, keepdims=True)
  # Calculate the marginal probability distribution
  p_y = np.mean(p_yx, axis=0)

  kl_divergence = np.sum(p_yx * (np.log(p_yx) - np.log(p_y)), axis=1)
  is_score = np.exp(np.mean(kl_divergence))

  return is_score


'''


def KID(img1, img2):
  from kid_notmain import calculate_kid_given_paths
  kid_score = calculate_kid_given_paths([getattr(img1, 'filename', None), getattr(img2, 'filename', None)], batch_size=1, cuda='cuda:0', dims=2048) #64 192 768 2048
  return kid_score



if __name__ == "__main__":
  image1 = Image.open('C:/Users/hyeli/Downloads/drive-download-20241030T092349Z-001/final_recon_img_ref1_d15.jpg') #img/save/input_ref/content/input_ref/10.jpg
  image2 = Image.open('C:/Users/hyeli/Downloads/drive-download-20241030T092349Z-001/final_recon_img_ref1_d3.jpg') #

  fid=LPIPS_MNIST(image1, image2)
  print(fid)
