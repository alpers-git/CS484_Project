#Part 3.1
#   Data Reading
import numpy as np
from PIL import Image
import cv2
import torch
import sys
sys.path.insert(0, 'data/')

import resnet

#image = cv2.imread('data/train/n01615121/n01615121_866.jpeg',1)
#cv2.imshow('image',image)
#cv2.waitKey(0)
#cv2.destroyAllWindows()

image = Image.open('data/train/n01615121/n01615121_1705.jpeg').convert('RGB')       #load an image
image_arr = np.asarray(image)                                                       #convert to a numpy array
org_size = image.size
max_dim = max(org_size[0], org_size[1])

#finds max dimension of image
padded_size = (max_dim, max_dim)
padded_im = Image.new("RGB", padded_size)
padded_im.paste(image, ( (int)((padded_size[0] - org_size[0])/2), 
                        (int)((padded_size[1] - org_size[1])/2) ))                  #pastes image in middle of a black image

padded_im = np.asarray(padded_im)
res_im = cv2.resize(padded_im, dsize=(224, 224), interpolation=cv2.INTER_LANCZOS4)  #resizes to 224x224
res_im = np.asarray(res_im)
res_im = res_im.astype(np.float32)

#Part 3.2
#   Feature extraction
resnet.resnet50()
#we append an augmented dimension to indicate batch_size, which is one
res_im = np.reshape(res_im, [1, 224, 224, 3])
# model takes as input images of size [batch_size, 3, im_height, im_width]
res_im = np.transpose(res_im, [0, 3, 1, 2])
# convert the Numpy image to torch.FloatTensor
res_im = torch.from_numpy(res_im)
# extract features
feature_vector = model(res_im)
# convert the features of type torch.FloatTensor to a Numpy array
# so that you can either work with them within the sklearn environment
# or save them as .mat files
feature_vector = feature_vector.numpy()

#for display and print
"""print(res_im.shape)
rgb_img = cv2.cvtColor(res_im, cv2.COLOR_BGR2RGB)                                   #do this or displayed img will have color channels in BGR order
cv2.imshow('image',rgb_img)
cv2.waitKey(0)                                                                      #press a random button to close window DO NOT USE RED X !!!
cv2.destroyAllWindows()"""


