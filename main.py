#Part 3.1
#   Data Reading
import numpy as np
from PIL import Image
import cv2

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
                        (int)((padded_size[1] - org_size[1])/2) ))                  #pastes image in middle of a plain black image

padded_im = np.asarray(padded_im)
res_im = cv2.resize(padded_im, dsize=(224, 224), interpolation=cv2.INTER_LANCZOS4)  #resizes to 224x224

#for display and print
print(res_im.shape)
rgb_img = cv2.cvtColor(res_im, cv2.COLOR_BGR2RGB)                                   #do this or displayed img will have color channels in BGR order
cv2.imshow('image',rgb_img)
cv2.waitKey(0)                                                                      #press a random button to close window DO NOT USE RED X !!!
cv2.destroyAllWindows()
