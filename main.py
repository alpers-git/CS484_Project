#need to install opencv and pytorch, sklearn from the anaconda cli for desired environment 

#Image Processing etc.. Related imports
import numpy as np
import PIL
from PIL import Image
import cv2
import torch

#Imports for data reading
import sys
import os
sys.path.insert(0, 'data/')

#ResNet Model 
import resnet

#Sklearn reletaed imports
import sklearn
from sklearn import svm
import pickle

#The sample train data, this must be converted to the original directory after all process finished
rootDir = 'data/train2/'
model = resnet.resnet50()

#holds the class names for each picture from the parent folder 
t_class_names = []
#holds the feature vectors generated from the ResNet50 Model
t_feature_vectors = []

#Part 3.1
#   Data Reading
#Getting the training data
#Can be converted to the function
for dirName, subdirList, fileList in os.walk(rootDir):
    #print('Found directory: %s' % dirName)
    class_name = dirName[len(rootDir):]
    print('\t%s' % class_name) 
    for fname in fileList:
        #print('\t%s' % fname) 
        image = Image.open(dirName+ '/' +fname).convert('RGB')    
        #print(image)
        image_arr = np.asarray(image)   #convert to a numpy array
        rgb_img = cv2.cvtColor(image_arr, cv2.COLOR_BGR2RGB)  #do this or displayed img will have color channels in BGR order   

        #cv2.imshow('image',rgb_img)
        #cv2.waitKey(0)                                                                      #press a random button to close window DO NOT USE RED X !!!
        #cv2.destroyAllWindows()
                                                        
        org_size = image.size
        max_dim = max(org_size[0], org_size[1])

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
        #model = resnet.resnet50()
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
        feature_vector = feature_vector.detach().numpy() .ravel()
        
        t_class_names.append(class_name)
        t_feature_vectors.append(feature_vector)

#Part 4 Training SVM

# Must install the pip install -U scikit-learn to run this 
clf = svm.SVC(gamma='linear')
clf.fit(t_feature_vectors, t_class_names)  
#Exporting The genereted SVM file just in case, may be its unneccessary later
filename = 'machine.svm'
pickle.dump(clf, open(filename, 'wb'))
 
#Part 5
rootDirTest = 'data/test/'
