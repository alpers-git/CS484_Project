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
#Data Reading
#Getting the training data
#Can be converted to the function
for dirName, subdirList, fileList in os.walk(rootDir):
    #print('Found directory: %s' % dirName)
    class_name = dirName[len(rootDir):]
    print('\t%s' % class_name) 
    for fname in fileList:
        #print('\t%s' % fname) 
        image = Image.open(dirName + '/' + fname).convert('RGB')    
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
        padded_im.paste(image, ((int)((padded_size[0] - org_size[0])/2), 
                                (int)((padded_size[1] - org_size[1])/2)))                  #pastes image in middle of a black image

        padded_im = np.asarray(padded_im)
        res_im = cv2.resize(padded_im, dsize=(224, 224), interpolation=cv2.INTER_LANCZOS4)  #resizes to 224x224
        res_im = np.asarray(res_im)
        res_im = res_im.astype(np.float32)
        
        #Part 3.2
        #Feature extraction
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
clf = svm.SVC(gamma='scale')
clf.fit(t_feature_vectors, t_class_names)  
#Exporting The genereted SVM file just in case, may be its unneccessary later
filename = 'machine.svm'
pickle.dump(clf, open(filename, 'wb'))
 
#Part 5
rootDirTest = 'data/test/'

image = cv2.imread(rootDirTest+ '/' + 'images/0.jpeg')
edge_detection = cv2.ximgproc.createStructuredEdgeDetection(rootDirTest + "model.yml.gz") # SHOULD THIS MODEL BE OUR MODEL ?????!!!!

rgb_im = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
edges = edge_detection.detectEdges(np.float32(rgb_im) / 255.0)

orimap = edge_detection.computeOrientation(edges)
edges = edge_detection.edgesNms(edges, orimap)

edge_boxes = cv2.ximgproc.createEdgeBoxes()
edge_boxes.setMaxBoxes(50)
boxes = edge_boxes.getBoundingBoxes(edges, orimap)

for b in boxes:
        x, y, w, h = b
        cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 1, cv2.LINE_AA)

cv2.imshow("edges", edges)
cv2.imshow("edgeboxes", image)
cv2.waitKey(0)
cv2.destroyAllWindows()

"""alpha = 0.65
beta = 0.75
eta = 1
minScore = 0.01
maxBoxes = 10000
edgeMinMag = 0.1
edgeMergeThr = 0.5
clusterMinMag = 0.5
maxAspectRatio = 3
minBoxArea = 1000
gamma = 2
kappa = 1.5  

retval = cv2.ximgproc.createEdgeBoxes( alpha, beta, eta, minScore, maxBoxes, edgeMinMag, edgeMergeThr, clusterMinMag, maxAspectRatio, minBoxArea, gamma, kappa)
print(retval)"""