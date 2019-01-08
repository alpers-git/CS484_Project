import numpy as np
import PIL
from PIL import Image
import cv2
import torch
import sys
import os
import sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import pickle

sys.path.insert(0, 'data/')
import resnet

def preprocessImage(image): # input is of type PIL Image
    image_arr = np.asarray(image) # convert to a numpy array                            
    org_size = image_arr.shape
    max_dim = max(org_size[0], org_size[1])

    padded_size = (max_dim, max_dim)
    padded_im = Image.new("RGB", padded_size)
    padded_im.paste(image, ((int)((padded_size[0] - org_size[0])/2), (int)((padded_size[1] - org_size[1])/2))) #pastes image in middle of a black image
    padded_im = np.asarray(padded_im)
    res_im = cv2.resize(padded_im, dsize=(224, 224), interpolation=cv2.INTER_LANCZOS4)  # resize to 224x224 using Lanczos interpolation
    res_im = np.asarray(res_im)
    res_im = res_im.astype(np.float32)
    
    # Part 3.2
    # Feature extraction
    # append a dimension to indicate batch_size, which is one
    res_im = np.reshape(res_im, [1, 224, 224, 3])
    # model accepts input images of size [batch_size, 3, im_height, im_width]
    res_im = np.transpose(res_im, [0, 3, 1, 2])
    # convert the Numpy image to torch.FloatTensor
    res_im = torch.from_numpy(res_im)
    # extract features
    feature_vector = model(res_im)
    # convert the features of type torch.FloatTensor to a Numpy array
    # so that you can either work with them within the sklearn environment
    # or save them as .mat files
    feature_vector = feature_vector.detach().numpy().ravel()
    
    return feature_vector

# The sample train data, this must be converted to the original directory after all process finished
rootDir = 'data/train/'
model = resnet.resnet50()
model.eval()

# Test data
test_data = []
test_data_file = open("data/test/bounding_box.txt", "r")
for line in test_data_file:
    test_data.append(line.rstrip('\n').split(","))
test_data_file.close()
test_data = np.asarray(test_data)
test_labels = test_data[:, 0]
test_proposals = test_data[:, 1:].astype(np.int)

t_class_names = []
t_feature_vectors = []

# Part 3.1
# Data Reading
# Getting the training data
# Can be converted to the function
for dirName, subdirList, fileList in os.walk(rootDir):
    # print('Found directory: %s' % dirName)
    class_name = dirName[len(rootDir):]
    print('\t%s' % class_name) 
    for fname in fileList:
        # print('\t%s' % fname) 
        image = Image.open(dirName + '/' + fname).convert('RGB')    

        # apply preprocessing to the image
        feature_vector = preprocessImage(image)
        t_class_names.append(class_name)
        t_feature_vectors.append(feature_vector)

# Part 4: Training one-vs-all classifiers
# Normalize feature vectors
#scaler = StandardScaler() # (x - mean) / stddev
#scaler.fit(t_feature_vectors)
#t_feature_vectors = scaler.transform(t_feature_vectors)

# Train one-vs-all classifiers for each object type
bsvm = []
unique_labels = np.unique(t_class_names)
for i in unique_labels:
    # Set labels belonging to class i to 1, rest to 0
    new_train_labels = np.asarray(t_class_names)
    new_train_labels[new_train_labels != i] = 0

    # Train binary SVM
    # TODO: Experiment with gamma value
    bsvm_i = SVC(kernel = "rbf", gamma = "auto", probability=True)
    bsvm_i.fit(t_feature_vectors, new_train_labels)
    bsvm.append(bsvm_i)

print("Training complete. Starting test phase...")

# Part 5: Testing
# Read test labels & proposals
test_data_file = open("/test/bounding_box.txt", "r")
test_data = test_data_file.read().splitlines()
test_data_file.close()

rootDirTest = 'data/test/'
edge_detection = cv2.ximgproc.createStructuredEdgeDetection(rootDirTest + "model.yml.gz")
test_predictions = []
for i in range(10):
    print("Test image " + str(i))
    # 5.1: Create edge boxes for each test image
    image = cv2.imread(rootDirTest + 'images/' + str(i) + '.jpeg')    
    rgb_im = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    edges = edge_detection.detectEdges(np.float32(rgb_im) / 255.0)
    orimap = edge_detection.computeOrientation(edges)
    edges = edge_detection.edgesNms(edges, orimap)
    edge_boxes = cv2.ximgproc.createEdgeBoxes()
    edge_boxes.setMaxBoxes(50)
    boxes = edge_boxes.getBoundingBoxes(edges, orimap)
    
    #for b in boxes:
    #    x, y, w, h = b
    #    cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 1, cv2.LINE_AA)
    #cv2.imshow("edgeboxes", image)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()    

    pil_image = Image.open(rootDirTest + 'images/' + str(i) + '.jpeg').convert('RGB')
    test_features_i = []
    for b in boxes:
        x, y, w, h = b
        # cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 1, cv2.LINE_AA)
    
        # Repeat same preprocessing in Part 3
        subimage_b = pil_image.crop((x, y, x + w, y + h))
        # subimage_b.show()
        feature_vector = preprocessImage(subimage_b)
        test_features_i.append(feature_vector)

    test_features_i = np.asarray(test_features_i)

    # Normalize features
    #sc = StandardScaler() # (x - mean) / stddev
    #sc.fit(test_features_i)
    #test_features_i = sc.transform(test_features_i)

    # 5.2: Localization result
    predictions_i = []
    for j in range(len(bsvm)):
        preds = bsvm[j].predict_proba(test_features_i)
        predictions_i.append(preds[:, 1])
    
    predictions_i = np.asarray(predictions_i)
    best_prediction = np.unravel_index(np.argmax(predictions_i, axis=None), predictions_i.shape)
    test_predictions.append(best_prediction)

test_predictions = np.asarray(test_predictions)
print(test_predictions[:,0])