import os
import numpy as np
from skimage import io, transform
from keras.models import load_model
from sklearn.metrics import accuracy_score
import cv2


IMAGE_SIZE = 50
DATA_DIR = "data"
TRAIN_DATA_FRACTION = 0.8


def test_train_split(data, labels, f):
    test_data_size = int(len(data) * f)
    return data[:test_data_size], labels[:test_data_size], \
        data[test_data_size:], labels[test_data_size:]


def transform_img(image):
    return transform.resize(image, (IMAGE_SIZE, IMAGE_SIZE, image.shape[2]))


def loadData():
    images = os.listdir(DATA_DIR)
    train_data = []
    train_labels = []
    for image in images:
        if image[-4:] == 'jpeg':
            transformed_image = transform_img(io.imread(DATA_DIR + '/' + image))
            train_data.append(transformed_image)
            label_file = image[:-5] + '.txt'
            with open(DATA_DIR + '/' + label_file) as f:
                content = f.readlines()
                label = int(float(content[0]))
                l = [0, 0]
                l[label] = 1
                train_labels.append(l)
    return np.array(train_data), np.array(train_labels)


#Loading model
model = load_model('model.h5')

#Classification
filename = 'testSky.jpeg'
testImg1 = transform_img(io.imread(filename))
prediction = model.predict(testImg1.reshape((1, 50, 50, 3)))

#Output image label
font = cv2.FONT_HERSHEY_SIMPLEX
testImg = cv2.imread(filename, cv2.IMREAD_COLOR)
width, height, chanels = testImg.shape

#Check if it's car or not and write it on image
if prediction[0][0] > 0.5:
    cv2.rectangle(testImg, (int(0.2 * height), int(0.7 * width)), (int(0.8 * height), int(0.85 * width)), (0, 0, 255),
                  cv2.FILLED)
    cv2.putText(testImg, 'NOT CAR', (int(0.45*height), int(0.8*width)), font, 0.5, (200, 255, 255), 2, cv2.LINE_AA)
else:
    cv2.rectangle(testImg, (int(0.2 * height), int(0.7 * width)), (int(0.8 * height), int(0.85 * width)), (255, 0, 0),
                  cv2.FILLED)
    cv2.putText(testImg, 'CAR', (int(0.47 * height), int(0.8 * width)), font, 0.5, (200, 255, 255), 2, cv2.LINE_AA)

#Show the image
cv2.imshow('image', testImg)
cv2.waitKey(0)
cv2.destroyAllWindows()