#Importing the Keras libraries and packages
from keras.models import Sequential #use: initialize CNN
from keras.layers import Convolution2D #use: make convolution layers - 2D means we're dealing with images
from keras.layers import MaxPooling2D #use: add pooling layers
from keras.layers import Flatten #use to make large feature vector
from keras.layers import Dense #use: create fully connected layers in ANN
from keras.preprocessing.image import ImageDataGenerator


#Initializing the CNN
classifier = Sequential()

#Convolution
classifier.add(Convolution2D(32, (3, 3), input_shape=(64, 64, 3), activation='relu'))

#Pooling
classifier.add(MaxPooling2D(pool_size=(2, 2)))

#Flattening
classifier.add(Flatten())

#Full connection
classifier.add(Dense(128, activation='relu'))
classifier.add(Dense(4, activation='softmax'))


classifier.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory(
        'data/train',
        target_size=(64, 64),
        batch_size=32,
        class_mode='categorical')

test_set = test_datagen.flow_from_directory(
        'data/test',
        target_size=(64, 64),
        batch_size=32,
        class_mode='categorical')

classifier.fit_generator(
        training_set,
        steps_per_epoch=20,
        epochs=25,
        validation_data=test_set,
        validation_steps=15)