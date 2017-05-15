import argparse
import os
from keras.callbacks import ModelCheckpoint
from keras.callbacks import TensorBoard
import numpy as np
import skimage.filters

parser = argparse.ArgumentParser(description='Process some integers.')

parser.add_argument('-p', '--prefix', type=str, default="example", help='prefix for output!')
parser.add_argument('-t', '--train', type=str, help='train data')
parser.add_argument('-l', '--log', type=str, help='log directory')

args = parser.parse_args()
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K
from keras.utils import plot_model

img_width, img_height = 100, 300

train_data_dir = args.train
log_dir = args.log
nb_train_samples = 9778
nb_validation_samples = 222
epochs = 50000
batch_size = 100

prefix = args.prefix
prefix_dir = os.path.dirname(prefix)
if prefix_dir and not os.path.exists(os.path.dirname(prefix)):
    os.makedirs(os.path.dirname(prefix))

if not os.path.exists(log_dir):
    os.makedirs(log_dir)

if K.image_data_format() == 'channels_first':
    input_shape = (3, img_width, img_height)
    print("there")
else:
    input_shape = (img_height, img_width, 1)
    print("here")

model = Sequential()
model.add(Conv2D(16, (3, 3), input_shape=input_shape))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=2))

model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=2))

model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=2))

model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=2))

model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors
model.add(Dense(32, activation='relu'))
model.add(Dense(1000, activation='softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

print(model.summary())
plot_model(model, to_file='model.png', show_shapes=True)



train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    rotation_range=90,
    zoom_range=0.2,
    fill_mode='constant',
    cval=255,
    horizontal_flip=True,
    vertical_flip=True,
    # preprocessing_function=preprocess
)

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    color_mode="grayscale",
    class_mode='categorical',
    #save_to_dir="data/train_generated",
)

filepath = "%s-{epoch:02d}-{acc:.2f}.hdf5" % prefix

checkpoint = ModelCheckpoint(filepath=filepath, save_best_only=True, monitor='acc', mode='max')
tensor_board = TensorBoard(log_dir=log_dir, histogram_freq=5, write_graph=True, write_images=True)

model.fit_generator(
    train_generator,
    steps_per_epoch=nb_train_samples // batch_size,
    epochs=epochs,
    callbacks=[checkpoint, tensor_board]
)

model_json = model.to_json()

json_path = prefix + ".json"
h5_path = prefix
with open(prefix + ".json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("%s.h5" % args.prefix)
print("Saved model to disk")
