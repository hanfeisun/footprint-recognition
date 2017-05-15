import argparse
import os
from keras.callbacks import ModelCheckpoint
from keras.callbacks import TensorBoard
from keras.models import model_from_json, load_model

parser = argparse.ArgumentParser(description='Process some integers.')

parser.add_argument('-p', '--prefix', type=str, default="example", help='prefix for output!')
parser.add_argument('-t', '--train', type=str, help='train data')
parser.add_argument('-v', '--valid', type=str, help='validation data')
parser.add_argument('-l', '--log', type=str, help='log directory')
parser.add_argument('-w', '--weight', type=str, help='resume')

args = parser.parse_args()
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K
#from keras.utils import plot_model

img_width, img_height = 300, 100

train_data_dir = args.train
log_dir = args.log

validation_data_dir = args.valid
nb_train_samples = 8888 * 3
nb_validation_samples = 1112
epochs = 50000
batch_size = 256

prefix = args.prefix
prefix_dir = os.path.dirname(prefix)
if prefix_dir and not os.path.exists(os.path.dirname(prefix)):
    os.makedirs(os.path.dirname(prefix))

# if not os.path.exists(log_dir):
#     os.makedirs(log_dir)

if K.image_data_format() == 'channels_first':
    input_shape = (3, img_width, img_height)
    print("there")
else:
    input_shape = (img_height, img_width, 1)
    print("here")

model = Sequential()
model.add(Conv2D(16, (7, 7), input_shape=input_shape, activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=2))

model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=2))

model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=2))

model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=2))

# model.add(Conv2D(32, (3, 3), activation='relu'))
# model.add(ZeroPadding2D())
# model.add(MaxPooling2D(pool_size=(2, 2), strides=2))

# model.add(Conv2D(64, (3, 3), activation='relu'))
# model.add(ZeroPadding2D())
# model.add(MaxPooling2D(pool_size=(2, 2), strides=2))

# model.add(Conv2D(64, (3, 3), activation='relu'))
# model.add(ZeroPadding2D())
# model.add(MaxPooling2D(pool_size=(2, 2), strides=2))


# model.add(Conv2D(256, (3, 3), activation='relu'))
# model.add(ZeroPadding2D())
# model.add(MaxPooling2D(pool_size=(2, 2), strides=2))

# model.add(Conv2D(256, (3, 3), activation='relu'))
# model.add(ZeroPadding2D())
# model.add(MaxPooling2D(pool_size=(2, 2), strides=2))

model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1))
model.add(Activation('sigmoid'))

# model.add(Conv2D(32, (3, 3), activation='relu'))
# model.add(Activation('relu'))
# model.add(MaxPooling2D(pool_size=(2, 2), strides=2))
#
# model.add(Conv2D(64, (3, 3)))
# model.add(Activation('relu'))
# model.add(MaxPooling2D(pool_size=(2, 2), strides=1))

# model.add(Flatten())
# model.add(Dense(64, activation='relu'))
# model.add(Dropout(0.5))
# model.add(Dense(1))
# model.add(Activation('sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])
if args.weight:
    model.load_weights(args.weight)
print(model.summary())
#plot_model(model, to_file='model.png', show_shapes=True)

train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    #rotation_range=90.0,
    fill_mode='constant',
    cval=255
)

# this is the augmentation configuration we will use for testing:
# only rescaling
test_datagen = ImageDataGenerator(rescale=1. / 255,    
                                  rotation_range=0,
                                  fill_mode='constant',
                                  cval=255
)


train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    color_mode="grayscale",
    class_mode='binary',
)

validation_generator = test_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    color_mode="grayscale",
    class_mode='binary')

# validation_generator = train_datagen.flow_from_directory(
#     train_data_dir,
#     target_size=(img_height, img_width),
#     batch_size=batch_size,
#     color_mode="grayscale",
#     class_mode='binary',
#     # save_to_dir="valid_generated",
# )
filepath = "%s-{epoch:02d}-{val_acc:.2f}.hdf5" % prefix

checkpoint = ModelCheckpoint(filepath=filepath, monitor='val_acc', mode='max')
tensor_board = TensorBoard(log_dir=log_dir, histogram_freq=5, write_graph=True, write_images=True)

model.fit_generator(
    train_generator,
    steps_per_epoch=nb_train_samples // batch_size,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=nb_validation_samples // batch_size, callbacks=[checkpoint, tensor_board])

model_json = model.to_json()

json_path = prefix + ".json"
h5_path = prefix
with open(prefix + ".json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("%s.h5" % args.prefix)
print("Saved model to disk")
  
