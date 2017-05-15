import numpy as np
import skimage
import skimage.transform
import skimage.feature
import skimage.io
import os
from os.path import join
import cv2
left_dir = "test/left"
left_items = sorted([x for x in os.listdir(left_dir) if x.endswith(".png")], key=lambda x: int(x.split(".")[0]))
right_dir = "test/right"
right_items = sorted([x for x in os.listdir(right_dir) if x.endswith(".png")], key=lambda x: int(x.split(".")[0]))



for name in left_items:
    print(name)
    image = cv2.imread(join(left_dir, name), 0)
    #trans = np.where(skimage.filters.gaussian(image, 0.8) > 0.6, 255, 0)
    trans = cv2.flip(image, 0)
    skimage.io.imsave("test/right/0_" + name, trans)
    trans = cv2.flip(image, 1)
    skimage.io.imsave("test/right/1_" + name, trans)


for name in right_items:
    print(name)
    image = cv2.imread(join(right_dir, name), 0)
    #trans = np.where(skimage.filters.gaussian(image, 0.8) > 0.6, 255, 0)
    trans = cv2.flip(image, 0)
    skimage.io.imsave("test/left/0_" + name, trans)
    trans = cv2.flip(image, 1)
    skimage.io.imsave("test/left/1_" + name, trans)
    #image = skimage.io.imread(join(train_dir, name), as_grey=True)
    #trans = np.where(skimage.filters.gaussian(image, 0.8) > 0.6, 255, 0)
    #skimage.io.imsave("copy/left/" + name, trans)


# train_dir = "data/valid/test/"
# items = sorted([x for x in os.listdir(train_dir) if x.endswith(".png")], key=lambda x: int(x.split(".")[0]))
# for name in items:
#     image = skimage.io.imread(join(train_dir, name), as_grey=True)
#     trans = np.where(skimage.filters.gaussian(image, 0.8) > 0.6, 255, 0)
#     skimage.io.imsave("data/validprocess/test/" + name, trans)
