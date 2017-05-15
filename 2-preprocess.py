import numpy as np
import skimage
import skimage.transform
import skimage.feature
import skimage.io
import os
from os.path import join

train_dir = "copy"

items = sorted([x for x in os.listdir(train_dir) if x.endswith(".png")], key=lambda x: int(x.split(".")[0]))
for name in items:
    image = skimage.io.imread(join(train_dir, name), as_grey=True)
    trans = np.where(skimage.filters.gaussian(image, 0.8) > 0.6, 255, 0)
    skimage.io.imsave("copyprocessed/process" + name, trans)

train_dir = "valid"
items = sorted([x for x in os.listdir(train_dir) if x.endswith(".png")], key=lambda x: int(x.split(".")[0]))
for name in items:
    image = skimage.io.imread(join(train_dir, name), as_grey=True)
    trans = np.where(skimage.filters.gaussian(image, 0.8) > 0.6, 255, 0)
    skimage.io.imsave("validprocessed/process" + name, trans)
