import os

import keras.models
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import argparse

from scipy import interp
from itertools import cycle
from sklearn import svm, datasets
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
from tensorflow.keras.preprocessing import image
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator

parser = argparse.ArgumentParser(description='files')
parser.add_argument("srs", nargs='+', type=str, )

args = parser.parse_args()

train_duck_dir = (args.srs[0] + '/duck')

train_human_dir = (args.srs[0] + '/human')

valid_duck_dir = (args.srs[1] + '/duck')

valid_human_dir = (args.srs[1] + '/human')

train_duck_names = os.listdir(train_duck_dir)

train_human_names = os.listdir(train_human_dir)

validation_human_names = os.listdir(valid_human_dir)

print('total training duck images:', len(os.listdir(train_duck_dir)))
print('total training human images:', len(os.listdir(train_human_dir)))
print('total validation duck images:', len(os.listdir(valid_duck_dir)))
print('total validation human images:', len(os.listdir(valid_human_dir)))

uploaded = []
for root, dirs, files in os.walk(args.srs[2]):
    for filename in files:
        uploaded.append(filename)
#print(uploaded)

nrows = 4
ncols = 4

pic_index = 0

fig = plt.gcf()
fig.set_size_inches(ncols * 4, nrows * 4)

pic_index += 8
next_duck_pic = [os.path.join(train_duck_dir, fname)
                 for fname in train_duck_names[pic_index - 8:pic_index]]
next_human_pic = [os.path.join(train_human_dir, fname)
                  for fname in train_human_names[pic_index - 8:pic_index]]

for i, img_path in enumerate(next_duck_pic + next_human_pic):
    sp = plt.subplot(nrows, ncols, i + 1)
    sp.axis('Off')

    img = mpimg.imread(img_path)
    plt.imshow(img)

plt.show()

train_datagen = ImageDataGenerator(rescale=1 / 255,
                                   rotation_range=45,
                                   horizontal_flip=True,
                                   )
validation_datagen = ImageDataGenerator(rescale=1 / 255)

train_generator = train_datagen.flow_from_directory(
    args.srs[0],
    classes=['duck', 'human'],
    target_size=(200, 200),
    batch_size=170,
    class_mode='binary'
)

validation_generator = validation_datagen.flow_from_directory(
    args.srs[1],
    classes=['duck', 'human'],
    target_size=(200, 200),
    batch_size=40,
    class_mode='binary',
    shuffle=False)

# model = tf.keras.models.Sequential([  # tf.keras.layers.Flatten(input_shape = (200,200,3)),
#       tf.keras.layers.Conv2D(32, 4, activation='relu', input_shape=(200, 200, 3)),
#       #tf.keras.layers.BatchNormalization(),
#       tf.keras.layers.MaxPool2D((4, 4)),
#       tf.keras.layers.Conv2D(64, 4, activation='relu'),
#      # tf.keras.layers.BatchNormalization(),
#       tf.keras.layers.MaxPool2D((4, 4)),
#       tf.keras.layers.Flatten(),
#       tf.keras.layers.Dropout(0.6),
#       tf.keras.layers.Dense(256, activation=tf.nn.relu),
#       tf.keras.layers.Dropout(0.4),
#       tf.keras.layers.Dense(64, activation=tf.nn.relu),
#       tf.keras.layers.Dropout(0.2),
#       tf.keras.layers.Dense(1, activation=tf.nn.sigmoid)])
#
# model.summary()
#
# model.compile(optimizer=tf.keras.optimizers.Adam(),
#                 loss='binary_crossentropy',
#                 metrics=['accuracy'])
model = keras.models.load_model('saved/')


# history = model.fit(train_generator,
#                      steps_per_epoch=None,
#                      epochs=8,
#                      verbose=1,
#                      validation_data=validation_generator)
#model.save('saved/')
for fn in uploaded:
    path = os.path.join(args.srs[2], fn)
    img = image.load_img(path, target_size=(200, 200))
    x = image.img_to_array(img)
    #plt.imshow(x / 255.)
    x = np.expand_dims(x, axis=0)
    images = np.vstack([x])
    classes = model.predict(images)
    print(classes[0])
    if classes[0] < 0.5:
        print(fn + " is a duck")
    else:
        print(fn + " is a human")
