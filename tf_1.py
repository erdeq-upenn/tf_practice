# -*- coding: utf-8 -*-
"""
Created on Sat May 28 13:58:19 2022

@author: dequan

# MIT License
#
# Copyright (c) 2017 François Chollet
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

"""
# -------------------------------------
# image Fashion MNIST
# import dataset 
import tensorflow as tf 
import matplotlib.pyplot as plt

fashion_mnish = tf.keras.datasets.fashion_mnist
(train_img,train_labels), (test_img,test_labels) = fashion_mnish.load_data()
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
# -------------------------------------
# EDA 

# plt.figure()
# plt.imshow(train_img[0])
# plt.colorbar()
# plt.grid(False)
# plt.show()

# # normalize figure values 
# train_img = train_img / 255.0
# test_img = test_img / 255.0
# # plot each fiture type of 5X 5 matrix 
# plt.figure(figsize=(10,10))
# for i in range(25):
#     plt.subplot(5,5,i+1)
#     plt.xticks([])
#     plt.yticks([])
#     plt.grid(False)
#     plt.imshow(train_img[i], cmap=plt.cm.binary)
#     plt.xlabel(class_names[train_labels[i]])
# plt.show()

# -------------------------------------
# build the model 
# 1. The first layer in this network, 
# tf.keras.layers.Flatten, transforms the format of the images from a 
# two-dimensional array (of 28 by 28 pixels) to a one-dimensional 
# array (of 28 * 28 = 784 pixels). 
# Think of this layer as unstacking rows of pixels in the image 
# and lining them up. This layer has no parameters to learn;
# it only reformats the data.

model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10)
])

# -------------------------------------
# compile the model
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

# -------------------------------------
# Train the model 
model.fit(train_img,train_labels,epochs=10)

# -------------------------------------
# Evaluate accuracy 
test_loss, test_acc = model.evaluate(test_img,  test_labels, verbose=2)
print('\nTest accuracy:', test_acc)

# This is a overfitting issue: 
# https://www.tensorflow.org/tutorials/keras/overfit_and_underfit#demonstrate_overfitting
# Always keep this in mind: deep learning models tend to be good at fitting to 
# the training data, but the real challenge is generalization, not fitting.

probability_model = tf.keras.Sequential([model, 
                                         tf.keras.layers.Softmax()])
predictions = probability_model.predict(test_img)

# -------------------------------------
# plot and validate the predictions 
def plot_image(i, predictions_array, true_label, img):
  true_label, img = true_label[i], img[i]
  plt.grid(False)
  plt.xticks([])
  plt.yticks([])

  plt.imshow(img, cmap=plt.cm.binary)

  predicted_label = np.argmax(predictions_array)
  if predicted_label == true_label:
    color = 'blue'
  else:
    color = 'red'

  plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                100*np.max(predictions_array),
                                class_names[true_label]),
                                color=color)

def plot_value_array(i, predictions_array, true_label):
  true_label = true_label[i]
  plt.grid(False)
  plt.xticks(range(10))
  plt.yticks([])
  thisplot = plt.bar(range(10), predictions_array, color="#777777")
  plt.ylim([0, 1])
  predicted_label = np.argmax(predictions_array)

  thisplot[predicted_label].set_color('red')
  thisplot[true_label].set_color('blue')
  
i = 12
plt.figure(figsize=(6,3))
plt.subplot(1,2,1)
plot_image(i, predictions[i], test_labels, test_img)
plt.subplot(1,2,2)
plot_value_array(i, predictions[i],  test_labels)
plt.show()

# Plot the first X test images, their predicted labels, and the true labels.
# Color correct predictions in blue and incorrect predictions in red.
num_rows = 5
num_cols = 3
num_images = num_rows*num_cols
plt.figure(figsize=(2*2*num_cols, 2*num_rows))
for i in range(num_images):
  plt.subplot(num_rows, 2*num_cols, 2*i+1)
  plot_image(i, predictions[i], test_labels, test_img)
  plt.subplot(num_rows, 2*num_cols, 2*i+2)
  plot_value_array(i, predictions[i], test_labels)
plt.tight_layout()
plt.show()

# -------------------------------------
# get the trained model 
img = test_img[1]
label_true = test_labels[1]
print(img.shape)
# change the dimention to ndim =3 
img = (np.expand_dims(img,0))
print(img.shape)

predictions_single = probability_model.predict(img)


