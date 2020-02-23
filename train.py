import pathlib
import imageio
import cv2

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import tensorflow as tf
import tensorflow_addons as tfa
import tensorflow.keras.backend as K
from tensorflow.keras.layers import \
    BatchNormalization, Conv2D, Activation, ReLU, \
    Input, AveragePooling2D, Dense, Flatten
from tensorflow.keras.regularizers import l2
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ReduceLROnPlateau, LearningRateScheduler
import tensorflow_hub as hub

from skimage.color import rgb2gray
from skimage.filters import threshold_otsu
from skimage.restoration import denoise_wavelet
from sklearn.model_selection import KFold
from sklearn.preprocessing import OneHotEncoder

SQUARE_LEFT = 570
SQUARE_RIGHT = 1400
SQUARE_TOP = 190
SQUARE_BOTTOM = 1020

IMG_HEIGHT = 224
IMG_WIDTH = 224

CLASS_NAMES = ['GOOD', 'BAD']

def crop_char_image_old(image, threshold=0.5):
    assert image.ndim == 2
    is_black = image > threshold

    is_black_vertical = np.sum(is_black, axis=0) > 50
    is_black_horizontal = np.sum(is_black, axis=1) > 50
    left = np.argmax(is_black_horizontal)
    right = np.argmax(is_black_horizontal[::-1])
    top = np.argmax(is_black_vertical)
    bottom = np.argmax(is_black_vertical[::-1])
    height, width = image.shape
    cropped_image = image[left:height - right, top:width - bottom]
    return cropped_image

def crop_char_image(image, mask, threshold=0.5):
    is_black = mask > threshold

    is_black_vertical = np.sum(is_black, axis=0) > 3
    is_black_horizontal = np.sum(is_black, axis=1) > 3
    left = np.argmax(is_black_horizontal)
    right = np.argmax(is_black_horizontal[::-1])
    top = np.argmax(is_black_vertical)
    bottom = np.argmax(is_black_vertical[::-1])
    width, height = mask.shape

    if height - (top + bottom) < IMG_HEIGHT:
        middle = top + (height - (bottom + top)) // 2
        top = max(middle - IMG_HEIGHT // 2, 0)
        bottom = height - min(middle + IMG_HEIGHT // 2, height)

    if width - (left + right) < IMG_WIDTH:
        middle = left + (width - (left + right)) // 2
        left = max(middle - IMG_WIDTH // 2, 0)
        right = width - min(middle + IMG_WIDTH // 2, width)

    cropped_image = image[left:width - right, top:height - bottom]
    return cropped_image

def resize(image, size=(IMG_HEIGHT, IMG_WIDTH)):
    return cv2.resize(image, size)


def analyze_image(im_path):
    '''
    Take an image_path (pathlib.Path object), preprocess it.
    '''
    # Read in data as RGB
    img = imageio.imread(str(im_path), as_gray=False, pilmode="RGB")

    # crop to the square
    img = img[SQUARE_TOP:SQUARE_BOTTOM, SQUARE_LEFT:SQUARE_RIGHT]

    #img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    #                            cv2.THRESH_BINARY,11,2)

    # denoise the image
    #img = denoise_wavelet(img, rescale_sigma=True)

    # thresholding
    # thresh = threshold_otsu(img)

    # find a crop mask
    img_mask = rgb2gray(img)
    img_mask = denoise_wavelet(img_mask, rescale_sigma=True)
#    img_mask = img_mask.astype(np.float32) / 255.
    #print(img_mask)

    thresh = 0.5
    img_mask = img_mask > thresh

    #print(img_mask)
    #crop background according to the mask
    img = crop_char_image(img, img_mask)

    #img = img.astype(np.float32)
    # opposite white and black
    #img = (1. - img).astype(np.float32)
    img = img / 255.
    #resize
    img = resize(img)
    return img


def analyze_list_of_images(im_path_list):
    all_df = []
    for im_path in im_path_list:
        im_df = analyze_image(im_path)
        all_df.append(im_df)
    return all_df


def show_batch(image_batch, label_batch):
    plt.figure(figsize=(10,10))
    for n in range(25):
        ax = plt.subplot(5,5,n+1)
        plt.imshow(image_batch[n])
        plt.title(CLASS_NAMES[label_batch[n]==1][0].title())
        plt.axis('off')


class CustomCallback(tf.keras.callbacks.Callback):
    def __init__(self, valid_data, batch_size=16):
        self.valid_inputs = valid_data[0]
        self.valid_outputs = valid_data[1]
        self.batch_size = batch_size

    def on_train_begin(self, logs={}):
        self.valid_predictions = []
        self.test_predictions = []

    def on_epoch_end(self, epoch, logs=None):
        score = model.evaluate(self.valid_inputs, self.valid_outputs, verbose=0)
        print(f'\nModel score: {score}')


def ResNet_pretrained_model(num_classes=2):
    m = tf.keras.Sequential([
        hub.KerasLayer("https://tfhub.dev/google/imagenet/resnet_v2_50/feature_vector/4",
                       trainable=False),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])
    m.build([None, IMG_HEIGHT, IMG_WIDTH, 3])
    return m


def ResNet_model(input_shape, depth):

    def resnet_layer(inputs,
                     num_filters=16,
                     kernel_size=3,
                     strides=1,
                     activation='relu',
                     batch_normalization=True,
                     conv_first=True):
        """2D Convolution-Batch Normalization-Activation stack builder

        # Arguments
            inputs (tensor): input tensor from input image or previous layer
            num_filters (int): Conv2D number of filters
            kernel_size (int): Conv2D square kernel dimensions
            strides (int): Conv2D square stride dimensions
            activation (string): activation name
            batch_normalization (bool): whether to include batch normalization
            conv_first (bool): conv-bn-activation (True) or
                bn-activation-conv (False)

        # Returns
            x (tensor): tensor as input to the next layer
        """
        conv = Conv2D(num_filters,
                      kernel_size=kernel_size,
                      strides=strides,
                      padding='same',
                      kernel_initializer='he_normal',
                      kernel_regularizer=l2(1e-4))

        x = inputs
        if conv_first:
            x = conv(x)
            if batch_normalization:
                x = BatchNormalization()(x)
            if activation is not None:
                x = Activation(activation)(x)
        else:
            if batch_normalization:
                x = BatchNormalization()(x)
            if activation is not None:
                x = Activation(activation)(x)
            x = conv(x)
        return x

    def resnet_v1(input_shape, depth, num_classes=2):
        """ResNet Version 1 Model builder [a]

        Stacks of 2 x (3 x 3) Conv2D-BN-ReLU
        Last ReLU is after the shortcut connection.
        At the beginning of each stage, the feature map size is halved (downsampled)
        by a convolutional layer with strides=2, while the number of filters is
        doubled. Within each stage, the layers have the same number filters and the
        same number of filters.
        Features maps sizes:
        stage 0: 32x32, 16
        stage 1: 16x16, 32
        stage 2:  8x8,  64
        The Number of parameters is approx the same as Table 6 of [a]:
        ResNet20 0.27M
        ResNet32 0.46M
        ResNet44 0.66M
        ResNet56 0.85M
        ResNet110 1.7M

        # Arguments
            input_shape (tensor): shape of input image tensor
            depth (int): number of core convolutional layers
            num_classes (int): number of classes (CIFAR10 has 10)

        # Returns
            model (Model): Keras model instance
        """
        if (depth - 2) % 6 != 0:
            raise ValueError('depth should be 6n+2 (eg 20, 32, 44 in [a])')
        # Start model definition.
        num_filters = 16
        num_res_blocks = int((depth - 2) / 6)

        inputs = Input(shape=input_shape)
        x = resnet_layer(inputs=inputs)
        # Instantiate the stack of residual units
        for stack in range(3):
            for res_block in range(num_res_blocks):
                strides = 1
                if stack > 0 and res_block == 0:  # first layer but not first stack
                    strides = 2  # downsample
                y = resnet_layer(inputs=x,
                                 num_filters=num_filters,
                                 strides=strides)
                y = resnet_layer(inputs=y,
                                 num_filters=num_filters,
                                 activation=None)
                if stack > 0 and res_block == 0:  # first layer but not first stack
                    # linear projection residual shortcut connection to match
                    # changed dims
                    x = resnet_layer(inputs=x,
                                     num_filters=num_filters,
                                     kernel_size=1,
                                     strides=strides,
                                     activation=None,
                                     batch_normalization=False)
                x = tf.keras.layers.add([x, y])
                x = Activation('relu')(x)
            num_filters *= 2

        # Add classifier on top.
        # v1 does not use BN after last shortcut connection-ReLU
        x = AveragePooling2D(pool_size=8)(x)
        y = Flatten()(x)
        outputs = Dense(num_classes,
                        activation='softmax',
                        kernel_initializer='he_normal')(y)

        # Instantiate model.
        model = Model(inputs=inputs, outputs=outputs)
        return model

    return resnet_v1(input_shape=input_shape, depth=depth)

def lr_schedule(epoch):
    """Learning Rate Schedule

    Learning rate is scheduled to be reduced after 80, 120, 160, 180 epochs.
    Called automatically every epoch as part of callbacks during training.

    # Arguments
        epoch (int): The number of epochs

    # Returns
        lr (float32): learning rate
    """
    lr = 1e-3
    if epoch > 180:
        lr *= 0.5e-3
    elif epoch > 160:
        lr *= 1e-3
    elif epoch > 120:
        lr *= 1e-2
    elif epoch > 80:
        lr *= 1e-1
    print('Learning rate: ', lr)
    return lr


def simple_cnn_model(input_shape):
    input_images = tf.keras.layers.Input(
        input_shape, dtype=tf.float32, name='input_image')
    x = tf.keras.layers.MaxPool2D(strides=2)(input_images)
    x = tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3))(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)

    x = tf.keras.layers.MaxPool2D(strides=2)(x)
    x = tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3))(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)

    x = tf.keras.layers.MaxPool2D(strides=2)(x)
    x = tf.keras.layers.Conv2D(filters=128, kernel_size=(3, 3))(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)

    x = tf.keras.layers.MaxPool2D(strides=2)(x)
    x = tf.keras.layers.Dropout(0.25)(x)
    x = tf.keras.layers.Flatten()(x)

    x = tf.keras.layers.Dense(128, activation='tanh')(x)
    x = tf.keras.layers.Dropout(0.5)(x)

    out = tf.keras.layers.Dense(1, activation='tanh', name='dense_output')(x)

    model = tf.keras.models.Model(inputs=[input_images], outputs=out)
    return model


def train_and_eval(model, train_data, valid_data,
                   initial_learning_rate, epochs, batch_size, loss_function):
    datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        # set input mean to 0 over the dataset
        featurewise_center=False,
        # set each sample mean to 0
        samplewise_center=False,
        # divide inputs by std of dataset
        featurewise_std_normalization=False,
        # divide each input by its std
        samplewise_std_normalization=False,
        # apply ZCA whitening
        zca_whitening=False,
        # epsilon for ZCA whitening
        zca_epsilon=1e-06,
        # randomly rotate images in the range (deg 0 to 180)
        rotation_range=90,
        # randomly shift images horizontally
        width_shift_range=0.4,
        # randomly shift images vertically
        height_shift_range=0.4,
        # set range for random shear
        shear_range=10.,
        # set range for random zoom
        zoom_range=4.,
        # set range for random channel shifts
        channel_shift_range=0.,
        # set mode for filling points outside the input boundaries
        fill_mode='nearest',
        # value used for fill_mode = "constant"
        cval=0.,
        # randomly flip images
        horizontal_flip=True,
        # randomly flip images
        vertical_flip=True,
        # set rescaling factor (applied before any other transformation)
        rescale=None,
        # set function that will be applied on each input
        preprocessing_function=None,
        # image data format, either "channels_first" or "channels_last"
        data_format="channels_first",
        # fraction of images reserved for validation (strictly between 0 and 1)
        validation_split=0.0)

    datagen.fit(train_data[0])
    custom_callback = CustomCallback(
        valid_data=(valid_data[0], valid_data[1]), batch_size=batch_size)
    lr_scheduler = LearningRateScheduler(lr_schedule)
    lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1),
                                   cooldown=0,
                                   patience=5,
                                   min_lr=0.5e-6)
    callbacks = [lr_reducer, lr_scheduler, custom_callback]
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule(0))
    model.compile(loss=loss_function, optimizer=optimizer, metrics=['accuracy'])

    # fits the model on batches with real-time data augmentation:
    model.fit_generator(datagen.flow(train_data[0], train_data[1], batch_size=batch_size),
                        steps_per_epoch=len(train_data[0]) / batch_size, epochs=epochs,
                        callbacks=callbacks)
    del optimizer, datagen
    return

train_good_paths = pathlib.Path('./data/nailgun/nailgun/').glob('./good/*.jpeg')
train_bad_paths = pathlib.Path('./data/nailgun/nailgun/').glob('./bad/*.jpeg')
# print(list(train_good))

train_good = analyze_list_of_images(train_good_paths)
train_bad = analyze_list_of_images(train_bad_paths)
train_total = np.array(train_good + train_bad)
train_total = train_total.astype('float32')
# train_total = np.expand_dims(train_total, axis=-1)
train_output = np.array([1] * len(train_good) + [0] * len(train_bad))

# shuffle data
shuffle_idx = list(range(len(train_total)))
np.random.shuffle(shuffle_idx)

# split into train and valid
train_idx = shuffle_idx[:178]
valid_idx = shuffle_idx[178:]

train_inputs = train_total[train_idx]
train_labels = train_output[train_idx]

valid_inputs = train_total[valid_idx]
valid_labels = train_output[valid_idx]

# labels to one-hot
onehotencoder = OneHotEncoder()
train_labels = onehotencoder.fit_transform(
    np.expand_dims(train_labels, -1)).toarray()
valid_labels = onehotencoder.fit_transform(
    np.expand_dims(valid_labels, -1)).toarray()

# show valid imgs
f = plt.figure()
columns = 4
rows = 5
for i, img in enumerate(valid_inputs):
    a = f.add_subplot(rows, columns, i + 1)
    a.set_title(valid_labels[i])
    plt.imshow(img)

plt.show()

#shuffle_idx = np.random.shuffle(list(range(len(train_total))))
#train_total = train_total[shuffle_idx]
#train_output = train_output[shuffle_idx]
#print(f'Data size: {len(train_total)}')

#print(train_output)

#cv = KFold(n_splits=20, random_state=42, shuffle=False)

n = 3
depth = n * 6 + 2

# model = ResNet_model(input_shape=train_total.shape[1:], depth=depth)

# train_and_eval(
#     model,
#     train_data=(train_total, train_output),
#     #valid_data=(valid_inputs, valid_labels),
#     initial_learning_rate=2e-3, epochs=100, batch_size=32,
#     loss_function=tf.keras.losses.SparseCategoricalCrossentropy())

#for fold, (train_idx, valid_idx) in enumerate(cv.split(train_output, groups=train_output)):
#    if fold > 0:
#        break
K.clear_session()
#model = simple_cnn_model(input_shape=train_total.shape[1:])
#model = ResNet_model(input_shape=train_total.shape[1:], depth=depth)
model = ResNet_pretrained_model()

#np.random.shuffle(train_idx)
#np.random.shuffle(valid_idx)
#train_inputs = train_total[train_idx]
#train_labels = train_output[train_idx]

#valid_inputs = train_total[valid_idx]
#valid_labels = train_output[valid_idx]
print(len(train_inputs), len(valid_inputs))

train_and_eval(
    model,
    train_data=(train_inputs, train_labels),
    valid_data=(valid_inputs, valid_labels),
    initial_learning_rate=2e-3, epochs=10, batch_size=8,
    loss_function='MSE')

    #histories.append(history)

    #print(np.sum(labels == 1), np.sum(labels == 0))

    #print(labels)

    #show_batch(train_inputs, labels)

    #break
