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

def ResNet_pretrained_model(input_shape, num_classes=2):
    m = tf.keras.Sequential([
        hub.KerasLayer("https://tfhub.dev/google/imagenet/resnet_v2_50/feature_vector/4",
                       trainable=False),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])
    m.build(input_shape)
    return m

# from https://keras.io/examples/cifar10_resnet/
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