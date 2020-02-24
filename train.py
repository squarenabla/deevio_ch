import random
import matplotlib.pyplot as plt

from sklearn.preprocessing import OneHotEncoder

from image_preprocessing import *
from models import *

MODEL_PATH = './model/1'
DATA_PATH = './data/nailgun/nailgun/'

IMG_HEIGHT = 224
IMG_WIDTH = 224

CLASS_NAMES = ['GOOD', 'BAD']

SEED = 42

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)


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


def train_and_eval(model, train_data, valid_data,
                   epochs, batch_size, loss_function):
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


seed_everything(SEED)

train_good_paths = pathlib.Path(DATA_PATH).glob('./good/*.jpeg')
train_bad_paths = pathlib.Path(DATA_PATH).glob('./bad/*.jpeg')

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

K.clear_session()
#model = simple_cnn_model(input_shape=train_total.shape[1:])
#model = ResNet_model(input_shape=train_total.shape[1:], depth=depth)
model = ResNet_pretrained_model(input_shape=[None, IMG_HEIGHT, IMG_WIDTH, 3])

print(len(train_inputs), len(valid_inputs))

train_and_eval(
    model,
    train_data=(train_inputs, train_labels),
    valid_data=(valid_inputs, valid_labels),
    epochs=10, batch_size=8, loss_function='MSE')

model.save(MODEL_PATH)