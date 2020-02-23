import pathlib
import imageio
import cv2
import os
import validators
import numpy as np

from skimage.color import rgb2gray
from skimage.filters import threshold_otsu
from skimage.restoration import denoise_wavelet

IMG_HEIGHT = 224
IMG_WIDTH = 224

SQUARE_LEFT = 570
SQUARE_RIGHT = 1400
SQUARE_TOP = 190
SQUARE_BOTTOM = 1020

CLASS_NAMES = ['BAD', 'GOOD']


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
    if not validators.url(str(im_path)) and not os.path.exists(str(im_path)):
        return None

    img = imageio.imread(str(im_path), as_gray=False, pilmode="RGB")

    # crop to the square
    img = img[SQUARE_TOP:SQUARE_BOTTOM, SQUARE_LEFT:SQUARE_RIGHT]

    # denoise the image
    # img = denoise_wavelet(img, rescale_sigma=True)

    # thresholding
    # thresh = threshold_otsu(img)

    # find a crop mask
    img_mask = rgb2gray(img)
    img_mask = denoise_wavelet(img_mask)

    thresh = 0.5
    img_mask = img_mask > thresh

    # crop background according to the mask
    img = crop_char_image(img, img_mask)

    # opposite white and black
    # img = (1. - img).astype(np.float32)

    img = img / 255.

    # resize
    img = resize(img)
    return img


def analyze_list_of_images(im_path_list):
    all_df = []
    for im_path in im_path_list:
        im_df = analyze_image(im_path)
        all_df.append(im_df)
    return all_df
