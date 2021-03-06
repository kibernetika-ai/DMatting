import tensorflow as tf
import numpy as np
import random
from scipy import ndimage
import glob
import os
import cv2 as cv
import math
from PIL import Image

trimap_kernel = [val for val in range(20, 40)]
g_mean = np.array(([126.88, 120.24, 112.19])).reshape([1, 1, 3])

unknown_code = 128


def generate_trimap(alpha):
    trimap = np.copy(alpha)
    k_size = random.choice(trimap_kernel)
    #trimap[np.less(alpha,255)]=unknown_code
    trimap[np.less(alpha,255)]=0
    trimap[np.where((ndimage.grey_dilation(alpha[:, :], size=(k_size, k_size)) - ndimage.grey_erosion(alpha[:, :],
                                                                                                      size=(k_size,
                                                                                                            k_size))) != 0)] = unknown_code
    return trimap


def norm_background(background, fh, fw):
    bh = background.shape[0]
    bw = background.shape[1]
    wratio = fw / bw
    hratio = fh / bh
    ratio = wratio if wratio > hratio else hratio
    if ratio > 1:
        bh = math.ceil(bw * ratio)
        bw = math.ceil(bh * ratio)
        background = cv.resize(background, (bh, bw), interpolation=cv.INTER_CUBIC)
    x = 0
    if bw > fw:
        x = np.random.randint(0, bw - fw)
    y = 0
    if bh > fh:
        y = np.random.randint(0, bh - fh)
    return background[y:y + fh, x:x + fw]


def random_crop(img):
    x = max(0, img.shape[1] - 320)
    y = max(0, img.shape[0] - 320)
    if x > 0:
        x = np.random.randint(0, x)
    if y > 0:
        y = np.random.randint(0, y)
    img = img[y:min(y + 320, img.shape[0]), x:min(x + 320, img.shape[1])]

    if img.shape != (320, 320):
        ret = cv.resize(img, dsize=(320, 320), interpolation=cv.INTER_NEAREST)
    return ret


def safe_crop(mat, x, y, crop_size=(320, 320)):
    crop_height, crop_width = crop_size
    if crop_height == 0:
        return cv.resize(mat, dsize=(320, 320), interpolation=cv.INTER_NEAREST)
    if len(mat.shape) == 2:
        ret = np.zeros((crop_height, crop_width), np.float32)
    else:
        ret = np.zeros((crop_height, crop_width, 3), np.float32)
    crop = mat[y:y + crop_height, x:x + crop_width]
    h, w = crop.shape[:2]
    ret[0:h, 0:w] = crop
    if crop_size != (320, 320):
        ret = cv.resize(ret, dsize=(320, 320), interpolation=cv.INTER_NEAREST)
    return ret


def random_choice(trimap, crop_size=(320, 320)):
    crop_height, crop_width = crop_size
    if crop_height == 0:
        return 0, 0
    y_indices, x_indices = np.where(trimap == unknown_code)
    num_unknowns = len(y_indices)
    x, y = 0, 0
    if num_unknowns > 0:
        ix = np.random.choice(range(num_unknowns))
        center_x = x_indices[ix]
        center_y = y_indices[ix]
        x = max(0, center_x - int(crop_width / 2))
        y = max(0, center_y - int(crop_height / 2))
    return x, y


def matting_input_fn(params):
    dataset = params['data_set']
    alpha_path = dataset + '/alpha'
    fg_path = dataset + '/foreground'
    backgrounds = params['backgrounds']
    backgrounds = glob.glob(backgrounds, recursive=True)
    background_count = params['background_count']
    different_sizes = [(0,0),(320, 320), (480, 480), (640, 640)]

    def _input_fn():
        def _generator():
            for aname in glob.iglob(alpha_path + '/*.png'):
                fname = os.path.basename(aname)
                fname = os.path.join(fg_path, fname)
                if not os.path.exists(fname):
                    continue
                alpha = Image.open(aname)
                alpha = np.array(alpha)
                if len(alpha.shape)>2:
                    alpha = alpha[:,:,0]
                #alpha[np.greater(alpha, 0)] = 255
                original_foreground = np.array(Image.open(fname).convert('RGB'))
                background = random.choice(backgrounds)
                background = np.array(Image.open(background).convert('RGB'))
                crop_size = random.choice(different_sizes)
                trimap = generate_trimap(alpha)
                x, y = random_choice(trimap, crop_size)
                background = random_crop(background)
                foreground = safe_crop(original_foreground, x, y, crop_size)
                train_alpha = safe_crop(alpha, x, y, crop_size)
                #trimap = safe_crop(trimap, x, y, crop_size)
                trimap = generate_trimap(train_alpha)
                train_alpha = np.expand_dims(train_alpha, 2).astype(np.float32) / 255
                trimap = np.expand_dims(trimap, 2).astype(np.float32)
                if np.sum(np.equal(trimap,unknown_code).astype(np.int32)) < 1:
                    continue
                background = background.astype(np.float32)
                foreground = foreground.astype(np.float32)
                raw_comp_background = train_alpha * foreground + (1. - train_alpha) * background
                reduceced_comp_background = raw_comp_background - g_mean
                if np.random.random_sample() > 0.5:
                    reduceced_comp_background = np.fliplr(reduceced_comp_background)
                    raw_comp_background = np.fliplr(raw_comp_background)
                    background = np.fliplr(background)
                    train_alpha = np.fliplr(train_alpha)
                    trimap = np.fliplr(trimap)
                    foreground = np.fliplr(foreground)
                yield {
                          'input': reduceced_comp_background,
                          'trimap': trimap,
                          'original_background': background,
                          'raw_comp_background': raw_comp_background,
                          'foreground': foreground,
                      }, train_alpha

        ds = tf.data.Dataset.from_generator(_generator,
                                            (
                                                {
                                                    'input': tf.float32,
                                                    'trimap': tf.float32,
                                                    'original_background': tf.float32,
                                                    'raw_comp_background': tf.float32,
                                                    'foreground': tf.float32
                                                }, tf.float32),
                                            (
                                                {
                                                    'input': tf.TensorShape([320, 320, 3]),
                                                    'trimap': tf.TensorShape([320, 320, 1]),
                                                    'original_background': tf.TensorShape(
                                                        [320, 320, 3]),
                                                    'raw_comp_background': tf.TensorShape(
                                                        [320, 320, 3]),
                                                    'foreground': tf.TensorShape([320, 320, 3])},
                                                tf.TensorShape(
                                                    [320, 320, 1])))
        return ds.shuffle(10).repeat().batch(params['batch_size'])

    return _input_fn
