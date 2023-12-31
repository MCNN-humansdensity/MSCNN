# -*- coding: utf-8 -*-
"""MSCNN

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1_gXe2mGKvqH6Y_J9RN_teC6KkAPo_Jz3

## Data
"""

# -*- coding: utf-8 -*-
"""
@Project Name  ColabProject
@File Name:    MSCNN
@Software:     Google Colab
@Time:         17/Dec/2022
@Author:       zaicy
@contact:      zaicyxu@gmail.com
@version:      1.0
@Description:  None
"""

import cv2
import os
import numpy as np
import scipy.io as sio
from multiprocessing.dummy import Pool as ThreadPool

import matplotlib.pyplot as plt

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
def visualization(img, dmap):
    plt.figure()

    plt.subplot(121)
    plt.imshow(img)

    plt.subplot(122)
    # plt.imshow(dmap)
    plt.colorbar()

    plt.tight_layout()
    plt.show()

def read_annotations():
    """read annotation data.

    Returns:
        count: ndarray, head count.
        position: ndarray, coordinate.
    """
    data = sio.loadmat(r'/content/drive/MyDrive/MSCNN-humansdensity-master/mall_dataset/mall_gt.mat')
    count = data['count']
    position = data['frame'][0]

    return count, position


def map_pixels(img, image_key, annotations, size):
    """map annotations to density map.

    Arguments:
        img: ndarray, img.
        image_key: int, image_key.
        annotations: ndarray, annotations.
        size: resize size.

    Returns:
        pixels: ndarray, density map.
    """
    gaussian_kernel = 15
    h, w = img.shape[:-1]
    sh, sw = size / h, size / w
    pixels = np.zeros((size, size))

    for a in annotations[image_key][0][0][0]:
        x, y = int(a[0] * sw), int(a[1] * sh)
        if y >= size or x >= size:
            print("{},{} is out of range, skipping annotation for {}".format(x, y, image_key))
        else:
            pixels[y, x] += 1

    pixels = cv2.GaussianBlur(pixels, (gaussian_kernel, gaussian_kernel), 0)

    return pixels


def get_data(i, size, annotations):
    """get data accoding to the image_key.

    Arguments:
        i: int, image_key.
        size: int, input shape of network.
        annotations: ndarray, annotations.

    Returns:
        img: ndarray, img.
        density_map: ndarray, density map.
    """
    name = r'/content/drive/MyDrive/MSCNN-humansdensity-master/mall_dataset/frames/seq_{}.jpg'.format(str(i + 1).zfill(6))
    img = cv2.imread(name)

    density_map = map_pixels(img, i, annotations, size // 4)

    img = cv2.resize(img, (size, size))
    img = img / 255.

    density_map = np.expand_dims(density_map, axis=-1)

    return img, density_map


def generator(indices, batch, size):
    """data generator.

    Arguments:
        indices: list, image_key.
        batch: int, batch size.
        size: int, input shape of network.

    Returns:
        images: ndarray, batch images.
        labels: ndarray, batch density maps.
    """
    count, position = read_annotations()

    i = 0
    n = len(indices)

    if batch > n:
        raise Exception('Batch size {} is larger than the number of dataset {}!'.format(batch, n))

    while True:
        if i + batch >= n:
            np.random.shuffle(indices)
            i = 0
            continue

        pool = ThreadPool(2)
        res = pool.map(lambda x: get_data(x, size, position), indices[i: i + batch])
        pool.close()
        pool.join()

        i += batch
        images = []
        labels = []

        for r in res:
            images.append(r[0])
            labels.append(r[1])

        images = np.array(images)
        labels = np.array(labels)

        yield images, labels


if __name__ == '__main__':
    count, position = read_annotations()
    img, density_map = get_data(10, 224, position)

    print(count[10][0])
    print(int(np.sum(density_map)))
    visualization(img, density_map)

"""## Model"""

# -*- coding: utf-8 -*-

# from keras.layers import Input, Conv2D, MaxPooling2D, concatenate, Activation
from keras.layers import *
from keras.models import Model
from keras.regularizers import l2
from keras.utils.vis_utils import plot_model
import pydotplus as pydot


def MSB(filters):
    """Multi-Scale Blob.

    Arguments:
        filters: int, filters num.

    Returns:
        f: function, layer func.
    """
    params = {'activation': 'relu', 'padding': 'same',
              'kernel_regularizer': l2(5e-4)}

    def f(x):
        x1 = Conv2D(filters, 9, **params)(x)
        x2 = Conv2D(filters, 7, **params)(x)
        x3 = Conv2D(filters, 5, **params)(x)
        x4 = Conv2D(filters, 3, **params)(x)
        x = concatenate([x1, x2, x3, x4])
        x = BatchNormalization()(x)
        x = Activation('relu')(x)

        return x
    return f


def MSCNN(input_shape):
    """Multi-scale convolutional neural network for crowd counting.

    Arguments:
        input_shape: tuple, image shape with (w, h, c).

    Returns:
        model: Model, keras model.
    """
    inputs = Input(shape=input_shape)

    x = Conv2D(64, 9, activation='relu', padding='same')(inputs)
    x = MSB(4 * 16)(x)
    x = MaxPooling2D()(x)
    x = MSB(4 * 32)(x)
    x = MSB(4 * 32)(x)
    x = MaxPooling2D()(x)
    x = MSB(3 * 64)(x)
    x = MSB(3 * 64)(x)
    x = Conv2D(1000, 1, activation='relu', kernel_regularizer=l2(5e-4))(x)
    x = Conv2D(1, 1, activation='relu')(x)

    model = Model(inputs=inputs, outputs=x)

    return model


if __name__ == '__main__':
    model = MSCNN((224, 224, 3))

    print(model.summary())
    # plot_model(model, to_file='images\model.png', show_shapes=True)

"""## Test"""

# -*- coding: utf-8 -*-

import cv2
import numpy as np
import sklearn.metrics as metrics
from PIL import Image
import matplotlib.pyplot as plt
from model import MSCNN
from data import visualization


def eva_regress(y_true, y_pred):
    """Evaluation
    evaluate the predicted resul.

    # Arguments
        y_true: List/ndarray, ture data.
        y_pred: List/ndarray, predicted data.
    """
    mae = metrics.mean_absolute_error(y_true, y_pred)
    mse = metrics.mean_squared_error(y_true, y_pred)

    print('mae:%f' % mae)
    print('mse:%f' % mse)


if __name__ == '__main__':

    model = MSCNN((224, 224, 3))
    model.load_weights('model\\final_weights.h5')
    cap = cv2.VideoCapture("test.mp4")
    while True:
        ret, frame = cap.read()
        img = frame
        if frame is None:
            break
        ori_img = img.copy()
        img = cv2.resize(img, (224, 224))
        img = img / 255.
        img = np.expand_dims(img, axis=0)
        dmap = model.predict(img)[0][:, :, 0]
        dmap = cv2.GaussianBlur(dmap, (15, 15), 0)
        height, width = dmap.shape
        fig, ax = plt.subplots()
        plt.figure(figsize=(10, 10))
        fig.set_size_inches(width / 100.0 / 3.0, height / 100.0 / 3.0)
        plt.gca().xaxis.set_major_locator(plt.NullLocator())
        plt.gca().yaxis.set_major_locator(plt.NullLocator())
        plt.subplots_adjust(top=1, bottom=0, left=0, right=1, hspace=0, wspace=0)
        plt.margins(0, 0)
        plt.imshow(dmap)
        plt.savefig("result.jpg")
        re_img = cv2.imread("result.jpg")
        ori_img = cv2.resize(ori_img, (400, 400))
        re_img = cv2.resize(re_img, (400, 400))
        cv2.imshow("mg0", ori_img)
        cv2.imshow("img1", re_img)
        cv2.waitKey(1)
        # visualization(img[0], dmap)
        print('count:', int(np.sum(dmap)))

"""## Train"""

# -*- coding: utf-8 -*-

import os
import sys
import argparse
import pandas as pd

from keras.optimizers import gradient_descent_v2
from keras.callbacks import ReduceLROnPlateau
from sklearn.model_selection import train_test_split

from model import MSCNN
from data import generator


def main(argv):
    parser = argparse.ArgumentParser()
    # Required arguments.
    parser.add_argument(
        "--size",
        default=224,
        help="The image size of train sample.")
    parser.add_argument(
        "--batch",
        default=2,
        help="The number of train samples per batch.")
    parser.add_argument(
        "--epochs",
        default=20,
        help="The number of train iterations.")

    args = parser.parse_args()

    train(int(args.batch), int(args.epochs),int(args.size))


def train(batch, epochs, size):
    """Train the model.

    Arguments:
        batch: Integer, The number of train samples per batch.
        epochs: Integer, The number of train iterations.
        size: Integer, image size.
    """
    if not os.path.exists('model'):
        os.makedirs('model')

    model = MSCNN((size, size, 3))

    opt = gradient_descent_v2.SGD(lr=1e-5, momentum=0.9, decay=0.0005)
    model.compile(optimizer=opt, loss='mse')

    lr = ReduceLROnPlateau(monitor='loss', min_lr=1e-7)

    indices = list(range(1500))
    train, test = train_test_split(indices, test_size=0.25)

    hist = model.fit_generator(
        generator(train, batch, size),
        validation_data=generator(test, batch, size),
        steps_per_epoch=len(train) // batch,
        validation_steps=len(test) // batch,
        epochs=epochs,
        callbacks=[lr])

    model.save_weights('model\\final_weights.h5')

    df = pd.DataFrame.from_dict(hist.history)
    df.to_csv('model\\history.csv', index=False, encoding='utf-8')


if __name__ == '__main__':
    main(sys.argv)