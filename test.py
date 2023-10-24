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