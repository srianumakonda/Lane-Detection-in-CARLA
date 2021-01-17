import tensorflow as tf
import cv2
import os
import numpy as np
from keras.models import *
from keras import backend as K
import matplotlib.pyplot as plt

if __name__ == "__main__":

    # Credit to https://github.com/nabsabraham/focal-tversky-unet/blob/master/losses.py for the loss functions
    smooth=1
    def tversky(y_true, y_pred):
        y_true_pos = K.flatten(y_true)
        y_pred_pos = K.flatten(y_pred)
        true_pos = K.sum(y_true_pos * y_pred_pos)
        false_neg = K.sum(y_true_pos * (1-y_pred_pos))
        false_pos = K.sum((1-y_true_pos)*y_pred_pos)
        alpha = 0.7
        return (true_pos + smooth)/(true_pos + alpha*false_neg + (1-alpha)*false_pos + smooth)
    def focal_tversky(y_true,y_pred):
        pt_1 = tversky(y_true, y_pred)
        gamma = 0.75
        return K.pow((1-pt_1), gamma)
    def dice(y_true, y_pred):
        smooth = 1.
        y_true_f = K.flatten(y_true)
        y_pred_f = K.flatten(y_pred)
        intersection = K.sum(y_true_f * y_pred_f)
        score = (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
        return score

    loaded_model = load_model("final_model.tf",
                            custom_objects={'focal_tversky':focal_tversky,'dice':dice})

    img_list = []
    pred_list = []

    filename_list = []

    for filename in os.listdir("img/val/"):
        filename_list.append(filename)
        print(filename)

    #     img = cv2.imread(filename)
    #     img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #     img = cv2.resize(img,(128,128))
    #     img_list.append(img)
    #     pred_list.append(np.squeeze(loaded_model.predict(np.array(img).reshape(1,128,128,1))))
    # img_list = np.asarray(img_list)
    # pred_list = np.asarray(pred_list)

    # plt.imshow(pred_list[0])

