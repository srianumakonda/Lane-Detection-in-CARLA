import loss
import tensorflow as tf
import cv2
import os
import numpy as np
from keras.models import *
from keras import backend as K
import matplotlib.pyplot as plt

if __name__ == "__main__":

    loaded_model = load_model("final_model.tf",
                            custom_objects={'focal_tversky':loss.focal_tversky,'dice':loss.dice})

    img_list = []
    pred_list = []

    for filename in os.listdir("img/train"):
        img = cv2.imread(filename)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img,(128,128))
        img_list.append(img)
        pred_list.append(np.squeeze(loaded_model.predict(np.array(img).reshape(1,128,128,1))))
    img_list = np.asarray(img_list)
    pred_list = np.asarray(pred_list)

    plt.imshow(pred_list[0])

