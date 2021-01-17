import loss
import tensorflow as tf
import cv2
import os
import numpy as np
from tensorflow.keras import backend as K
import matplotlib.pyplot as plt
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

if __name__ == "__main__":

    loaded_model = tf.keras.models.load_model("final_model.h5",
                            custom_objects={'focal_tversky':loss.focal_tversky,'dice':loss.dice})

    img_list = []
    pred_list = []
    filepath = "carla-dataset\\train"

    for filename in os.listdir(filepath):
        img = cv2.imread(os.path.join(filepath,filename))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img,(128,128))
        img_list.append(img)
        pred_list.append(np.squeeze(loaded_model.predict(np.array(img).reshape(1,128,128,1))))
    img_list = np.asarray(img_list)
    pred_list = np.asarray(pred_list)

    plt.imshow(pred_list[0])
    plt.show()

