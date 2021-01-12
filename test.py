import cv2
import numpy as np
import os
import tensorflow as tf 
from keras.models import *

import_model = load_model("C:\\Users\\User\\Documents\\Artificial Intelligence\\Lane-Detection\\model_10_epochs.tf")

capture_vid = cv2.VideoCapture("C:\\Users\\User\\Downloads\\Road Stock Footage HD.mp4");

while True:
    ret, frame = capture_vid.read()

    gray_img = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

    pred_img = cv2.resize(gray_img,(128,128))
    model_pred = import_model.predict(np.array(pred_img).reshape(1,128,128,1))
    model_pred = np.asarray(model_pred,np.uint8)
    model_pred = np.squeeze(model_pred)

    merge_img = cv2.addWeighted(model_pred,0.5,pred_img,0.5,0)
    merge_img = cv2.resize(merge_img,(640,480))

    cv2.imshow('frame', merge_img)
    if cv2.waitKey(1) and 0xFF == ord('q'):
        break

capture_vid.release()
cv2.destroyAllWindows()