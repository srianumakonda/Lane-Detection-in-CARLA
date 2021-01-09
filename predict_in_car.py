import cv2
import numpy as np
import os
import logging
import tensorflow as tf 
from keras.models import *
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
logging.getLogger('tensorflow').setLevel(logging.FATAL)

import_model = load_model("C:\\Users\\User\\Documents\\Artificial Intelligence\\Lane-Detection\\model_10_epochs.tf")

capture_vid = cv2.VideoCapture(0);
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.avi', fourcc, 20.0, (640,480), 0)

while True:
    ret, frame = capture_vid.read()
    # print(capture_vid.get(cv2.CAP_PROP_FRAME_HEIGHT),capture_vid.get(cv2.CAP_PROP_FRAME_WIDTH))
    gray_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    pred_img = cv2.resize(gray_img,(128,128))
    model_pred = import_model.predict(np.array(pred_img).reshape(1,128,128,1))
    model_pred = np.asarray(model_pred,np.uint8)
    model_pred = np.squeeze(model_pred)

    merge_img = cv2.addWeighted(model_pred,0.95,pred_img,0.05,0)
    merge_img = cv2.resize(merge_img,(640,480))

    cv2.imshow('frame', merge_img)
    out.write(merge_img)

    # cv2.imshow('frame', gray_img)
    # out.write(gray_img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

capture_vid.release()
out.release()
cv2.destroyAllWindows()