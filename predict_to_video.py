import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image
from preprocess_data import non_aug_data

if __name__ == "__main__":
    pred_img_path = "predictions"
    filename_pred = []
    train_set = non_aug_data("X_train",".\\carla-dataset\\train")
    train_set = train_set.reshape(2922,128,128,1)
    framesize = (128,128)

    # for idx, filename in enumerate(os.listdir(pred_img_path)):
    #     if idx%8 != 0:
    #         os.remove(os.path.join(pred_img_path,filename))

    if int(train_set.ravel().shape[0])//128//128 == len(os.listdir(pred_img_path)):

        print("Successfully deleted unecessary files")

        for i in os.listdir(pred_img_path):
            filename_pred.append(i) 

        out = cv2.VideoWriter('output_video.avi',cv2.VideoWriter_fourcc(*'DIVX'), 1, framesize, False)

        for i in range(len(filename_pred)):
            img_act = train_set[i][:, :, ::-1].copy() 
            img_act = np.squeeze(img_act)
            img_act *= 255
            img_act = np.uint8(img_act)

            
            img_pred = cv2.imread(os.path.join(pred_img_path,filename_pred[i]))
            img_pred = cv2.cvtColor(img_pred, cv2.COLOR_BGR2GRAY)
            img_pred = cv2.resize(img_pred,(128,128))
            img_pred *= 255
            img_pred = np.uint8(img_pred)

            merge_img = cv2.addWeighted(img_pred,0.6,img_act,0.4,0)
            out.write(merge_img)
                    
    else:
        raise RuntimeError("Length of train images is not the same as prediction images.")
    out.release()
    print("Finished writing video.")