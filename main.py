from preprocess_data import FixData
from visualize_data import visualize_img
from preprocess_data import SplitData
from model import UNet_Model
import loss

import numpy as np
import shutil
import os
import tensorflow as tf
from keras import backend as K
import random
import matplotlib
import matplotlib.pyplot as plt

if __name__ == "__main__":

    filepath = "carla-dataset" #make sure you specify the directory of the dataset here
    fix_data = FixData(filepath)
    fix_data.rename_folders()
    fix_data.rename_sort_data()
    fix_data.create_val_set(0.2)

    visualize_img(filepath, 2)
    fig = plt.figure(figsize = (20, 10)) 
    plt.bar(["Train", "Test"], [len(os.listdir(os.path.join(filepath, "train"))), len(os.listdir(os.path.join(filepath, "test")))],
            color="maroon")
    matplotlib.rc('xtick', labelsize=20) 
    matplotlib.rc('ytick', labelsize=20) 

    split_data = SplitData(filepath)
    split_data.resize_img(128,0,30)
    split_data.print_img_pair(0)
    X_train, y_train, X_val, y_val, X_test, y_test = split_data.data()
    print(split_data.get_shapes())



    model = UNet_Model()
    train_model = model.unet(input_size=(128,128,1),loss=loss.focal_tversky,metrics=loss.dice)
    # model.model_summary()


    class DisplayCallback(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            choose_rand = random.randint(0,int(np.array(X_test).ravel().shape[0])//128//128)
            prediction = np.array(X_test[choose_rand]).reshape(1,128,128,1)
            plt.figure(figsize=(15, 15))

            plt.subplot(3, 3, 1)
            plt.imshow(np.squeeze(X_test[choose_rand]),cmap="gray")
            plt.title("Image")

            plt.subplot(3, 3, 2)
            plt.imshow(np.squeeze(y_test[choose_rand]),cmap="gray")
            plt.title("Mask")

            plt.subplot(3, 3, 3)
            plt.imshow(np.squeeze(train_model.predict(prediction)),cmap="gray")
            plt.title("Predicted Mask")

            plt.axis('off')

            print ('\nSample Prediction after epoch {}\n'.format(epoch+1))
            plt.show()

    model.train_model(filepath=filepath, X_train=X_train, y_train=y_train, 
                    X_val=X_val, y_val=y_val, epochs=20, display_callback=DisplayCallback())

    train_model.evaluate(X_test,y_test)
