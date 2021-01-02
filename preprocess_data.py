import os
import shutil
from PIL import Image, ImageOps
from sklearn.model_selection import train_test_split
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

class FixData():

    def __init__(self, filepath):
        self.filepath = filepath
        self.train_path = self.filepath + "/train"
        self.train_label_path = self.filepath + "/train_label"
        self.test_path, self.test_label_path = self.rename_folders()
        self.val_path = None
        self.val_label_path = None

    def rename_folders(self):
        try:
            os.rename(os.path.join(self.filepath, "val"), os.path.join(self.filepath, "test"))
            os.rename(os.path.join(self.filepath, "val_label"), os.path.join(self.filepath, "test_label"))
            print("Directory names have successfully been changed.")
        except FileNotFoundError:
            print("Folder does not exist. Either cell has already ran or folder does not exist.")
        except OSError:
            pass
        return self.filepath + "/test", self.filepath + "/test_label"

    def move_folders(self):
        os.makedirs(os.path.join(self.filepath, "train_2"))
        os.makedirs(os.path.join(self.filepath, "train_label_2"))
        shutil.move(self.train_path, os.path.join(self.filepath, "train_2"))
        shutil.move(self.train_label_path, os.path.join(self.filepath, "train_label_2"))
        os.rename(os.path.join(self.filepath, "train_label_2"), os.path.join(self.filepath, "train_label"))
        os.rename(os.path.join(self.filepath, "train_2"), os.path.join(self.filepath, "train"))
        self.new_names()        

    def add_zeros(self, num):
        return str(num.zfill(4))
    
    def new_names(self):
        self.new_train_path = self.train_path + "/train"
        self.new_train_label_path = self.train_label_path + "/train_label"

    def rename_sort_data(self):
        for filename in os.listdir(self.new_train_path):
            updated_filename = filename[44:]
            updated_filename = os.path.join(self.new_train_path, self.add_zeros(updated_filename[:-4])+".png")
            os.rename(os.path.join(self.new_train_path, filename), updated_filename)

        for filename in os.listdir(self.new_train_label_path):
            updated_filename = self.add_zeros(filename[44:-10])
            updated_filename = os.path.join(self.new_train_label_path, updated_filename+".png")
            os.rename(os.path.join(self.new_train_label_path, filename), updated_filename)

        for filename in os.listdir(self.test_path):
            updated_filename = self.add_zeros(filename[44:-19])
            updated_filename = os.path.join(self.test_path, updated_filename+".png")
            os.rename(os.path.join(self.test_path, filename), updated_filename)

        for filename in os.listdir(self.test_label_path):
            updated_filename = self.add_zeros(filename[44:-25])
            updated_filename = os.path.join(self.test_label_path, updated_filename+".png")
            os.rename(os.path.join(self.test_label_path, filename), updated_filename)

    def create_val_set(self, pct):
        if len(os.listdir(self.new_train_path)) == len(os.listdir(self.new_train_label_path)):
            num_val_img = int(len(os.listdir(self.new_train_path))//(pct*100))
            os.makedirs(os.path.join(self.filepath, "val"))
            os.makedirs(os.path.join(self.filepath, "val_label"))
            val_path = self.filepath + "/val"
            val_label_path = self.filepath + "/val_label"
            for img in os.listdir(self.new_train_path)[-num_val_img:]:
               shutil.move(os.path.join(self.new_train_path, img), val_path + "/" + img)

            for img in os.listdir(self.new_train_label_path)[-num_val_img:]:
                shutil.move(os.path.join(self.new_train_label_path, img), val_label_path + "/" + img) 

            if len(os.listdir(val_path)) > 0 and len(os.listdir(val_label_path)) > 0:
                print("Validation directories created.")
        else:
            return "Files in train and train_label are not the length. Please double check this."
            
class SplitData(FixData):
    def __init__(self, filepath):
        super().__init__(filepath)
        self.X_train = []
        self.X_val = []
        self.X_test = []
        self.y_train = []
        self.y_val = []
        self.y_test = []

        self.val_path = self.filepath + "/val"
        self.val_label_path = self.filepath + "/val_label"
        self.new_train_path = self.train_path + "/train"
        self.new_train_label_path = self.train_label_path + "/train_label"

    def resize_img(self, img_size):
        print(self.val_path, self.val_label_path)
        for img in os.listdir(self.new_train_path):
            open_img = Image.open(os.path.join(self.new_train_path, img)).resize((img_size, img_size))
            open_img = ImageOps.grayscale(open_img)
            self.X_train.append(np.array(open_img)/255.0)

        for img in os.listdir(self.val_path):
            open_img = Image.open(os.path.join(self.val_path, img)).resize((img_size, img_size))
            open_img = ImageOps.grayscale(open_img)
            self.X_val.append(np.array(open_img)/255.0)

        for img in os.listdir(self.test_path):
            open_img = Image.open(os.path.join(self.test_path, img)).resize((img_size, img_size))
            open_img = ImageOps.grayscale(open_img)
            self.X_test.append(np.array(open_img)/255.0)

        for img in os.listdir(self.new_train_label_path):
            open_img = Image.open(os.path.join(self.new_train_label_path, img)).resize((img_size, img_size))
            open_img = ImageOps.grayscale(open_img)
            self.y_train.append(np.array(open_img)/255.0)

        for img in os.listdir(self.val_label_path):
            open_img = Image.open(os.path.join(self.val_label_path, img)).resize((img_size, img_size))
            open_img = ImageOps.grayscale(open_img)
            self.y_val.append(np.array(open_img)/255.0)

        for img in os.listdir(self.test_label_path):
            open_img = Image.open(os.path.join(self.test_label_path, img)).resize((img_size, img_size))
            open_img = ImageOps.grayscale(open_img)
            self.y_test.append(np.array(open_img)/255.0)

    def print_img_pair(self, num):
        fig = plt.figure(figsize=(10,10))
        plt.subplot(2, 2, 1)
        plt.imshow(np.array(self.X_train[num]), cmap="gray")
        plt.subplot(2, 2, 2)
        plt.imshow(np.array(self.y_train[num]), cmap="gray")

    def get_shapes(self):
        return f"X_train: {np.array(self.X_train).shape}, X_val: {np.array(self.X_val).shape}, X_test: {np.array(self.X_test).shape}, y_train: {np.array(self.y_train).shape}, y_val: {np.array(self.y_val).shape}, y_test: {np.array(self.y_test).shape}"

    def data(self):
        updated_X_train = np.array(self.X_train).reshape(2922,256,256,1)
        updated_y_train = np.array(self.y_train).reshape(2922,256,256,1)
        updated_X_val = np.array(self.X_val).reshape(153,256,256,1)
        updated_y_val = np.array(self.y_val).reshape(153,256,256,1)
        updated_X_test = np.array(self.X_test).reshape(129,256,256,1)
        updated_y_test = np.array(self.y_test).reshape(129,256,256,1)

        clean_data = dict(featurewise_center=True,
                          featurewise_std_normalization=True,
                          rotation_range=90.,
                          width_shift_range=0.1,
                          height_shift_range=0.1,
                          zoom_range=0.2,
                          rescale=1./255)
        X_data_gen = ImageDataGenerator(**clean_data)
        y_data_gen = ImageDataGenerator(**clean_data)
        X_data_gen.fit(updated_X_train, augment=True, seed=42)
        y_data_gen.fit(updated_y_train, augment=True, seed=42)

        X = X_data_gen.flow_from_directory(self.train_path, seed=42, class_mode=None, color_mode="grayscale", target_size=(256,256))
        y = y_data_gen.flow_from_directory(self.train_label_path, seed=42, class_mode=None, color_mode="grayscale", target_size=(256,256))
        return X, y, updated_X_val, updated_y_val, updated_X_test, updated_y_test

        


    
