import os
import shutil
from PIL import Image, ImageOps, ImageFilter
import numpy as np
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

    def add_zeros(self, num):
        return str(num.zfill(4))

    def rename_sort_data(self):
        for filename in os.listdir(self.train_path):
            updated_filename = filename[44:]
            updated_filename = os.path.join(self.train_path, self.add_zeros(updated_filename[:-4])+".png")
            os.rename(os.path.join(self.train_path, filename), updated_filename)

        for filename in os.listdir(self.train_label_path):
            updated_filename = self.add_zeros(filename[44:-10])
            updated_filename = os.path.join(self.train_label_path, updated_filename+".png")
            os.rename(os.path.join(self.train_label_path, filename), updated_filename)

        for filename in os.listdir(self.test_path):
            updated_filename = self.add_zeros(filename[44:-19])
            updated_filename = os.path.join(self.test_path, updated_filename+".png")
            os.rename(os.path.join(self.test_path, filename), updated_filename)

        for filename in os.listdir(self.test_label_path):
            updated_filename = self.add_zeros(filename[44:-25])
            updated_filename = os.path.join(self.test_label_path, updated_filename+".png")
            os.rename(os.path.join(self.test_label_path, filename), updated_filename)

    def create_val_set(self, pct):
        if len(os.listdir(self.train_path)) == len(os.listdir(self.train_label_path)):
            num_val_img = int(len(os.listdir(self.train_path))//(pct*100))
            os.makedirs(os.path.join(self.filepath, "val"))
            os.makedirs(os.path.join(self.filepath, "val_label"))
            val_path = self.filepath + "/val"
            val_label_path = self.filepath + "/val_label"
            for img in os.listdir(self.train_path)[-num_val_img:]:
               shutil.move(os.path.join(self.train_path, img), val_path + "/" + img)

            for img in os.listdir(self.train_label_path)[-num_val_img:]:
                shutil.move(os.path.join(self.train_label_path, img), val_label_path + "/" + img) 

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

    def process_img(self, img, threshold_val, img_size, subset, rot_range):
        if rot_range % 5 != 0:
            raise ValueError("Number must be divisable by 5. Please input a number that is a multiple of 5")
        else:
            if subset == "train":
                open_img = Image.open(os.path.join(self.train_path, img)).resize((img_size, img_size))
                open_img = ImageOps.grayscale(open_img)
                open_img = np.array(open_img)
                open_img[open_img<threshold_val] = 0
                open_img = Image.fromarray(open_img)
                self.X_train.append(np.array(open_img)/np.max(open_img))
                flip_img = ImageOps.flip(open_img)
                self.X_train.append(np.array(flip_img)/np.max(flip_img))
                mir_img = ImageOps.mirror(open_img)
                self.X_train.append(np.array(mir_img)/np.max(mir_img))
                for i in range(5,rot_range,5):
                    rot_img = open_img.rotate(i)
                    self.X_train.append(np.array(rot_img)/np.max(rot_img))

            if subset == "val":
                open_img = Image.open(os.path.join(self.val_path, img)).resize((img_size, img_size))
                open_img = ImageOps.grayscale(open_img)
                open_img = np.array(open_img)
                open_img[open_img<threshold_val] = 0
                open_img = Image.fromarray(open_img)
                self.X_val.append(np.array(open_img)/np.max(open_img))
                flip_img = ImageOps.flip(open_img)
                self.X_val.append(np.array(flip_img)/np.max(flip_img))
                mir_img = ImageOps.mirror(open_img)
                self.X_val.append(np.array(mir_img)/np.max(mir_img))
                for i in range(5,rot_range,5):
                    rot_img = open_img.rotate(i)
                    self.X_val.append(np.array(rot_img)/np.max(rot_img))

            if subset == "test":
                open_img = Image.open(os.path.join(self.test_path, img)).resize((img_size, img_size))
                open_img = ImageOps.grayscale(open_img)
                open_img = np.array(open_img)
                open_img[open_img<threshold_val] = 0
                open_img = Image.fromarray(open_img)
                self.X_test.append(np.array(open_img)/np.max(open_img))
                flip_img = ImageOps.flip(open_img)
                self.X_test.append(np.array(flip_img)/np.max(flip_img))
                mir_img = ImageOps.mirror(open_img)
                self.X_test.append(np.array(mir_img)/np.max(mir_img))
                for i in range(5,rot_range,5):
                    rot_img = open_img.rotate(i)
                    self.X_test.append(np.array(rot_img)/np.max(rot_img))

    def process_mask(self, img, img_size, subset, rot_range):

        if rot_range % 5 != 0:
            raise ValueError("Number must be divisable by 5. Please input a number that is a multiple of 5")
        else:
            if subset == "train":
                open_img = Image.open(os.path.join(self.train_label_path, img)).resize((img_size, img_size))
                open_img = ImageOps.grayscale(open_img)
                open_img = np.array(open_img)
                open_img[open_img>0] = 1
                open_img = Image.fromarray(open_img)
                self.y_train.append(np.array(open_img))
                flip_img = ImageOps.flip(open_img)
                self.y_train.append(np.array(flip_img))
                mir_img = ImageOps.mirror(open_img)
                self.y_train.append(np.array(mir_img))
                for i in range(5,rot_range,5):
                    rot_img = open_img.rotate(i)
                    self.y_train.append(np.array(rot_img))
            if subset == "val":
                open_img = Image.open(os.path.join(self.val_label_path, img)).resize((img_size, img_size))
                open_img = ImageOps.grayscale(open_img)
                open_img = np.array(open_img)
                open_img[open_img>0] = 1
                open_img = Image.fromarray(open_img)
                self.y_val.append(np.array(open_img))
                flip_img = ImageOps.flip(open_img)
                self.y_val.append(np.array(flip_img))
                mir_img = ImageOps.mirror(open_img)
                self.y_val.append(np.array(mir_img))
                for i in range(5,rot_range,5):
                    rot_img = open_img.rotate(i)
                    self.y_val.append(np.array(rot_img))
            if subset == "test":
                open_img = Image.open(os.path.join(self.test_label_path, img)).resize((img_size, img_size))
                open_img = ImageOps.grayscale(open_img)
                open_img = np.array(open_img)
                open_img[open_img>0] = 1
                open_img = Image.fromarray(open_img)
                self.y_test.append(np.array(open_img))
                flip_img = ImageOps.flip(open_img)
                self.y_test.append(np.array(flip_img))
                mir_img = ImageOps.mirror(open_img)
                self.y_test.append(np.array(mir_img))
                for i in range(5,rot_range,5):
                    rot_img = open_img.rotate(i)
                    self.y_test.append(np.array(rot_img))

    def resize_img(self, img_size, threshold_val, rot_range):
        for img in os.listdir(self.train_path):
            self.process_img(img,threshold_val,img_size,"train",rot_range)

        for img in os.listdir(self.val_path):
            self.process_img(img,threshold_val,img_size,"val",rot_range)
        
        for img in os.listdir(self.test_path):
            self.process_img(img,threshold_val,img_size,"test",rot_range)

        for img in os.listdir(self.train_label_path):
            self.process_mask(img,img_size,"train",rot_range)

        for img in os.listdir(self.val_label_path):
            self.process_mask(img,img_size,"val",rot_range)
    
        for img in os.listdir(self.test_label_path):
            self.process_mask(img,img_size,"test",rot_range)
            
    def print_img_pair(self, num):
        fig = plt.figure(figsize=(10,10))
        plt.subplot(2, 2, 1)
        plt.imshow(np.array(self.X_train[num]), cmap="gray")
        plt.subplot(2, 2, 2)
        plt.imshow(np.array(self.y_train[num]), cmap="gray")

    def get_shapes(self):
        return f"X_train: {np.array(self.X_train).shape}, X_val: {np.array(self.X_val).shape}, X_test: {np.array(self.X_test).shape}, y_train: {np.array(self.y_train).shape}, y_val: {np.array(self.y_val).shape}, y_test: {np.array(self.y_test).shape}"

    def data(self):
        updated_X_train = np.array(self.X_train).reshape(int(np.array(self.X_train).ravel().shape[0])//128//128,128,128,1).astype('float64')
        updated_y_train = np.array(self.y_train).reshape(int(np.array(self.y_train).ravel().shape[0])//128//128,128,128,1).astype('float64')
        updated_X_val = np.array(self.X_val).reshape(int(np.array(self.X_val).ravel().shape[0])//128//128,128,128,1).astype('float64')
        updated_y_val = np.array(self.y_val).reshape(int(np.array(self.y_val).ravel().shape[0])//128//128,128,128,1).astype('float64')
        updated_X_test = np.array(self.X_test).reshape(int(np.array(self.X_test).ravel().shape[0])//128//128,128,128,1).astype('float64')
        updated_y_test = np.array(self.y_test).reshape(int(np.array(self.y_test).ravel().shape[0])//128//128,128,128,1).astype('float64')
        return updated_X_train, updated_y_train, updated_X_val, updated_y_val, updated_X_test, updated_y_test

def non_aug_data(subset,data_path):
    if subset == "X_train":
        aug_list = []
        for img in os.listdir(data_path):
            open_img = Image.open(os.path.join(data_path, img)).resize((128, 128))
            open_img = ImageOps.grayscale(open_img)
            open_img = np.array(open_img)
            aug_list.append(open_img)
        return np.array(aug_list)/np.max(aug_list)

    if subset == "X_val":
        aug_list = []
        for img in os.listdir(data_path):
            open_img = Image.open(os.path.join(data_path, img)).resize((128, 128))
            open_img = ImageOps.grayscale(open_img)
            open_img = np.array(open_img)
            aug_list.append(open_img)
        return np.array(aug_list)/np.max(aug_list)

    if subset == "X_test":
        aug_list = []
        for img in os.listdir(data_path):
            open_img = Image.open(os.path.join(data_path, img)).resize((128, 128))
            open_img = ImageOps.grayscale(open_img)
            open_img = np.array(open_img)
            aug_list.append(open_img)
        return np.array(aug_list)/np.max(aug_list)

    if subset == "y_train":
        aug_list = []
        for img in os.listdir(data_path):
            open_img = Image.open(os.path.join(data_path, img)).resize((128, 128))
            open_img = ImageOps.grayscale(open_img)
            open_img = np.array(open_img)
            open_img[open_img>0] = 1
            aug_list.append(open_img)
        return np.array(aug_list)

    if subset == "y_val":
        aug_list = []
        for img in os.listdir(data_path):
            open_img = Image.open(os.path.join(data_path, img)).resize((128, 128))
            open_img = ImageOps.grayscale(open_img)
            open_img = np.array(open_img)
            open_img[open_img>0] = 1
            aug_list.append(open_img)
        return np.array(aug_list)

    if subset == "y_test":
        aug_list = []
        for img in os.listdir(data_path):
            open_img = Image.open(os.path.join(data_path, img)).resize((128, 128))
            open_img = ImageOps.grayscale(open_img)
            open_img = np.array(open_img)
            open_img[open_img>0] = 1
            aug_list.append(open_img)
        return np.array(aug_list)