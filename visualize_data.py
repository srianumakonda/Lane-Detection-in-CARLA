import os
from PIL import Image
import matplotlib.pyplot as plt

def visualize_img(filepath, n_img):
  counter = 0
  images = []
  labels = []
  img_path = os.path.join(filepath, "train/")
  label_path = os.path.join(filepath, "train_label/")
  num_work =[]

  while counter < n_img:
    for i in range(3204):
      if os.path.exists(os.path.join(img_path, str(i) + ".png")):
        if os.path.exists(os.path.join(label_path, str(i) + ".png")):
          num_work.append(i)
          counter += 1
          continue
  num_work = num_work[:n_img]

  for i in num_work:
    image = Image.open(os.path.join(img_path, str(i) + ".png"))
    images.append(image)
    label = Image.open(os.path.join(label_path, str(i) + ".png"))
    labels.append(label)

  fig,a = plt.subplots(n_img,2,figsize=(n_img*15,n_img*10))

  for i in range(n_img):
    a[i][0].imshow(images[i])
    a[i][1].imshow(labels[i])

  plt.show()