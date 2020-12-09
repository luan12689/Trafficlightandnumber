import cv2
import os
from numpy import expand_dims
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import ImageDataGenerator
from matplotlib import pyplot as plt

iglobal = 0


def augmentation(pathfile, file, savepathfile):
    global iglobal
    img = cv2.imread(pathfile+file)
    scale_percent = 30  # percent of original size
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)
    # resize image
    img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
    cv2.imwrite(savepathfile+str(iglobal)+".jpg", img)
    iglobal = iglobal+1

    img = load_img(pathfile+file)
    img = img_to_array(img)
    data = expand_dims(img, 0)

    # Dinh nghia 1 doi tuong Data Generator voi bien phap chinh sua anh Zoom tu 0.5x den 2x
    myImageGen = ImageDataGenerator(zoom_range=[0.5, 2])
    # Batch_Size= 1 -> Moi lan sinh ra 1 anh
    gen = myImageGen.flow(data, batch_size=1)
    # Sinh ra 9 anh va hien thi len man hinh
    for i in range(3):
        myBatch = gen.next()
        image = myBatch[0].astype('uint8')
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(savepathfile+str(iglobal)+".jpg", image)
        iglobal = iglobal + 1

    myImageGen = ImageDataGenerator(width_shift_range=[-150, 150])
    # Batch_Size= 1 -> Moi lan sinh ra 1 anh
    gen = myImageGen.flow(data, batch_size=1)
    # Sinh ra 9 anh va hien thi len man hinh
    for i in range(3):
        myBatch = gen.next()
        image = myBatch[0].astype('uint8')
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(savepathfile + str(iglobal) + ".jpg", image)
        iglobal = iglobal + 1

    myImageGen = ImageDataGenerator(shear_range=45)
    # Batch_Size= 1 -> Moi lan sinh ra 1 anh
    gen = myImageGen.flow(data, batch_size=1)
    for i in range(3):
        myBatch = gen.next()
        image = myBatch[0].astype('uint8')
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(savepathfile + str(iglobal) + ".jpg", image)
        iglobal = iglobal + 1

    myImageGen = ImageDataGenerator(horizontal_flip=True, vertical_flip=True)
    # Batch_Size= 1 -> Moi lan sinh ra 1 anh
    gen = myImageGen.flow(data, batch_size=1)
    for i in range(3):
        myBatch = gen.next()
        image = myBatch[0].astype('uint8')
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(savepathfile + str(iglobal) + ".jpg", image)
        iglobal = iglobal + 1

    myImageGen = ImageDataGenerator(brightness_range=[0.5, 2.0])
    # Batch_Size= 1 -> Moi lan sinh ra 1 anh
    gen = myImageGen.flow(data, batch_size=1)
    for i in range(3):
        myBatch = gen.next()
        image = myBatch[0].astype('uint8')
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(savepathfile + str(iglobal) + ".jpg", image)
        iglobal = iglobal + 1


if __name__ == '__main__':
    path = os.path.abspath('..') + '//Augmentation final//data//'
    savepath = os.path.abspath('..') + '//Augmentation final//result//'
    # path = os.path.abspath('..')+'//light//'
    print(path)
    for f in os.listdir(path):
        print(f)
        if f.endswith('.jpg') or f.endswith('.JPG'):
            img = cv2.imread(path+f)
            augmentation(path, f, savepath)

