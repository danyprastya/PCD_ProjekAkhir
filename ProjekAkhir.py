#===========================Mengidentifikasi tingkat kematangan buah tomat berdasarkan warna==============================#
#Tahapan yang diperlukan
# 1. Memasukkan dataset untuk Train dan Test (DONE, BELUM DI TRAIN AJA SAMA TEST) NOTE:KEDUA DEF TRAIN & TEST MUNGKIN BISA DIBUAT LEBIH EFISIEN CARA KERJANYA
# 2. Melakukan pre proses pada foto yang jadi subjek (GA TERLALU YAKIN)
# 3. Memproses gambar sesuai dengan dataset
# 4. Tampilkan hasilnya di GUI yang ada
# 5. DONE GA BANG ??!

#===============================================Import Library y Dibutuhkan===============================================#
#Library untuk akses pathing
import os
import shutil
import itertools
import pathlib
from PIL import Image
from PIL import ImageEnhance
import random

#Library untuk handling & visualisasi data dan subjek terkait
import numpy as np
import pandas as pd
import cv2 
import matplotlib.pyplot as plt #untuk visualisasi data
import seaborn as sns #library tambahan untuk visualisasi data
from skimage.io import imread
from sklearn.model_selection import train_test_split #untuk train dataset
from sklearn.preprocessing import OneHotEncoder #untuk preprocessing data menjadi vektor biner
from sklearn.metrics import confusion_matrix, classification_report #untuk ngukur performa model & buat data detail tentang performa model 
sns.set_style('whitegrid')

#Library untuk Deep Learning (Metode CNN)
import tensorflow as tf
from tensorflow import keras
import tensorflow.keras.backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input
from tensorflow.keras.layers import Dense, Flatten, GlobalAveragePooling2D, Dropout

import warnings
warnings.filterwarnings('ignore')

#=======================================Memasukkan dataset untuk Train & Test=============================================#

K.clear_session()

#declare variabel path nya
dataset = 'dataset_tomat'
path_data_train =  'dataset_tomat'
path_data_test =  'dataset_tomat' #mungkin bakal diilangin, gatau juga


#hitung total gambar dalam folder2nya
def hitung_data(rootdir):
    for path in pathlib.Path(rootdir).iterdir():
        if path.is_dir():
            print("Ada " + str(len([name for name in os.listdir(path) 
            if os.path.isfile(os.path.join(path,name))])) 
            + " file didalam " + str(path.name))


#membuat dataset train
def data_train(path_data_train):
    #inisialisasi variabel untuk path dari data
    path_file = []
    label = []
    
    #membuka direktori yang bersangkutan
    folds = os.listdir(path_data_train)

    #susun alamat direktorinya sesuai variabel yang bersangkutan
    for fold in folds :
        f_path = os.path.join(path_data_train, fold)
        filelists = os.listdir(f_path)

        for file in filelists:
            path_file.append(os.path.join(f_path, file))
            label.append(fold)

    #membuat struktur datanya dengan di concatenate dua matrix 1 dimensi yang berupa path foto dan foto nya sendiri menjadi dataframe
    seri_f = pd.Series(path_file, name= 'Path File')
    seri_l = pd.Series(label, name= 'Label')
    train_df = pd.concat([seri_f, seri_l], axis=1)
    print("\n", train_df , "\n")


#membuat dataset test
def data_test(path_data_test):
    #inisialisasi variabel untuk path dari data
    path_file = []
    label = []

    #membuka direktori yang bersangkutan
    folds = os.listdir(path_data_train)

    #susun alamat direktorinya sesuai variabel yang bersangkutan
    for fold in folds :
        f_path = os.path.join(path_data_train, fold)
        filelists = os.listdir(f_path)

        for file in filelists:
            path_file.append(os.path.join(f_path, file))
            label.append(fold)

    #membuat struktur datanya dengan di concatenate dua matrix 1 dimensi yang berupa path foto dan foto nya sendiri menjadi dataframe
    seri_f = pd.Series(path_file, name= 'Path File')
    seri_l = pd.Series(label, name= 'Label')
    test_df = pd.concat([seri_f, seri_l], axis=1)
    
    return test_df

test_df = data_test(path_data_test)
valid , test = train_test_split(test_df, train_size= 0.5, shuffle=True, random_state= 42)

hitung_data(dataset)

#===================================Melakukan pre processing gambar pada dataset==========================================#
folder_gambar = 'Ripe'
jmlh_gambar = 2

def preproses():
    j=1
    plt.figure(figsize=(10,10))
    for i in range(jmlh_gambar):
        folder = os.path.join(path_data_test,folder_gambar)
        a = random.choice(os.listdir(folder))
        
        img = Image.open(os.path.join(folder,a))
        img_copy = img.copy()
        
        plt.subplot(jmlh_gambar,2,j)
        plt.title(label='Original', loc='center')
        plt.imshow(img)
        j += 1
        
        img1 = ImageEnhance.Color(img_copy).enhance(1.35)
        img1 = ImageEnhance.Contrast(img1).enhance(1.45)
        img1 = ImageEnhance.Sharpness(img1).enhance(2.5)
        
        plt.subplot(jmlh_gambar,2,j)
        plt.title(label='Setelah Diproses', loc = 'center')
        plt.imshow(img1)
        j += 1
    plt.show()

preproses()