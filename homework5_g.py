import os
import sys
import glob
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.misc.pilutil import imread,imresize

import tensorflow as tf
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
# from tensorflow.keras.layers import Merge
from tensorflow.keras.optimizers import *
import tensorflow.keras.backend as K

###############################################################
# Training Data List Creat
###############################################################
train_real_data_dir = 'train/Real/'
train_white_data_dir = 'train/White/'

# real_list = glob.glob(train_real_data_dir)
# train_real_data_list = []
# train_real_data_list.extend(real_list)
train_real_data_list =  ['train/Real/'+dirname for dirname in os.listdir(train_real_data_dir)]
print(os.listdir(train_real_data_dir))


# white_list = glob.glob(train_white_data_dir)
# train_white_data_list = []
# train_white_data_list.extend(white_list)
train_white_data_list =  ['train/White/'+dirname for dirname in os.listdir(train_white_data_dir)]
print(os.listdir(train_real_data_dir))

###############################################################
# Define D and G and parameter
###############################################################
channels = 1
img_row = img_col = 128
img_shape=(channels, img_row, img_col)

def dis(input_shape, ndf=64):
    def conv_block(x, filters, stride, bn=False, lrelu=False):
        x = Lambda(lambda x: tf.pad(x, [[0,0], [0,0], [1,1], [1,1]], mode="CONSTANT")) (x)
        x = Conv2D(filters=filters, kernel_size=4, strides=(stride, stride), padding="valid",
                kernel_initializer=tf.random_normal_initializer(0, 0.02), data_format="channels_first") (x)
        if bn:
            x = BatchNormalization(axis=1, momentum=0.1, epsilon=1e-5,
                    gamma_initializer=tf.random_normal_initializer(1.0, 0.02)) (x)
        if lrelu:
            x = LeakyReLU(alpha=0.2) (x)
        return x
     
    img_A = Input(input_shape)
    img_B = Input(input_shape)
    combined_img = Concatenate(axis=1)([img_A, img_B])

    x1 = conv_block(combined_img, ndf, 2, lrelu=True)
    x2 = conv_block(x1, ndf * 2, 2, bn=True, lrelu=True)
    x3 = conv_block(x2, ndf * 4, 2, bn=True, lrelu=True)
    x4 = conv_block(x3, ndf * 8, 1, bn=True, lrelu=True)
    x5 = conv_block(x4, 1, 1)
    out = Activation('sigmoid') (x5)
    
    model = Model([img_A,img_B], out)
    return model

def gen(input_shape, ngf=64, ** kwargs):
    def conv_block(x, filters, lrelu=False, bn=False):
        if lrelu:
            x = LeakyReLU(alpha=0.2) (x)
        x = Conv2D(filters=filters, kernel_size=4, strides=(2, 2), padding="same",
                kernel_initializer=tf.random_normal_initializer(0, 0.02), data_format="channels_first") (x)
        if bn:
            x = BatchNormalization(axis=1, momentum=0.1, epsilon=1e-5,
                    gamma_initializer=tf.random_normal_initializer(1.0, 0.02)) (x)
        return x
    
    def deconv_block(input, filters, skip_input=None, bn=False, dropout=0.0):
        x = Concatenate(axis=1) ([input, skip_input]) if skip_input != None else input
        x = Activation('relu') (x)
        x = Conv2DTranspose(filters=filters, kernel_size=4, strides=(2, 2), padding="same",
                kernel_initializer=tf.random_normal_initializer(0, 0.02), data_format="channels_first") (x)
        if bn:
            x = BatchNormalization(axis=1, momentum=0.1, epsilon=1e-5,
                    gamma_initializer=tf.random_normal_initializer(1.0, 0.02)) (x)
        if dropout > 0.0:
            x = Dropout(dropout) (x)
        return x

    img_A = Input(input_shape)
    e1 = conv_block(img_A, ngf)
    e2 = conv_block(e1, ngf * 2, lrelu=True, bn=True)
    e3 = conv_block(e2, ngf * 4, lrelu=True, bn=True)
    e4 = conv_block(e3, ngf * 8, lrelu=True, bn=True)
    e5 = conv_block(e4, ngf * 8, lrelu=True, bn=True)
    e6 = conv_block(e5, ngf * 8, lrelu=True, bn=True)
    e7 = conv_block(e6, ngf * 8, lrelu=True, bn=True)

    d7 = deconv_block(e7, ngf * 8, bn=True, dropout=0.5)
    d6 = deconv_block(d7, ngf * 8, skip_input=e6, bn=True, dropout=0.5)
    d5 = deconv_block(d6, ngf * 8, skip_input=e5, bn=True)
    d4 = deconv_block(d5, ngf * 4, skip_input=e4, bn=True)
    d3 = deconv_block(d4, ngf * 2, skip_input=e3, bn=True)
    d2 = deconv_block(d3, ngf, skip_input=e2, bn=True)
    d1 = deconv_block(d2, 1, skip_input=e1)
    out_img = Activation('tanh') (d1)
    
    model = Model(img_A, out_img)
    return model

###############################################################
#就規定好shape並且傳給G跟D來創構Model
###############################################################
input_shape=(channels,img_row,img_col)
crop_shape=(img_row,img_col)
G = gen(input_shape)
D = dis(input_shape)

###############################################################
# 定義訓練D的模型
###############################################################
D_optimizer = SGD(lr=0.0001)
D.compile(loss='binary_crossentropy', optimizer=D_optimizer,metrics=['accuracy'])

###############################################################
# 定義訓練G的模型
###############################################################
AM_optimizer = Adam(lr=0.0002, beta_1=0.5)
img_A = Input(input_shape)
img_B = Input(input_shape)
fake_A = G(img_B)
D.trainable=False
valid = D([fake_A,img_B])
AM = Model([img_A,img_B],[valid,fake_A])
AM.compile(loss=['binary_crossentropy','mae'], loss_weights=[1, 100], optimizer=AM_optimizer)

###############################################################
# Define Image Generator
###############################################################
def generator_training_Img(real_list_dir,white_list_dir,resize=None,batch_size=32):
    batch_real_img=[]
    batch_white_img=[]
    for _ in range(batch_size):
        random_index = int(np.random.randint(len(real_list_dir),size=1))
        real_img = imread(real_list_dir[random_index],mode='L')
        white_img = imread(white_list_dir[random_index],mode='L')
        if resize:
            real_img = imresize(real_img,resize)
            white_img = imresize(white_img,resize)
        batch_real_img.append(real_img)
        batch_white_img.append(white_img)
    batch_real_img = np.array(batch_real_img)/127.5-1
    batch_real_img = np.expand_dims(batch_real_img,axis=1)
    batch_white_img = np.array(batch_white_img)/127.5-1
    batch_white_img = np.expand_dims(batch_white_img,axis=1)
    return batch_real_img,batch_white_img

###############################################################
# Training Phase
###############################################################
batch_size=32
all_epoch=11000
total_epoch=11000
D_losses, G_losses = [], []

valid = np.ones((batch_size, 1, 14, 14))
fake  = np.zeros((batch_size, 1, 14, 14))
  
start_time=datetime.datetime.now()
for now_iter in range(total_epoch-all_epoch+1, total_epoch+1):
    ori_img,white_img = generator_training_Img(real_list_dir=train_real_data_list,
                                               white_list_dir=train_white_data_list,
                                               resize=(img_row,img_col),
                                               batch_size=32)
    
    ###################################
    #Training Discriminator Phase
    ###################################
    fake_A = G.predict(white_img)
    
    D_loss_Real = D.train_on_batch([ori_img,white_img],valid)
    D_loss_Fake = D.train_on_batch([fake_A,white_img],fake)
    D_loss = 0.5 * np.add(D_loss_Real, D_loss_Fake)
    D_losses.append(D_loss[0])
    
    ###################################
    #Training Generator Phase
    ###################################
    G_loss = AM.train_on_batch([ori_img, white_img],[valid, ori_img])
    G_losses.append(G_loss[0])

    end_time = datetime.datetime.now() - start_time
    print("[Epoch %d/%d] [D loss: %f, acc: %3d%%] [G loss1: %f, loss2: %f] [time:%s]" \
        % (now_iter, total_epoch, D_loss[0],D_loss[1]*100,G_loss[0],G_loss[1],end_time))

###############################################################
# Display結果
###############################################################
plt.gray()
n = 5
r,c=(n,3)
plt.figure(figsize=(c*6,r*6))
for i in range(r):
    ori_img,white_img = generator_training_Img(real_list_dir=train_real_data_list,
                                      white_list_dir=train_white_data_list,
                                      resize=(img_row,img_col),
                                      batch_size=1)#batch_size)
    ax = plt.subplot(r, c, i*c + 1)
    a = G.predict(white_img).reshape(img_row,img_col)
    a = (a+1)/2
    
    plt.imshow(a)
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    ax = plt.subplot(r, c, i*c + 2)
    a = ori_img.reshape(img_row,img_col)
    plt.imshow(a)
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    ax = plt.subplot(r, c, i*c + 3)
    a = white_img.reshape(img_row,img_col)
    plt.imshow(a)
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)   
plt.show()

#####################################
#畫出epoch與loss的折線圖
#####################################
plt.style.use('seaborn')
plt.style.use('seaborn-poster')
fig, ax = plt.subplots()
ax.plot(range(1, len(D_losses)+1), D_losses, 'r', label='Discriminative Loss')
ax.plot(range(1, len(G_losses)+1), G_losses, 'g', label='Generative Loss')
ax.set_xlabel("Epochs")
ax.set_ylabel("Loss")
ax.legend(loc='upper right', frameon=False)
plt.show()

#####################################
#生成測試集預測值的CSV檔
#####################################
def generator_test_Img(list_dir,resize):
    output_training_img=[]
    for i in list_dir:
        img = imread(i,mode='L')
        img = imresize(img,resize)
        output_training_img.append(img)
    output_training_img = np.array(output_training_img)/127.5-1
    output_training_img = np.expand_dims(output_training_img,axis=1) # (batch,img_row,img_col) ==> (batch,1,img_row,img_cok)
    return output_training_img

def numpy_to_csv(input_image,image_number=10,save_csv_name='predict.csv'):
    save_image=np.zeros([int(input_image.size/image_number),image_number],dtype=np.float32)

    for image_index in range(image_number):
        save_image[:,image_index]=input_image[image_index,:,:].flatten()

    base_word='id'
    df = pd.DataFrame(save_image)
    index_col=[]
    for i in range(n):
        col_word=base_word+str(i)
        index_col.append(col_word)
    df.index.name='index'
    df.columns=index_col
    df.to_csv(save_csv_name)
    print("Okay! numpy_to_csv")

test_data_dir='test/White/'
# test_data_dir_list=glob.glob(test_data_dir)
# test_data_list=[]
# test_data_list.extend(test_data_dir_list)
test_data_list = ['test/White/'+dirname for dirname in os.listdir(test_data_dir)]

n=10
output_img_col = output_img_row=128
white_img = generator_test_Img(list_dir=test_data_list,resize=(output_img_col,output_img_row))

image_array = G.predict(white_img).squeeze(1)
image_array = (image_array+1)/2
print(image_array.shape)

numpy_to_csv(input_image=image_array,image_number=n,save_csv_name='Prediction.csv')