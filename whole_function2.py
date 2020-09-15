
# coding: utf-8

# In[ ]:


import os
import sys
import random
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#from tqdm.tqdm import tqdm #windows
from tqdm import tqdm
from itertools import chain
import skimage
from skimage import io, transform, morphology
from skimage.io import imread, imshow, imread_collection, concatenate_images
from skimage.transform import resize
from skimage.morphology import label
import skimage.transform as trans

from keras.models import Model, load_model
from keras.layers import Input ,Dropout
from keras.layers.core import Lambda
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.pooling import MaxPooling2D
from keras.layers.merge import concatenate
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import backend as K
import tensorflow as tf
from numpy import *
from PIL import Image
import glob
from keras.layers import *#這行是關鍵(解決zero concentrate問題
from keras import optimizers
import cv2
from skimage import img_as_ubyte
import imutils
import pydicom
import pylab
from skimage import img_as_ubyte
from PIL import Image
import PIL.ImageOps    
#Global
#global box


# In[ ]:


#coding: utf-8
def distance(p0, p1):
    return math.sqrt(((p0[0] - p1[0])**2) + ((p0[1] - p1[1])**2))
def midpoint(p0, p1):
    x = (p0[0]+p1[0])/2
    y = (p0[1]+p1[1])/2
    
    return int(x),int(y)

def Area_and_length(im_path):
    im = cv2.imread(im_path)
    #im = cv2.resize(im,(350,350),interpolation=cv2.INTER_CUBIC)
    imgray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
 
    #imgray = np.array(Image.open('photo.jpg'))
    ret,thresh = cv2.threshold(imgray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)  
    aa,contours,hierarchy = cv2.findContours(thresh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)  #cv2.RETR_EXTERNAL 定义只检测外围轮廓

    cnts = contours #if imutils.is_cv2() else contours[1]  #用imutils来判断是opencv是2还是2+

    for cnt in cnts:

        x, y, w, h = cv2.boundingRect(cnt)

         # 最小外接矩形框，有方向角
        rect = cv2.minAreaRect(cnt)
        box = cv2.cv.Boxpoints() if imutils.is_cv2()else cv2.boxPoints(rect)
        box = np.int0(box)
        cv2.drawContours(im, [box], 0, (255, 0, 0), 5)

    #正方形的四個點
    A = box[0]
    B = box[1]
    C = box[2]
    D = box[3]
    
    #正方形的長寬
    length = int(distance(A,B))
    width = int(distance(B,C))
    #正方形的四邊中點座標（EFGH)
    midpoint1_1,midpoint1_2 = midpoint(A,B)
    midpoint2_1,midpoint2_2 = midpoint(B,C)
    midpoint3_1,midpoint3_2 = midpoint(C,D)
    midpoint4_1,midpoint4_2 = midpoint(D,A)

    E = (midpoint1_1,midpoint1_2)
    F = (midpoint2_1,midpoint2_2)
    G = (midpoint3_1,midpoint3_2)
    H = (midpoint4_1,midpoint4_2)
    #中點連線
    cv2.line(im,E,G,(0, 0, 255),5)
    cv2.line(im,F,H,(0, 0, 255),5)

    #圈出中點
    #cv2.circle(im,(midpoint1_1,midpoint1_2), 15, (255, 0, 0), 2)#-1 = 實心
    #cv2.circle(im,(midpoint2_1,midpoint2_2), 15, (255, 0, 0), 2)
    #cv2.circle(im,(midpoint3_1,midpoint3_2), 15, (255, 0, 0), 2)
    #cv2.circle(im,(midpoint4_1,midpoint4_2), 15, (255, 0, 0), 2)
    
    #橢圓形的面積pi*a*b(長短邊半徑)
    length =  length/96*2.54
    width = width/96*2.54
    
    area = round(math.pi*(length/2)*(width/2),2)
    
    #印出結果
    #print("面積：",area)
    #print("長：",round(length,2))
    #print("寬：",round(width,2))
    plt.imshow(im)
    plt.show()

    return area, length, width

def LVM(A4C_EDV,PSAX_EDV_big,PSAX_EDV_small):
    A4C_area, A4C_length, A4C_width = Area_and_length(A4C_EDV)
    PSAX_big_area, PSAX_big_length, PSAX_big_width = Area_and_length(PSAX_EDV_big)
    PSAX_small_area, PSAX_small_length, PSAX_small_width = Area_and_length(PSAX_EDV_small)
    
    t =  int(math.sqrt(PSAX_big_area/math.pi)-(A4C_width/2))
    LVM = int(1.05*(((5/6)*PSAX_big_area*(A4C_length+t))-((5/6)*PSAX_small_area*A4C_length)))
    print("LVM:",LVM)
    
    return LVM
#https://blog.csdn.net/dcrmg/article/details/89927816


# In[ ]:


def Area_and_length2(ori_im,pre_im,n):
    global length_new
    global width_new
    global box2
    #im = preds_train[4]
    implt = (pre_im * 255).astype(np.uint8)
    implt = np.asarray(implt)
    implt = implt[:, :, -1]
    im2 = cv2.resize(implt, (350, 350), interpolation=cv2.INTER_CUBIC)
    imRGB =cv2.cvtColor(im2,cv2.COLOR_GRAY2RGB)

    implt_ori = (ori_im * 255).astype(np.uint8)
    implt_ori = np.asarray(implt_ori)
    implt_ori = implt_ori[:, :, -1]
    im2_ori = cv2.resize(implt_ori, (350, 350), interpolation=cv2.INTER_CUBIC)
    imRGB_ori =cv2.cvtColor(im2_ori,cv2.COLOR_GRAY2RGB)
    
    #靘菔?
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(11, 11))
    im2 = cv2.erode(im2,kernel)
    imRGB = cv2.erode(imRGB,kernel)

    #imgray = np.array(Image.open('photo.jpg'))
    ret,thresh = cv2.threshold(im2,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)  
    aa,contours,hierarchy = cv2.findContours(thresh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)  #cv2.RETR_EXTERNAL 定义只检测外围轮廓

    cnts = contours #if imutils.is_cv2() else contours[1]  #用imutils来判断是opencv是2还是2+

    cou=[]
    if len(cnts)>1 :
        for cnt in cnts:
            cou.append(cnt.shape[0])

    # print(cou)
    # print(cou.index(max(cou)))

        for cnt in cnts:
    
            if int(cnt.shape[0]) ==max(cou):

                x, y, w, h = cv2.boundingRect(cnt)

                # 最小外接矩形框，有方向角
                rect = cv2.minAreaRect(cnt)
                box2 = cv2.cv.Boxpoints() if imutils.is_cv2()else cv2.boxPoints(rect)
                box2 = np.int0(box2)
                cv2.drawContours(imRGB, [box2], 0, (255, 0, 0), 4)
                cv2.drawContours(imRGB, [box2], 0, (255, 0, 0), 4)
    else:
        for cnt in cnts:
            x, y, w, h = cv2.boundingRect(cnt)

            # 最小外接矩形框，有方向角
            rect = cv2.minAreaRect(cnt)
            box2 = cv2.cv.Boxpoints() if imutils.is_cv2()else cv2.boxPoints(rect)
            box2 = np.int0(box2)
            cv2.drawContours(imRGB, [box2], 0, (255, 0, 0), 4)

    #正方形的四個點
    A = box2[0]
    B = box2[1]
    C = box2[2]
    D = box2[3]

    #正方形的長寬
    length = int(distance(A,B))
    width = int(distance(B,C))
    #正方形的四邊中點座標（EFGH)
    midpoint1_1,midpoint1_2 = midpoint(A,B)
    midpoint2_1,midpoint2_2 = midpoint(B,C)
    midpoint3_1,midpoint3_2 = midpoint(C,D)
    midpoint4_1,midpoint4_2 = midpoint(D,A)

    E = (midpoint1_1,midpoint1_2)
    F = (midpoint2_1,midpoint2_2)
    G = (midpoint3_1,midpoint3_2)
    H = (midpoint4_1,midpoint4_2)
    #中點連線
    cv2.line(imRGB,E,G,(0, 0, 255),4)
    cv2.line(imRGB,F,H,(0, 0, 255),4)

    #圈出中點
    #cv2.circle(im,(midpoint1_1,midpoint1_2), 15, (255, 0, 0), 2)#-1 = 實心
    #cv2.circle(im,(midpoint2_1,midpoint2_2), 15, (255, 0, 0), 2)
    #cv2.circle(im,(midpoint3_1,midpoint3_2), 15, (255, 0, 0), 2)
    #cv2.circle(im,(midpoint4_1,midpoint4_2), 15, (255, 0, 0), 2)
    #print("pixel長：",round(length,2))
    #print("pixel寬：",round(width,2))
    #橢圓形的面積pi*a*b(長短邊半徑)
    length =  length/96*2.54
    width = width/96*2.54

    area = round(math.pi*(length/2)*(width/2),2)

    #印出結果
    #print("面積：",area)
    #print("長：",round(length,2))
    #print("寬：",round(width,2))
    #plt.imshow(imRGB)
    #plt.show()
    
    if n =="A4C_LV" or n =="A2C_LV":
        if length > width:
            length_new= width
            width_new =length
        else:
            length_new = length
            width_new = width
    else:
            length_new = length
            width_new = width
    length_new = round(length_new,2)
    width_new = round(width_new,2)
    #print("change面積：",area)
    #print("change長：",round(length_new,2))
    #print("change寬：",round(width_new,2))

    return imRGB_ori,imRGB, area, length_new, width_new


# In[ ]:


# 設定參數
IMG_WIDTH =  128
IMG_HEIGHT = 128
IMG_CHANNELS = 1
def mean_iou(y_true, y_pred):
    prec = []
    for t in np.arange(0.5, 1.0, 0.05):
        y_pred_ = tf.to_int32(y_pred > t)
        score, up_opt = tf.metrics.mean_iou(y_true, y_pred_, 2)
        K.get_session().run(tf.local_variables_initializer())
        with tf.control_dependencies([up_opt]):
            score = tf.identity(score)
        prec.append(score)
    return K.mean(K.stack(prec), axis=0)
def get_unet():
    inputs = Input((IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))
    #s = Lambda(lambda x: x / 255) (inputs)

    c1 = Conv2D(8, (3, 3), activation='relu', padding='same') (inputs)#(s)
    c1 = Conv2D(8, (3, 3), activation='relu', padding='same') (c1)
    p1 = MaxPooling2D((2, 2)) (c1)

    c2 = Conv2D(16, (3, 3), activation='relu', padding='same') (p1)
    c2 = Conv2D(16, (3, 3), activation='relu', padding='same') (c2)
    p2 = MaxPooling2D((2, 2)) (c2)

    c3 = Conv2D(32, (3, 3), activation='relu', padding='same') (p2)
    c3 = Conv2D(32, (3, 3), activation='relu', padding='same') (c3)
    p3 = MaxPooling2D((2, 2)) (c3)

    c4 = Conv2D(64, (3, 3), activation='relu', padding='same') (p3)
    c4 = Conv2D(64, (3, 3), activation='relu', padding='same') (c4)
    p4 = MaxPooling2D(pool_size=(2, 2)) (c4)

    c5 = Conv2D(128, (3, 3), activation='relu', padding='same') (p4)
    c5 = Conv2D(128, (3, 3), activation='relu', padding='same') (c5)

    u6 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same') (c5)
    u6 = concatenate([u6, c4])
    c6 = Conv2D(64, (3, 3), activation='relu', padding='same') (u6)
    c6 = Conv2D(64, (3, 3), activation='relu', padding='same') (c6)

    u7 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same') (c6)
    u7 = concatenate([u7, c3])
    c7 = Conv2D(32, (3, 3), activation='relu', padding='same') (u7)
    c7 = Conv2D(32, (3, 3), activation='relu', padding='same') (c7)

    u8 = Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same') (c7)
    u8 = concatenate([u8, c2])
    c8 = Conv2D(16, (3, 3), activation='relu', padding='same') (u8)
    c8 = Conv2D(16, (3, 3), activation='relu', padding='same') (c8)

    u9 = Conv2DTranspose(8, (2, 2), strides=(2, 2), padding='same') (c8)
    u9 = concatenate([u9, c1], axis=3)
    c9 = Conv2D(8, (3, 3), activation='relu', padding='same') (u9)
    c9 = Conv2D(8, (3, 3), activation='relu', padding='same') (c9)

    outputs = Conv2D(1, (1, 1), activation='sigmoid') (c9)

    model = Model(inputs=[inputs], outputs=[outputs])
    ADAM = optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    model.compile(optimizer=ADAM, loss='binary_crossentropy', metrics=[mean_iou])
    #model.summary()#'adam'
    return model
smooth = 1.
model = get_unet()
#model2 = get_unet()


# In[ ]:


def addTransparency(img, factor = 0.7 ):
    img = img.convert('RGBA')
    img_blender = Image.new('RGBA', img.size, (0,0,0,0))
    img = Image.blend(img_blender, img, factor)
    return img


# In[ ]:


def Binary(img):
    #img = Image.open( r"F:\馬交計畫\code\20191003_Label_for_traning\Training _data\A4C\A4C_LV\9ADBK6OO_52.jpg")
    Lim = img.convert('L')
    threshold = 150
 
    table = []
    for i in range(256):
        if i < threshold:
            table.append(0)
        else:
            table.append(1)

    photo = Lim.point(table, '1')
    return photo
    #plt.imshow(photo)
    #plt.show()


# In[ ]:


def Transparent(ori_image,img,n):
#img = Image.open( r"F:\馬交計畫\code\20191003_Label_for_traning\Training _data\A4C\A4C_LV\9ADBK6OO_52.jpg")
    #width, height = img.size
    width, height, channel = img.shape
    img = img_as_ubyte(img)
    img = img[:, :, -1]
    img = Image.fromarray(img)
    img =Binary(img)
    img = img.convert('RGBA')
    ori_pre_image = img.copy()
    
    ori_image = img_as_ubyte(ori_image)
    ori_image = ori_image[:, :, -1]
    ori_image = Image.fromarray(ori_image)
    ori_image = ori_image.convert('RGBA')
    for x in range(width):
        for y in range(height):
            R,G,B,alpha = img.getpixel((x,y))
            #concentrate = 100
            
            if R<50 or B<50 or G<50:
                current_color = img.getpixel((x,y))
                new_color = (0,0,0,0)
                img.putpixel( (x,y), new_color)
            if n == "A4C_LV":
                if R>100 or B>100 or G>100:
                    current_color = img.getpixel((x,y))
                    new_color = (255,0,0,30)#RED調淡
                    img.putpixel( (x,y), new_color)
            if n == "A4C_LA":
                if R>100 or B>100 or G>100:
                    current_color = img.getpixel((x,y))
                    new_color = (255,255,0,30)#Yellow調淡
                    img.putpixel( (x,y), new_color)
            if n == "A2C_LV":
                if R>100 or B>100 or G>100:
                    current_color = img.getpixel((x,y))
                    new_color = (0,255,0,30)#Green調淡
                    img.putpixel( (x,y), new_color)
            if n == "A2C_LA":
                if R>100 or B>100 or G>100:
                    current_color = img.getpixel((x,y))
                    new_color = (0,255,255,30)#BlUE調淡
                    img.putpixel( (x,y), new_color)    
            if n == "PSAX_big":
                if R>100 or B>100 or G>100:
                    current_color = img.getpixel((x,y))
                    new_color = (0,0,255,30)#red #Deep BlUE調淡(0,0,255,50)
                    img.putpixel( (x,y), new_color) 
            if n == "PSAX_small":
                if R>100 or B>100 or G>100:
                    current_color = img.getpixel((x,y))
                    new_color = (255,125,125,30)#light pink調淡
                    img.putpixel( (x,y), new_color) 
    #ori_image = Image.fromarray(ori_image)
    #img = Image.fromarray(img)
    return ori_image,ori_pre_image,img




# In[ ]:


#Read model
#K.clear_session()#清空剛剛跑得model下面load model才load的進來
def loading_model(a):
    #os.chdir("/media/linlab/Seagate Backup Plus Drive/馬交計畫/code/20191007_code_summary/Previous_code/model")
    if(a==1):
        os.chdir(r"F:\馬交計畫_洪\code\20191007_code_summary\Previous_code\model")
        print("Loading A4C model.....")
        model1 = load_model('20190816_LV.h5', custom_objects={'mean_iou': mean_iou})#20190816_LV_2.h5
        model2 = load_model('20180807model-A4C-lower.h5', custom_objects={'mean_iou': mean_iou})
        print("A4C model loading finish!")
        print("Loading A2C model.....")
        model3 = load_model('20181012 model-A2C-upper.h5', custom_objects={'mean_iou': mean_iou})
        model4 = load_model('20181012 model-A2C-lower.h5', custom_objects={'mean_iou': mean_iou})#20190816_LV_2.h5
        print("A2C model loading finish!")
        print("Loading PSAX model.....")
        model5 = load_model('20190814_PSAX_big.h5', custom_objects={'mean_iou': mean_iou})
        model6 = load_model('20190814_PSAX_small.h5', custom_objects={'mean_iou': mean_iou})
        print("PSAX model loading finish!")
    
    elif(a==2):
        os.chdir(r"F:\馬交計畫_洪\code\20191030_Unet_250\model")
        print("Loading A4C model.....")
        model1 = load_model('20191030_A4C_LV_250.h5', custom_objects={'mean_iou': mean_iou})#20190816_LV_2.h5
        model2 = load_model('20191030_A4C_LA_250.h5', custom_objects={'mean_iou': mean_iou})
        print("A4C model loading finish!")
        print("Loading A2C model.....")
        model3 = load_model('20191030_A2C_LA_250.h5', custom_objects={'mean_iou': mean_iou})
        model4 = load_model('20191030_A2C_LV_250.h5', custom_objects={'mean_iou': mean_iou})#20190816_LV_2.h5
        print("A2C model loading finish!")
        print("Loading PSAX model.....")
        model5 = load_model('20191030_PSAX_big_250.h5', custom_objects={'mean_iou': mean_iou})
        model6 = load_model('20191030_PSAX_small_250.h5', custom_objects={'mean_iou': mean_iou})
        print("PSAX model loading finish!")
        
    elif(a==3):
        os.chdir(r"F:\馬交計畫_洪\code\20191030_Unet_250\model")
        print("Loading A4C model.....")
        model1 = load_model('20191115_A4C_LV_500.h5', custom_objects={'mean_iou': mean_iou})#20190816_LV_2.h5
        model2 = load_model('20191115_A4C_LA_500.h5', custom_objects={'mean_iou': mean_iou})
        print("A4C model loading finish!")
        print("Loading A2C model.....")
        model3 = load_model('20191115_A2C_LV_500.h5', custom_objects={'mean_iou': mean_iou})
        model4 = load_model('20191115_A2C_LA_500.h5', custom_objects={'mean_iou': mean_iou})#20190816_LV_2.h5
        print("A2C model loading finish!")
        print("Loading PSAX model.....")
        model5 = load_model('20191115_PSAX_BIG_500.h5', custom_objects={'mean_iou': mean_iou})
        model6 = load_model('20191115_PSAX_SMALL_500.h5', custom_objects={'mean_iou': mean_iou})
        print("PSAX model loading finish!")

    elif(a==4):
        os.chdir(r"F:\馬交計畫_洪\code\20191030_Unet_250\model")
        print("Loading A4C model.....")
        model1 = load_model('20191115_A4C_LV_500.h5', custom_objects={'mean_iou': mean_iou})#20190816_LV_2.h5
        model2 = load_model('20191115_A4C_LA_500.h5', custom_objects={'mean_iou': mean_iou})
        print("A4C model loading finish!")
        print("Loading A2C model.....")
        model3 = load_model('20191115_A2C_LV_500.h5', custom_objects={'mean_iou': mean_iou})
        model4 = load_model('20191115_A2C_LA_500.h5', custom_objects={'mean_iou': mean_iou})#20190816_LV_2.h5
        print("A2C model loading finish!")
        print("Loading PSAX model.....")
        model5 = load_model('20191030_PSAX_big_250.h5', custom_objects={'mean_iou': mean_iou})#difference to 3 only PSAXmodel
        model6 = load_model('20191030_PSAX_small_250.h5', custom_objects={'mean_iou': mean_iou})
        print("PSAX model loading finish!")

    return model1,model2,model3,model4,model5,model6

def loading_model_linux(a):
    if(a==1):
        os.chdir("/media/linlab/Seagate Backup Plus Drive/馬交計畫_洪/code/20191007_code_summary/Previous_code/model")
        #os.chdir(r"F:\馬交計畫\code\20191007_code_summary\Previous_code\model")
        print("Loading A4C model.....")
        model1 = load_model('20190816_LV.h5', custom_objects={'mean_iou': mean_iou})#20190816_LV_2.h5
        model2 = load_model('20180807model-A4C-lower.h5', custom_objects={'mean_iou': mean_iou})
        print("A4C model loading finish!")
        print("Loading A2C model.....")
        model3 = load_model('20181012 model-A2C-upper.h5', custom_objects={'mean_iou': mean_iou})
        model4 = load_model('20181012 model-A2C-lower.h5', custom_objects={'mean_iou': mean_iou})#20190816_LV_2.h5
        print("A2C model loading finish!")
        print("Loading PSAX model.....")
        model5 = load_model('20190814_PSAX_big.h5', custom_objects={'mean_iou': mean_iou})
        model6 = load_model('20190814_PSAX_small.h5', custom_objects={'mean_iou': mean_iou})
        print("PSAX model loading finish!")
    
    elif(a==2):
        os.chdir("/media/linlab/Seagate Backup Plus Drive/馬交計畫_洪/code/20191030_Unet_250/model")
        print("Loading A4C model.....")
        model1 = load_model('20191030_A4C_LV_250.h5', custom_objects={'mean_iou': mean_iou})#20190816_LV_2.h5
        model2 = load_model('20191030_A4C_LA_250.h5', custom_objects={'mean_iou': mean_iou})
        print("A4C model loading finish!")
        print("Loading A2C model.....")
        model3 = load_model('20191030_A2C_LA_250.h5', custom_objects={'mean_iou': mean_iou})
        model4 = load_model('20191030_A2C_LV_250.h5', custom_objects={'mean_iou': mean_iou})#20190816_LV_2.h5
        print("A2C model loading finish!")
        print("Loading PSAX model.....")
        model5 = load_model('20191030_PSAX_big_250.h5', custom_objects={'mean_iou': mean_iou})
        model6 = load_model('20191030_PSAX_small_250.h5', custom_objects={'mean_iou': mean_iou})
        print("PSAX model loading finish!")
        
    elif(a==3):
        os.chdir("/media/linlab/Seagate Backup Plus Drive/馬交計畫_洪/code/20191030_Unet_250/model")
        print("Loading A4C model.....")
        model1 = load_model('20191115_A4C_LV_500.h5', custom_objects={'mean_iou': mean_iou})#20190816_LV_2.h5
        model2 = load_model('20191115_A4C_LA_500.h5', custom_objects={'mean_iou': mean_iou})
        print("A4C model loading finish!")
        print("Loading A2C model.....")
        model3 = load_model('20191115_A2C_LV_500.h5', custom_objects={'mean_iou': mean_iou})
        model4 = load_model('20191115_A2C_LA_500.h5', custom_objects={'mean_iou': mean_iou})#20190816_LV_2.h5
        print("A2C model loading finish!")
        print("Loading PSAX model.....")
        model5 = load_model('20191115_PSAX_BIG_500.h5', custom_objects={'mean_iou': mean_iou})
        model6 = load_model('20191115_PSAX_SMALL_500.h5', custom_objects={'mean_iou': mean_iou})
        print("PSAX model loading finish!")

    elif(a==4):
        os.chdir("/media/linlab/Seagate Backup Plus Drive/馬交計畫_洪/code/20191030_Unet_250/model")
        print("Loading A4C model.....")
        model1 = load_model('20191115_A4C_LV_500.h5', custom_objects={'mean_iou': mean_iou})#20190816_LV_2.h5
        model2 = load_model('20191115_A4C_LA_500.h5', custom_objects={'mean_iou': mean_iou})
        print("A4C model loading finish!")
        print("Loading A2C model.....")
        model3 = load_model('20191115_A2C_LV_500.h5', custom_objects={'mean_iou': mean_iou})
        model4 = load_model('20191115_A2C_LA_500.h5', custom_objects={'mean_iou': mean_iou})#20190816_LV_2.h5
        print("A2C model loading finish!")
        print("Loading PSAX model.....")
        model5 = load_model('20191030_PSAX_big_250.h5', custom_objects={'mean_iou': mean_iou})
        model6 = load_model('20191030_PSAX_small_250.h5', custom_objects={'mean_iou': mean_iou})
        print("PSAX model loading finish!")
        
    elif(a==5):
        os.chdir("/media/linlab/Seagate Backup Plus Drive/馬交計畫_洪/code/20191030_Unet_250/model")
        print("Loading A4C model.....")
        model1 = load_model('20191115_A4C_LV_500.h5', custom_objects={'mean_iou': mean_iou})#20190816_LV_2.h5
        model2 = load_model('20191115_A4C_LA_500.h5', custom_objects={'mean_iou': mean_iou})
        print("A4C model loading finish!")
        print("Loading A2C model.....")
        model3 = load_model('20191115_A2C_LV_500.h5', custom_objects={'mean_iou': mean_iou})
        model4 = load_model('20191115_A2C_LA_500.h5', custom_objects={'mean_iou': mean_iou})#20190816_LV_2.h5
        print("A2C model loading finish!")
        print("Loading PSAX model.....")
        model5 = load_model('20200323_PSAX_BIG_600.h5', custom_objects={'mean_iou': mean_iou})#20200322_PSAX_BIG_500.h5
        model6 = load_model('20200323_PSAX_SMALL_600.h5', custom_objects={'mean_iou': mean_iou})#20200322_PSAX_SMALL_500.h5
        print("PSAX model loading finish!")
    return model1,model2,model3,model4,model5,model6
def A4C_LV_loading_model():
    #os.chdir("/media/linlab/Seagate Backup Plus Drive/馬交計畫/code/20191007_code_summary/Previous_code/model")
    os.chdir(r"F:\馬交計畫\code\20191007_code_summary\Previous_code\model")
    model1 = load_model('20190816_LV.h5', custom_objects={'mean_iou': mean_iou})#20190816_LV_2.h5
    model2 = load_model('20180807model-A4C-lower.h5', custom_objects={'mean_iou': mean_iou})
    #model3 = load_model('20181012 model-A2C-upper.h5', custom_objects={'mean_iou': mean_iou})
    #model4 = load_model('20181012 model-A2C-lower.h5', custom_objects={'mean_iou': mean_iou})#20190816_LV_2.h5
    #model5 = load_model('20190814_PSAX_big.h5', custom_objects={'mean_iou': mean_iou})
    #model6 = load_model('20190814_PSAX_small.h5', custom_objects={'mean_iou': mean_iou})
    return model1,model2#,model3,model4,model5,model6
def PSAX_loading_model():
    #os.chdir("/media/linlab/Seagate Backup Plus Drive/馬交計畫/code/20191007_code_summary/Previous_code/model")
    os.chdir(r"F:\馬交計畫\code\20191007_code_summary\Previous_code\model")
    #model1 = load_model('20190816_LV.h5', custom_objects={'mean_iou': mean_iou})#20190816_LV_2.h5
    #model2 = load_model('20180807model-A4C-lower.h5', custom_objects={'mean_iou': mean_iou})
    #model3 = load_model('20181012 model-A2C-upper.h5', custom_objects={'mean_iou': mean_iou})
    #model4 = load_model('20181012 model-A2C-lower.h5', custom_objects={'mean_iou': mean_iou})#20190816_LV_2.h5
    model5 = load_model('20190814_PSAX_big.h5', custom_objects={'mean_iou': mean_iou})
    model6 = load_model('20190814_PSAX_small.h5', custom_objects={'mean_iou': mean_iou})
    return model5,model6#,model3,model4,model5,model6
def A2C_loading_model():
    #os.chdir("/media/linlab/Seagate Backup Plus Drive/馬交計畫/code/20191007_code_summary/Previous_code/model")
    os.chdir(r"F:\馬交計畫\code\20191007_code_summary\Previous_code\model")
    #model1 = load_model('20190816_LV.h5', custom_objects={'mean_iou': mean_iou})#20190816_LV_2.h5
    #model2 = load_model('20180807model-A4C-lower.h5', custom_objects={'mean_iou': mean_iou})
    model3 = load_model('20181012 model-A2C-upper.h5', custom_objects={'mean_iou': mean_iou})
    model4 = load_model('20181012 model-A2C-lower.h5', custom_objects={'mean_iou': mean_iou})#20190816_LV_2.h5
    #model5 = load_model('20190814_PSAX_big.h5', custom_objects={'mean_iou': mean_iou})
    #model6 = load_model('20190814_PSAX_small.h5', custom_objects={'mean_iou': mean_iou})
    return model3,model4#,model3,model4,model5,model6


# In[ ]:


def single_dicom(path):
    ds = pydicom.dcmread(path,force=True)
    ds.PixelRepresentation = 0
    #ds.dir()
    return ds.pixel_array
    #number = ds.pixel_array.shape[0]


# In[ ]:

#def imgsize_detect(path):
#    path = Image.open(path)
#    #path = Image.open("/media/linlab/Seagate Backup Plus Drive/馬交計畫/code/20191003_Label_for_traning/Training #_data/Patient/A4C/A4C_image/7ULMRT2X_50.jpg")
#    path2 = np.asarray(path)
#    if len(path2.shape) == 3:
#       if path2.shape[0]== 434:#region = (160,50,460,350)
#            region = (160,50,460,350)#(143,42,493,392)#350x350
#           path = path.crop(region) 
#            path = np.asarray(path)
#        if  path2.shape[0]== 422:
#            region = (100,35,500,435)#(143,42,493,392)#350x350
#            path = path.crop(region) 
#            path = np.asarray(path)
#            #print(path.shape)
#       if path2.shape[0] ==600:   
#            region = (225,145,625,545)#(143,42,493,392)#350x350
#            path = path.crop(region) 
#            path = np.asarray(path)
#            #print(path.shape)
#    elif len(path2.shape) == 2:
#        region = (260,110,660,510)#(143,42,493,392)#350x350
#        path = path.crop(region) 
#        path = np.asarray(path)
#        #print(len(c.shape))
#        print(path.shape)
#        path = np.atleast_3d(path)
#        path = np.asarray(path)
#        path = cv2.cvtColor(path, cv2.COLOR_GRAY2BGR)
#    else:
#        print("new shape:",path2.shape)
#    return path


def imgsize_detect(path):
    path = Image.open(path)
    #path = Image.open("/media/linlab/Seagate Backup Plus Drive/馬交計畫/code/20191003_Label_for_traning/Training _data/Patient/A4C/A4C_image/7ULMRT2X_50.jpg")
    path2 = np.asarray(path)
    print(path2.shape)
    #if len(path2.shape) == 3:
    if path2.shape[0]== 434:#region = (160,50,460,350)
        region = (160,50,460,350)#(143,42,493,392)#350x350
        path = path.crop(region) 
            #path = np.asarray(path)
    elif  path2.shape[0]== 422:
        region = (100,35,500,435)#(143,42,493,392)#350x350
        path = path.crop(region) 
            #path = np.asarray(path)
            #print(path.shape)
    elif path2.shape[0] ==600:   
        region = (225,145,625,545)#(143,42,493,392)#350x350
        path = path.crop(region) 
            #path = np.asarray(path)
            #print(path.shape)
    elif  path2.shape[0]== 480:
        region = (130,60,530,460)#(143,42,493,392)#350x350#(120,60,520,460)
        path = path.crop(region) 
    #elif len(path2.shape) == 2:
    else:
        region = (260,110,660,510)#(143,42,493,392)#350x350
        path = path.crop(region) 
        #path = np.asarray(path)
        #print(len(c.shape))
        #print(path.shape)
    path = np.atleast_3d(path)
    path = np.asarray(path)
    path = cv2.cvtColor(path, cv2.COLOR_GRAY2BGR)
    path = Image.fromarray(path)
    print("new shape:",path2.shape)
    return path

def single_image(path1):
    ###path= image_path
    #A4C
    im = imgsize_detect(path1)
    train = Image.fromarray(path1)
#    im = Image.open(path1) 
#     region = (143,42,493,392)
#     train = im.crop(region)
    train = train.convert('L') 
    train = train.resize((IMG_WIDTH,IMG_HEIGHT),Image.BILINEAR)
    train = np.asarray(train)
    train = train/255.
    #array
    return train
# def single_image(path1):
#     ###path= image_path
#     #A4C
#     im = Image.open(path1) 
#     region = (143,42,493,392)
#     train = im.crop(region)
#     train = train.convert('L') 
#     train = train.resize((IMG_WIDTH,IMG_HEIGHT),Image.BILINEAR)
#     train = np.asarray(train)
#     train = train/255.
#     #array
#     return train


# In[ ]:


def large_amount_image(path1,path2,path3):
    #Load in testing data
    #A4C
    #path1 = "/media/linlab/Seagate Backup Plus Drive/Jess/馬交計畫/code/20190807_LVMI/Data/Data_image/A4C/A4C_EDV2"
    #PSAX
    #path2 = "/media/linlab/Seagate Backup Plus Drive/Jess/馬交計畫/code/20190807_LVMI/Data/Data_image/PSAX/PSAX_EDV2"

    #A4C#path1
    os.chdir(path1) 
    files = os.listdir(path1) #path
    files.sort()
    A4C_image_list = []


    for File in files:
        im = Image.open(File)
        #print("A4C:",File)
        region = (143,42,493,392)#350x350
        train = im.crop(region) 
        train = train.convert('L') 
        train = train.resize((IMG_WIDTH,IMG_HEIGHT),Image.BILINEAR)
        train = np.asarray(train)
        train = train/255.
        A4C_image_list.append(train)

    #PSAX#path2
    os.chdir(path2) 
    files = os.listdir(path2) #path
    files.sort()
    PSAX_image_list = []

    for File in files:
        im = Image.open(File)
        #print("PSAX:",File)
        region = (143,42,493,392)#350x350
        train = im.crop(region) 
        train = train.convert('L') 
        train = train.resize((IMG_WIDTH,IMG_HEIGHT),Image.BILINEAR)
        train = np.asarray(train)
        train = train/255.
        PSAX_image_list.append(train)

    #A2C#path2
    os.chdir(path3) 
    files = os.listdir(path3) #path
    files.sort()
    A2C_image_list = []

    for File in files:
        im = Image.open(File)
        #print("A2C:",File)
        region = (143,42,493,392)#350x350
        train = im.crop(region) 
        train = train.convert('L') 
        train = train.resize((IMG_WIDTH,IMG_HEIGHT),Image.BILINEAR)
        train = np.asarray(train)
        train = train/255.
        A2C_image_list.append(train)


    #轉成陣列
    A4C_image_list= np.asarray(A4C_image_list)
    PSAX_image_list= np.asarray(PSAX_image_list)
    A2C_image_list= np.asarray(A2C_image_list)

    #print(A4C_image_list.shape)
    #print(PSAX_image_list.shape)


    A4C_image_list_4= A4C_image_list.reshape(A4C_image_list.shape[0],128,128,1).astype('float32')#增加一個維度
    PSAX_image_list_4 = PSAX_image_list.reshape(PSAX_image_list.shape[0],128,128,1).astype('float32')#增加一個維度
    A2C_image_list_4 = A2C_image_list.reshape(A2C_image_list.shape[0],128,128,1).astype('float32')#增加一個維度
    print("A4C DATA AMOUNT:",A4C_image_list_4.shape)
    print("PSAX DATA AMOUNT:",PSAX_image_list_4.shape)
    print("A2C DATA AMOUNT:",A2C_image_list_4.shape)
    return A4C_image_list_4,PSAX_image_list_4,A2C_image_list_4


# In[ ]:

def imgsize_detect_img(path):
    #path = Image.open(path)
    #path = Image.open("/media/linlab/Seagate Backup Plus Drive/馬交計畫/code/20191003_Label_for_traning/Training _data/Patient/A4C/A4C_image/7ULMRT2X_50.jpg")
    path2 = np.asarray(path)
    if len(path2.shape) == 3:
        if path2.shape[0]== 434 :#region = (160,50,460,350)
            region = (143,42,493,392)#(143,42,493,392)#350x350#(160,50,460,350)
            path = path.crop(region) 
            path = np.asarray(path)
        if path2.shape[0]== 422:
            region = (100,35,500,435)#(143,42,493,392)#350x350
            path = path.crop(region) 
            path = np.asarray(path)
            #print(path.shape)
        if path2.shape[0]== 480 or path2.shape[0]== 484:
            region = (130,65,500,465)#(143,42,493,392)#350x350
            path = path.crop(region) 
            path = np.asarray(path)
        if path2.shape[0] ==600:   
            region = (250,170,600,495)#(225,145,625,545)#(143,42,493,392)#350x350
            path = path.crop(region) 
            path = np.asarray(path)
            #print(path.shape)
    elif len(path2.shape) == 2:
        region = (285,135,635,485)#(260,110,660,510)#(143,42,493,392)#350x350
        path = path.crop(region) 
        path = np.asarray(path)
        #print(len(c.shape))
        #print(path.shape)
        path = np.atleast_3d(path)
        path = np.asarray(path)
        path = cv2.cvtColor(path, cv2.COLOR_GRAY2BGR)
    else:
        print("new shape:",path2.shape)
    return path
def dicom_image_processing(array):
    s = []
    #region = (143,42,493,392)#350x350
    print(array.shape)
    for i in range(int(array.shape[0])):
        im = Image.fromarray(array[i])
        im = imgsize_detect_img(im)
        #print(im)
        try:
            im = Image.fromarray(im)
        #cropImg = im.crop(region)
        except TypeError:
            pass
        gray = im.convert('L')
        resize = gray.resize((128,128),Image.BILINEAR)
        train = np.asarray(resize)
        train = train/255.
        s.append(train)

    s = np.asarray(s)#list 轉 array
    s4 = s.reshape(s.shape[0],128,128,1).astype('float32')#增加一個維度

    #print(s.shape)
    #print(s4.shape)

    #plt.imshow(s[0], cmap = 'gray')
    #plt.show()
    #plt.imshow(np.squeeze(s4[0]), cmap = 'gray')
    #plt.show()
    return s, s4
# def dicom_image_processing(array):
#     s = []
#     region = (143,42,493,392)#350x350
#     for i in range(int(array.shape[0])):
#         im = Image.fromarray(array[i])
#         cropImg = im.crop(region)
#         gray = cropImg.convert('L')
#         resize = gray.resize((128,128),Image.BILINEAR)
#         train = np.asarray(resize)
#         train = train/255.
#         s.append(train)

#     s = np.asarray(s)#list 轉 array
#     s4 = s.reshape(s.shape[0],128,128,1).astype('float32')#增加一個維度

#     #print(s.shape)
#     #print(s4.shape)

#     #plt.imshow(s[0], cmap = 'gray')
#     #plt.show()
#     #plt.imshow(np.squeeze(s4[0]), cmap = 'gray')
#     #plt.show()
#     return s, s4


# In[ ]:

def LV_volum(area, length):
    if area>0 and length>0:
        a = 8*(area**2)/(3* math.pi *length)
    else:
        a=0
    return round(a,2) 
# def LV_volum(area, length):
#     a = ((8*area)**2)/(3* math.pi *length)
#     return a 

def EF(v_list):
    big = max(v_list)
    small = min(v_list)
    EF_volum = ((big-small)/big)*100
    return round(EF_volum,2)


# In[ ]:


def A4C_all(A4C_path,A4C_LV_model,A4C_LA_model):
    #img_path = 
    #A4C_path = r"F:\馬交計畫\code\20191007_code_summary\Demo_dicom\96HASOTE"

    image_array = single_dicom(A4C_path)
    s,s4 = dicom_image_processing(image_array)

    #Load model
    #A4C_LV_model,A4C_LA_model,A2C_LV_model,A2C_LA_model,PSAX_big_model,PSAX_small_model = A4C_LV_loading_model()
    #A4C_LV_model,A4C_LA_model = A4C_LV_loading_model()
    preds_train_A4C_LV= A4C_LV_model.predict(s4, verbose=1)
    preds_train_A4C_LA= A4C_LA_model.predict(s4, verbose=1)
    #covex_hull get area
    A4C_LV_whole_area = []
    A4C_LV_whole_length = []
    A4C_LV_whole_width = []
    A4C_LV_whole_LV_volum = []
    A4C_LV_whole_im = []
    A4C_LV_whole_im_ori = []


    A4C_LA_whole_area = []
    A4C_LA_whole_length = []
    A4C_LA_whole_width = []
    A4C_LA_whole_LV_volum = []
    A4C_LA_whole_im = []
    A4C_LA_whole_im_ori = []
    A4C_whole_trans = []

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(5, 5))



    for a in range(preds_train_A4C_LV.shape[0]):
        #print("現在是",a)
        #c = cv2.erode(preds_train_A4C_LV[a],kernel)
        #d = cv2.erode(preds_train_A4C_LA[a],kernel)
        #######LV                    
        A4C_LV_imRGB_ori,A4C_LV_imRGB, A4C_LV_area, A4C_LV_length, A4C_LV_width = Area_and_length2(s4[a],preds_train_A4C_LV[a],"A4C_LV")
        A4C_LV_volum_one = LV_volum(A4C_LV_area, A4C_LV_length)
        A4C_LV_ori_image,A4C_LV_ori_pre_image,A4C_LV_img= Transparent(s4[a],preds_train_A4C_LV[a],"A4C_LV")
        
        A4C_LV_ori_image_color = A4C_LV_ori_image.copy()
        A4C_LV_ori_image_color.paste(A4C_LV_img,(0, 0),A4C_LV_img)
    
        A4C_LV_whole_area.append(A4C_LV_area)
        A4C_LV_whole_length.append(A4C_LV_length)
        A4C_LV_whole_width.append(A4C_LV_width)
        A4C_LV_whole_LV_volum.append(A4C_LV_volum_one)
        A4C_LV_whole_im.append(A4C_LV_imRGB)
        A4C_LV_whole_im_ori.append(A4C_LV_imRGB_ori)
        
        
        #######LA
        A4C_LA_imRGB_ori,A4C_LA_imRGB, A4C_LA_area, A4C_LA_length, A4C_LA_width = Area_and_length2(s4[a],preds_train_A4C_LA[a],"A4C_LA")
        A4C_LA_volum_one = LV_volum(A4C_LA_area, A4C_LA_length) 
        A4C_LA_ori_image,A4C_LA_ori_pre_image,A4C_LA_img= Transparent(s4[a],preds_train_A4C_LA[a],"A4C_LA")
        A4C_LV_ori_image_color.paste(A4C_LA_img,(0, 0),A4C_LA_img)
    
        A4C_LA_whole_area.append(A4C_LA_area)
        A4C_LA_whole_length.append(A4C_LA_length)
        A4C_LA_whole_width.append(A4C_LA_width)
        A4C_LA_whole_LV_volum.append(A4C_LA_volum_one)
        A4C_LA_whole_im.append(A4C_LA_imRGB)
        A4C_LA_whole_im_ori.append(A4C_LA_imRGB_ori)
    
        A4C_whole_trans.append(A4C_LV_ori_image_color)
        
        pylab.rcParams['figure.dpi'] = 150
        pylab.rcParams['figure.figsize'] = (15.0, 8.0)

    A4C_LV_big = A4C_LV_whole_area.index(max(A4C_LV_whole_area))
    A4C_LV_small = A4C_LV_whole_area.index(min(A4C_LV_whole_area))
    A4C_LA_big = A4C_LA_whole_area.index(max(A4C_LA_whole_area))
    A4C_LA_small = A4C_LA_whole_area.index(min(A4C_LA_whole_area))
    #whole_im[big]
    #whole_im[small]

    plt.figure()
    plt.subplot(1,2,1)
    plt.title('LV_EDV & LA_EDV ')
    plt.imshow(A4C_whole_trans[A4C_LV_big])

    plt.subplot(1,2,2)
    plt.title('LV_ESV & LA_EDV')
    plt.imshow(A4C_whole_trans[A4C_LV_small])

    plt.show()


    # plt.figure()
    # plt.subplot(1,2,1)
    # plt.title('LA_EDV')
    # plt.imshow(A4C_LA_whole_im_ori[A4C_LV_big])

    # plt.subplot(1,2,2)
    # plt.title('LA_ESV')
    # plt.imshow(A4C_LA_whole_im_ori[A4C_LV_small])

    # plt.show()

    plt.figure()
    plt.subplot(2,2,1)
    plt.title('Area')
    plt.plot(A4C_LV_whole_area)
    plt.plot(A4C_LA_whole_area)

    plt.subplot(2,2,2)
    plt.title('Volum')
    plt.plot(A4C_LV_whole_LV_volum)
    plt.plot(A4C_LA_whole_LV_volum)

    plt.subplot(2,2,3)
    plt.title('Length')
    plt.plot(A4C_LV_whole_length)
    plt.plot(A4C_LA_whole_length)

    plt.subplot(2,2,4)
    plt.title('Width')
    plt.plot(A4C_LV_whole_width)
    plt.plot(A4C_LA_whole_width)

    plt.show()
    
    LV_all = []
    LV_all.append(A4C_LV_whole_area)
    LV_all.append(A4C_LV_whole_LV_volum)
    LV_all.append(A4C_LV_whole_length)
    LV_all.append(A4C_LV_whole_width)
    
    LA_all = []
    LA_all.append(A4C_LA_whole_area)
    LA_all.append(A4C_LA_whole_LV_volum)
    LA_all.append(A4C_LA_whole_length)
    LA_all.append(A4C_LA_whole_width)
    #Volum
    LV_Volum_big = A4C_LV_whole_LV_volum[A4C_LV_big]
    LV_Volum_small = A4C_LV_whole_LV_volum[A4C_LV_small]
    LA_Volum_big = A4C_LA_whole_LV_volum[A4C_LV_big]
    LA_Volum_small = A4C_LA_whole_LV_volum[A4C_LV_small]
    
    Vol = [LV_Volum_big,LV_Volum_small,LA_Volum_big,LA_Volum_small]
    #length
    LV_length_big = A4C_LV_whole_length[A4C_LV_big]
    LV_length_small = A4C_LV_whole_length[A4C_LV_small]
    LA_length_big = A4C_LA_whole_length[A4C_LV_big]
    LA_length_small = A4C_LA_whole_length[A4C_LV_small]
    
    Len = [LV_length_big,LV_length_small,LA_length_big,LA_length_small]
    #Width
    LV_Width_big = A4C_LV_whole_width[A4C_LV_big]
    LV_Width_small = A4C_LV_whole_width[A4C_LV_small]
    LA_Width_big = A4C_LA_whole_width[A4C_LV_big]
    LA_Width_small = A4C_LA_whole_width[A4C_LV_small]
    
    Wid = [LV_Width_big,LV_Width_small,LA_Width_big,LA_Width_small]
    #Area
    LV_Area_big = A4C_LV_whole_area[A4C_LV_big]
    LV_Area_small = A4C_LV_whole_area[A4C_LV_small]
    LA_Area_big = A4C_LA_whole_area[A4C_LV_big]
    LA_Area_small = A4C_LA_whole_area[A4C_LV_small]
    
    Area = [LV_Area_big,LV_Area_small,LA_Area_big,LA_Area_small]
    
    return Vol, Len, Wid, Area, LV_all,LA_all

def A4C_all_2(A4C_path,A4C_LV_model,A4C_LA_model,A=1):
    #img_path = 
    #A4C_path = r"F:\馬交計畫\code\20191007_code_summary\Demo_dicom\96HASOTE"

    image_array = single_dicom(A4C_path)
    s,s4 = dicom_image_processing(image_array)

    #Load model
    #A4C_LV_model,A4C_LA_model,A2C_LV_model,A2C_LA_model,PSAX_big_model,PSAX_small_model = A4C_LV_loading_model()
    #A4C_LV_model,A4C_LA_model = A4C_LV_loading_model()
    preds_train_A4C_LV= A4C_LV_model.predict(s4, verbose=1)
    preds_train_A4C_LA= A4C_LA_model.predict(s4, verbose=1)
    #covex_hull get area
    A4C_LV_whole_area = []
    A4C_LV_whole_length = []
    A4C_LV_whole_width = []
    A4C_LV_whole_LV_volum = []
    A4C_LV_whole_im = []
    A4C_LV_whole_im_ori = []


    A4C_LA_whole_area = []
    A4C_LA_whole_length = []
    A4C_LA_whole_width = []
    A4C_LA_whole_LV_volum = []
    A4C_LA_whole_im = []
    A4C_LA_whole_im_ori = []
    A4C_whole_trans = []
    A4C_LV_whole_test1 = []
    A4C_LA_whole_test1 = []
    A4C_LA_whole_test2 = []
    A4C_LA_whole_test3 = []
    

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(5, 5))



    for a in range(preds_train_A4C_LV.shape[0]):
        #print("現在是",a)
        #c = cv2.erode(preds_train_A4C_LV[a],kernel)
        #d = cv2.erode(preds_train_A4C_LA[a],kernel)
        #######LV                    
        A4C_LV_imRGB_ori,A4C_LV_imRGB, A4C_LV_area, A4C_LV_length, A4C_LV_width = Area_and_length2(s4[a],preds_train_A4C_LV[a],"A4C_LV")
        A4C_LV_volum_one = LV_volum(A4C_LV_area, A4C_LV_length)
        A4C_LV_ori_image,A4C_LV_ori_pre_image,A4C_LV_img= Transparent(s4[a],preds_train_A4C_LV[a],"A4C_LV")
        
        A4C_LV_ori_image_color = A4C_LV_ori_image.copy()
        A4C_LV_ori_image_color.paste(A4C_LV_img,(0, 0),A4C_LV_img)
        
        A4C_LV_ori_image_2 = A4C_LV_ori_image.copy()
        A4C_LV_ori_image_2.paste(A4C_LV_img,(0, 0),A4C_LV_img)
    
        A4C_LV_whole_area.append(A4C_LV_area)
        A4C_LV_whole_length.append(A4C_LV_length)
        A4C_LV_whole_width.append(A4C_LV_width)
        A4C_LV_whole_LV_volum.append(A4C_LV_volum_one)
        A4C_LV_whole_im.append(A4C_LV_imRGB)
        A4C_LV_whole_im_ori.append(A4C_LV_imRGB_ori)
        A4C_LV_whole_test1.append(A4C_LV_ori_image_2)
        
        
        


#######LA
        A4C_LA_imRGB_ori,A4C_LA_imRGB, A4C_LA_area, A4C_LA_length, A4C_LA_width = Area_and_length2(s4[a],preds_train_A4C_LA[a],"A4C_LA")
        A4C_LA_volum_one = LV_volum(A4C_LA_area, A4C_LA_length) 
        A4C_LA_ori_image,A4C_LA_ori_pre_image,A4C_LA_img= Transparent(s4[a],preds_train_A4C_LA[a],"A4C_LA")
        A4C_LV_ori_image_color.paste(A4C_LA_img,(0, 0),A4C_LA_img)
        A4C_LA_ori_image_2 = A4C_LA_ori_image.copy()
        A4C_LA_ori_image_2.paste(A4C_LA_img,(0, 0),A4C_LA_img)
    
        A4C_LA_whole_area.append(A4C_LA_area)
        A4C_LA_whole_length.append(A4C_LA_length)
        A4C_LA_whole_width.append(A4C_LA_width)
        A4C_LA_whole_LV_volum.append(A4C_LA_volum_one)
        A4C_LA_whole_im.append(A4C_LA_imRGB)
        A4C_LA_whole_im_ori.append(A4C_LA_imRGB_ori)
        A4C_LA_whole_test2.append(A4C_LA_ori_image)
        A4C_LA_whole_test3.append(A4C_LA_ori_pre_image)
        A4C_LA_whole_test1.append(A4C_LA_ori_image_2)
    
        A4C_whole_trans.append(A4C_LV_ori_image_color)
        
        pylab.rcParams['figure.dpi'] = 150
        pylab.rcParams['figure.figsize'] = (15.0, 8.0)

    A4C_LV_big = A4C_LV_whole_area.index(max(A4C_LV_whole_area))
    A4C_LV_small = A4C_LV_whole_area.index(min(A4C_LV_whole_area))
    A4C_LA_big = A4C_LA_whole_area.index(max(A4C_LA_whole_area))
    A4C_LA_small = A4C_LA_whole_area.index(min(A4C_LA_whole_area))
    #whole_im[big]
    #whole_im[small]
    if A==1:
        plt.figure()
        plt.subplot(1,2,1)
        plt.title('LV_EDV & LA_EDV ')
        plt.imshow(A4C_whole_trans[A4C_LV_big])

        plt.subplot(1,2,2)
        plt.title('LV_ESV & LA_EDV')
        plt.imshow(A4C_whole_trans[A4C_LV_small])

        plt.show()
    elif A==2:#single prediction image for LV or LA only
        print("shape:",len(A4C_LA_whole_area))
        print("A4C_LV_big",A4C_LV_big)
        print("shapeimage:",A4C_LA_whole_area[A4C_LV_big])
        plt.figure()
        plt.subplot(1,2,1)
        plt.title('LV_EDV & LA_EDV ')
        plt.imshow(A4C_LA_whole_im_ori[A4C_LV_big])

        plt.subplot(1,2,2)
        plt.title('LV_ESV & LA_EDV')
        plt.imshow(A4C_LA_whole_im_ori[A4C_LV_small])
        plt.show()
        
        plt.figure()
        plt.subplot(1,2,1)
        plt.title('LV_EDV & LA_EDV ')
        plt.imshow(A4C_LV_whole_test1[A4C_LV_big])

        plt.subplot(1,2,2)
        plt.title('LV_ESV & LA_EDV')
        plt.imshow(A4C_LV_whole_test1[A4C_LV_small])

        plt.show()
        
        plt.figure()
        plt.subplot(1,2,1)
        plt.title('LV_EDV & LA_EDV ')
        plt.imshow(A4C_LA_whole_test1[A4C_LV_big])

        plt.subplot(1,2,2)
        plt.title('LV_ESV & LA_EDV')
        plt.imshow(A4C_LA_whole_test1[A4C_LV_small])

        plt.show()
        #=====================================
        p = preds_train_A4C_LV[A4C_LV_big].reshape(128,128)
        pp = preds_train_A4C_LV[A4C_LV_small].reshape(128,128)
        #print(preds_train_A4C_LV[A4C_LV_big].shape)
        plt.figure()
        plt.subplot(1,2,1)
        plt.title('LV_EDV & LA_EDV ')
        plt.imshow(p,cmap="gray")

        plt.subplot(1,2,2)
        plt.title('LV_ESV & LA_EDV')
        plt.imshow(pp,cmap="gray")

        plt.show()
        #=====================================
        p = preds_train_A4C_LA[A4C_LV_big].reshape(128,128)
        pp = preds_train_A4C_LA[A4C_LV_small].reshape(128,128)
        #print(preds_train_A4C_LA[A4C_LV_big].shape)
        plt.figure()
        plt.subplot(1,2,1)
        plt.title('LV_EDV & LA_EDV ')
        plt.imshow(p,cmap="gray")

        plt.subplot(1,2,2)
        plt.title('LV_ESV & LA_EDV')
        plt.imshow(pp,cmap="gray")

        plt.show()
        
#         print(preds_train_A4C_LA[A4C_LV_big].shape)
#         plt.figure()
#         plt.subplot(1,2,1)
#         plt.title('LV_EDV & LA_EDV ')
#         plt.imshow(preds_train_A4C_LA[A4C_LV_big])

#         plt.subplot(1,2,2)
#         plt.title('LV_ESV & LA_EDV')
#         plt.imshow(preds_train_A4C_LA[A4C_LV_small])

#         plt.show()
        

    	# plt.figure()
   	 # plt.subplot(1,2,1)
   	 # plt.title('LA_EDV')
   	 # plt.imshow(A4C_LA_whole_im_ori[A4C_LV_big])
	
   	 # plt.subplot(1,2,2)
   	 # plt.title('LA_ESV')
   	 # plt.imshow(A4C_LA_whole_im_ori[A4C_LV_small])

   	 # plt.show()

        plt.figure()
        plt.subplot(2,2,1)
        plt.title('Area')
        plt.plot(A4C_LV_whole_area)
        plt.plot(A4C_LA_whole_area)

        plt.subplot(2,2,2)
        plt.title('Volum')
        plt.plot(A4C_LV_whole_LV_volum)
        plt.plot(A4C_LA_whole_LV_volum)

        plt.subplot(2,2,3)
        plt.title('Length')
        plt.plot(A4C_LV_whole_length)
        plt.plot(A4C_LA_whole_length)

        plt.subplot(2,2,4)
        plt.title('Width')
        plt.plot(A4C_LV_whole_width)
        plt.plot(A4C_LA_whole_width)

        plt.show()    

    LV_all = []
    LV_all.append(A4C_LV_whole_area)
    LV_all.append(A4C_LV_whole_LV_volum)
    LV_all.append(A4C_LV_whole_length)
    LV_all.append(A4C_LV_whole_width)
    
    LA_all = []
    LA_all.append(A4C_LA_whole_area)
    LA_all.append(A4C_LA_whole_LV_volum)
    LA_all.append(A4C_LA_whole_length)
    LA_all.append(A4C_LA_whole_width)
    #Volum
    LV_Volum_big = A4C_LV_whole_LV_volum[A4C_LV_big]
    LV_Volum_small = A4C_LV_whole_LV_volum[A4C_LV_small]
    LA_Volum_big = A4C_LA_whole_LV_volum[A4C_LV_big]
    LA_Volum_small = A4C_LA_whole_LV_volum[A4C_LV_small]
    
    Vol = [LV_Volum_big,LV_Volum_small,LA_Volum_big,LA_Volum_small]
    #length
    LV_length_big = A4C_LV_whole_length[A4C_LV_big]
    LV_length_small = A4C_LV_whole_length[A4C_LV_small]
    LA_length_big = A4C_LA_whole_length[A4C_LV_big]
    LA_length_small = A4C_LA_whole_length[A4C_LV_small]
    
    Len = [LV_length_big,LV_length_small,LA_length_big,LA_length_small]
    #Width
    LV_Width_big = A4C_LV_whole_width[A4C_LV_big]
    LV_Width_small = A4C_LV_whole_width[A4C_LV_small]
    LA_Width_big = A4C_LA_whole_width[A4C_LV_big]
    LA_Width_small = A4C_LA_whole_width[A4C_LV_small]
    
    Wid = [LV_Width_big,LV_Width_small,LA_Width_big,LA_Width_small]
    #Area
    LV_Area_big = A4C_LV_whole_area[A4C_LV_big]
    LV_Area_small = A4C_LV_whole_area[A4C_LV_small]
    LA_Area_big = A4C_LA_whole_area[A4C_LV_big]
    LA_Area_small = A4C_LA_whole_area[A4C_LV_small]
    
    Area = [LV_Area_big,LV_Area_small,LA_Area_big,LA_Area_small]
    
   
    return A4C_LV_whole_area,A4C_LA_whole_area,A4C_LV_whole_LV_volum,A4C_LA_whole_LV_volum,A4C_LV_whole_length,A4C_LA_whole_length,A4C_LV_whole_width,A4C_LA_whole_width,Vol,Len, Wid, Area, LV_all,LA_all    


# In[ ]:


def PSAX_all(A4C_path,A4C_LV_model,A4C_LA_model):
    #img_path = 
    #A4C_path = r"F:\馬交計畫\code\20191007_code_summary\Demo_dicom\96HASOTE"

    image_array = single_dicom(A4C_path)
    s,s4 = dicom_image_processing(image_array)

    #Load model
    #A4C_LV_model,A4C_LA_model,A2C_LV_model,A2C_LA_model,PSAX_big_model,PSAX_small_model = A4C_LV_loading_model()
    #A4C_LV_model,A4C_LA_model = PSAX_loading_model()
    preds_train_A4C_LV= A4C_LV_model.predict(s4, verbose=1)
    preds_train_A4C_LA= A4C_LA_model.predict(s4, verbose=1)
    #covex_hull get area
    A4C_LV_whole_area = []
    A4C_LV_whole_length = []
    A4C_LV_whole_width = []
    A4C_LV_whole_LV_volum = []
    A4C_LV_whole_im = []
    A4C_LV_whole_im_ori = []


    A4C_LA_whole_area = []
    A4C_LA_whole_length = []
    A4C_LA_whole_width = []
    A4C_LA_whole_LV_volum = []
    A4C_LA_whole_im = []
    A4C_LA_whole_im_ori = []
    A4C_whole_trans = []

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(5, 5))



    for a in range(preds_train_A4C_LV.shape[0]):
        #print("現在是",a)
        #c = cv2.erode(preds_train_A4C_LV[a],kernel)
        #d = cv2.erode(preds_train_A4C_LA[a],kernel)
        #######LV                    
        A4C_LV_imRGB_ori,A4C_LV_imRGB, A4C_LV_area, A4C_LV_length, A4C_LV_width = Area_and_length2(s4[a],preds_train_A4C_LV[a],"PSAX_big")
        A4C_LV_volum_one = LV_volum(A4C_LV_area, A4C_LV_length)
        A4C_LV_ori_image,A4C_LV_ori_pre_image,A4C_LV_img= Transparent(s4[a],preds_train_A4C_LV[a],"PSAX_big")
        
        A4C_LV_ori_image_color = A4C_LV_ori_image.copy()
        A4C_LV_ori_image_color.paste(A4C_LV_img,(0, 0),A4C_LV_img)
    
        A4C_LV_whole_area.append(A4C_LV_area)
        A4C_LV_whole_length.append(A4C_LV_length)
        A4C_LV_whole_width.append(A4C_LV_width)
        A4C_LV_whole_LV_volum.append(A4C_LV_volum_one)
        A4C_LV_whole_im.append(A4C_LV_imRGB)
        A4C_LV_whole_im_ori.append(A4C_LV_imRGB_ori)
        
        
        #######LA
        A4C_LA_imRGB_ori,A4C_LA_imRGB, A4C_LA_area, A4C_LA_length, A4C_LA_width = Area_and_length2(s4[a],preds_train_A4C_LA[a],"PSAX_small")
        A4C_LA_volum_one = LV_volum(A4C_LA_area, A4C_LA_length) 
        A4C_LA_ori_image,A4C_LA_ori_pre_image,A4C_LA_img= Transparent(s4[a],preds_train_A4C_LA[a],"PSAX_small")
        A4C_LV_ori_image_color.paste(A4C_LA_img,(0, 0),A4C_LA_img)
    
        A4C_LA_whole_area.append(A4C_LA_area)
        A4C_LA_whole_length.append(A4C_LA_length)
        A4C_LA_whole_width.append(A4C_LA_width)
        A4C_LA_whole_LV_volum.append(A4C_LA_volum_one)
        A4C_LA_whole_im.append(A4C_LA_imRGB)
        A4C_LA_whole_im_ori.append(A4C_LA_imRGB_ori)
    
        A4C_whole_trans.append(A4C_LV_ori_image_color)
        
        pylab.rcParams['figure.dpi'] = 150
        pylab.rcParams['figure.figsize'] = (15.0, 8.0)

    A4C_LV_big = A4C_LV_whole_area.index(max(A4C_LV_whole_area))
    A4C_LV_small = A4C_LV_whole_area.index(min(A4C_LV_whole_area))
    A4C_LA_big = A4C_LA_whole_area.index(max(A4C_LA_whole_area))
    A4C_LA_small = A4C_LA_whole_area.index(min(A4C_LA_whole_area))
    #whole_im[big]
    #whole_im[small]

    plt.figure()
    plt.subplot(1,2,1)
    plt.title('LV_EDV & LA_EDV ')
    plt.imshow(A4C_whole_trans[A4C_LV_big])

    plt.subplot(1,2,2)
    plt.title('LV_ESV & LA_EDV')
    plt.imshow(A4C_whole_trans[A4C_LV_small])

    plt.show()


    # plt.figure()
    # plt.subplot(1,2,1)
    # plt.title('LA_EDV')
    # plt.imshow(A4C_LA_whole_im_ori[A4C_LV_big])

    # plt.subplot(1,2,2)
    # plt.title('LA_ESV')
    # plt.imshow(A4C_LA_whole_im_ori[A4C_LV_small])

    # plt.show()

    plt.figure()
    plt.subplot(2,2,1)
    plt.title('Area')
    plt.plot(A4C_LV_whole_area)
    plt.plot(A4C_LA_whole_area)

    plt.subplot(2,2,2)
    plt.title('Volum')
    plt.plot(A4C_LV_whole_LV_volum)
    plt.plot(A4C_LA_whole_LV_volum)

    plt.subplot(2,2,3)
    plt.title('Length')
    plt.plot(A4C_LV_whole_length)
    plt.plot(A4C_LA_whole_length)

    plt.subplot(2,2,4)
    plt.title('Width')
    plt.plot(A4C_LV_whole_width)
    plt.plot(A4C_LA_whole_width)

    plt.show()
    
    LV_all = []
    LV_all.append(A4C_LV_whole_area)
    LV_all.append(A4C_LV_whole_LV_volum)
    LV_all.append(A4C_LV_whole_length)
    LV_all.append(A4C_LV_whole_width)
    
    LA_all = []
    LA_all.append(A4C_LA_whole_area)
    LA_all.append(A4C_LA_whole_LV_volum)
    LA_all.append(A4C_LA_whole_length)
    LA_all.append(A4C_LA_whole_width)
    
    #Volum
    LV_Volum_big = A4C_LV_whole_LV_volum[A4C_LV_big]
    LV_Volum_small = A4C_LV_whole_LV_volum[A4C_LV_small]
    LA_Volum_big = A4C_LA_whole_LV_volum[A4C_LV_big]
    LA_Volum_small = A4C_LA_whole_LV_volum[A4C_LV_small]
    
    Vol = [LV_Volum_big,LV_Volum_small,LA_Volum_big,LA_Volum_small]
    #length
    LV_length_big = A4C_LV_whole_length[A4C_LV_big]
    LV_length_small = A4C_LV_whole_length[A4C_LV_small]
    LA_length_big = A4C_LV_whole_length[A4C_LV_big]
    LA_length_small = A4C_LV_whole_length[A4C_LV_small]
    
    Len = [LV_length_big,LV_length_small,LA_length_big,LA_length_small]
    #Width
    LV_Width_big = A4C_LV_whole_width[A4C_LV_big]
    LV_Width_small = A4C_LV_whole_width[A4C_LV_small]
    LA_Width_big = A4C_LA_whole_width[A4C_LV_big]
    LA_Width_small = A4C_LA_whole_width[A4C_LV_small]
    
    Wid = [LV_Width_big,LV_Width_small,LA_Width_big,LA_Width_small]
    #Area
    LV_Area_big = A4C_LV_whole_area[A4C_LV_big]
    LV_Area_small = A4C_LV_whole_area[A4C_LV_small]
    LA_Area_big = A4C_LA_whole_area[A4C_LV_big]
    LA_Area_small = A4C_LA_whole_area[A4C_LV_small]
    
    Area = [LV_Area_big,LV_Area_small,LA_Area_big,LA_Area_small]
    
    return Vol, Len, Wid, Area,LV_all,LA_all
    


# In[ ]:


def A2C_all(A4C_path,A4C_LV_model,A4C_LA_model):
    #img_path = 
    #A4C_path = r"F:\馬交計畫\code\20191007_code_summary\Demo_dicom\96HASOTE"

    image_array = single_dicom(A4C_path)
    s,s4 = dicom_image_processing(image_array)

    #Load model
    #A4C_LV_model,A4C_LA_model,A2C_LV_model,A2C_LA_model,PSAX_big_model,PSAX_small_model = A4C_LV_loading_model()
   # A4C_LV_model,A4C_LA_model = A2C_loading_model()
    preds_train_A4C_LV= A4C_LV_model.predict(s4, verbose=1)
    preds_train_A4C_LA= A4C_LA_model.predict(s4, verbose=1)
    #covex_hull get area
    A4C_LV_whole_area = []
    A4C_LV_whole_length = []
    A4C_LV_whole_width = []
    A4C_LV_whole_LV_volum = []
    A4C_LV_whole_im = []
    A4C_LV_whole_im_ori = []


    A4C_LA_whole_area = []
    A4C_LA_whole_length = []
    A4C_LA_whole_width = []
    A4C_LA_whole_LV_volum = []
    A4C_LA_whole_im = []
    A4C_LA_whole_im_ori = []
    A4C_whole_trans = []

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(5, 5))



    for a in range(preds_train_A4C_LV.shape[0]):
        #print("現在是",a)
        #c = cv2.erode(preds_train_A4C_LV[a],kernel)
        #d = cv2.erode(preds_train_A4C_LA[a],kernel)
        #######LV                    
        A4C_LV_imRGB_ori,A4C_LV_imRGB, A4C_LV_area, A4C_LV_length, A4C_LV_width = Area_and_length2(s4[a],preds_train_A4C_LV[a],"A2C_LV")
        A4C_LV_volum_one = LV_volum(A4C_LV_area, A4C_LV_length)
        A4C_LV_ori_image,A4C_LV_ori_pre_image,A4C_LV_img= Transparent(s4[a],preds_train_A4C_LV[a],"A2C_LV")
        
        A4C_LV_ori_image_color = A4C_LV_ori_image.copy()
        A4C_LV_ori_image_color.paste(A4C_LV_img,(0, 0),A4C_LV_img)
    
        A4C_LV_whole_area.append(A4C_LV_area)
        A4C_LV_whole_length.append(A4C_LV_length)
        A4C_LV_whole_width.append(A4C_LV_width)
        A4C_LV_whole_LV_volum.append(A4C_LV_volum_one)
        A4C_LV_whole_im.append(A4C_LV_imRGB)
        A4C_LV_whole_im_ori.append(A4C_LV_imRGB_ori)
        
        
        #######LA
        A4C_LA_imRGB_ori,A4C_LA_imRGB, A4C_LA_area, A4C_LA_length, A4C_LA_width = Area_and_length2(s4[a],preds_train_A4C_LA[a],"A2C_LA")
        A4C_LA_volum_one = LV_volum(A4C_LA_area, A4C_LA_length) 
        A4C_LA_ori_image,A4C_LA_ori_pre_image,A4C_LA_img= Transparent(s4[a],preds_train_A4C_LA[a],"A2C_LA")
        A4C_LV_ori_image_color.paste(A4C_LA_img,(0, 0),A4C_LA_img)
    
        A4C_LA_whole_area.append(A4C_LA_area)
        A4C_LA_whole_length.append(A4C_LA_length)
        A4C_LA_whole_width.append(A4C_LA_width)
        A4C_LA_whole_LV_volum.append(A4C_LA_volum_one)
        A4C_LA_whole_im.append(A4C_LA_imRGB)
        A4C_LA_whole_im_ori.append(A4C_LA_imRGB_ori)
    
        A4C_whole_trans.append(A4C_LV_ori_image_color)
        
        pylab.rcParams['figure.dpi'] = 150
        pylab.rcParams['figure.figsize'] = (15.0, 8.0)

    A4C_LV_big = A4C_LV_whole_area.index(max(A4C_LV_whole_area))
    A4C_LV_small = A4C_LV_whole_area.index(min(A4C_LV_whole_area))
    A4C_LA_big = A4C_LA_whole_area.index(max(A4C_LA_whole_area))
    A4C_LA_small = A4C_LA_whole_area.index(min(A4C_LA_whole_area))
    #whole_im[big]
    #whole_im[small]

    plt.figure()
    plt.subplot(1,2,1)
    plt.title('LV_EDV & LA_EDV ')
    plt.imshow(A4C_whole_trans[A4C_LV_big])

    plt.subplot(1,2,2)
    plt.title('LV_ESV & LA_EDV')
    plt.imshow(A4C_whole_trans[A4C_LV_small])

    plt.show()


    # plt.figure()
    # plt.subplot(1,2,1)
    # plt.title('LA_EDV')
    # plt.imshow(A4C_LA_whole_im_ori[A4C_LV_big])

    # plt.subplot(1,2,2)
    # plt.title('LA_ESV')
    # plt.imshow(A4C_LA_whole_im_ori[A4C_LV_small])

    # plt.show()

    plt.figure()
    plt.subplot(2,2,1)
    plt.title('Area')
    plt.plot(A4C_LV_whole_area)
    plt.plot(A4C_LA_whole_area)

    plt.subplot(2,2,2)
    plt.title('Volum')
    plt.plot(A4C_LV_whole_LV_volum)
    plt.plot(A4C_LA_whole_LV_volum)

    plt.subplot(2,2,3)
    plt.title('Length')
    plt.plot(A4C_LV_whole_length)
    plt.plot(A4C_LA_whole_length)

    plt.subplot(2,2,4)
    plt.title('Width')
    plt.plot(A4C_LV_whole_width)
    plt.plot(A4C_LA_whole_width)

    plt.show()
        
    LV_all = []
    LV_all.append(A4C_LV_whole_area)
    LV_all.append(A4C_LV_whole_LV_volum)
    LV_all.append(A4C_LV_whole_length)
    LV_all.append(A4C_LV_whole_width)
    
    LA_all = []
    LA_all.append(A4C_LA_whole_area)
    LA_all.append(A4C_LA_whole_LV_volum)
    LA_all.append(A4C_LA_whole_length)
    LA_all.append(A4C_LA_whole_width)
    
    #Volum
    LV_Volum_big = A4C_LV_whole_LV_volum[A4C_LV_big]
    LV_Volum_small = A4C_LV_whole_LV_volum[A4C_LV_small]
    LA_Volum_big = A4C_LA_whole_LV_volum[A4C_LV_big]
    LA_Volum_small = A4C_LA_whole_LV_volum[A4C_LV_small]
    
    Vol = [LV_Volum_big,LV_Volum_small,LA_Volum_big,LA_Volum_small]
    #length
    LV_length_big = A4C_LV_whole_length[A4C_LV_big]
    LV_length_small = A4C_LV_whole_length[A4C_LV_small]
    LA_length_big = A4C_LV_whole_length[A4C_LV_big]
    LA_length_small = A4C_LV_whole_length[A4C_LV_small]
    
    Len = [LV_length_big,LV_length_small,LA_length_big,LA_length_small]
    #Width
    LV_Width_big = A4C_LV_whole_width[A4C_LV_big]
    LV_Width_small = A4C_LV_whole_width[A4C_LV_small]
    LA_Width_big = A4C_LA_whole_width[A4C_LV_big]
    LA_Width_small = A4C_LA_whole_width[A4C_LV_small]
    
    Wid = [LV_Width_big,LV_Width_small,LA_Width_big,LA_Width_small]
    #Area
    LV_Area_big = A4C_LV_whole_area[A4C_LV_big]
    LV_Area_small = A4C_LV_whole_area[A4C_LV_small]
    LA_Area_big = A4C_LA_whole_area[A4C_LV_big]
    LA_Area_small = A4C_LA_whole_area[A4C_LV_small]
    
    Area = [LV_Area_big,LV_Area_small,LA_Area_big,LA_Area_small]
    
    return Vol, Len, Wid, Area, LV_all,LA_all
    


# In[ ]:


def EF_vol2(A4C_Vol):
    big = A4C_Vol[0]
    small = A4C_Vol[1]
    EF_volum = ((big-small)/big)*100
    return round(EF_volum,2)

def LVM_vol2(A4C_Vol, A4C_Len, A4C_Wid, A4C_Area,PSAX_Vol, PSAX_Len, PSAX_Wid, PSAX_Area):
    A4C_width = round(A4C_Wid[0],2)
    PSAX_big_area = round(PSAX_Area[0],2)
    PSAX_small_area =  round(PSAX_Area[2],2)
    A4C_length = round(A4C_Len[0],2)
    
    t =  int(math.sqrt(PSAX_big_area/math.pi)-(A4C_width/2))
    
    LVM = int(1.05*(((5/6)*PSAX_big_area*(A4C_length+t))-((5/6)*PSAX_small_area*A4C_length)))
    #print("LVM:",LVM)
    
    return round(LVM,2)
def LAV_vol2(A4C_Vol, A4C_Len, A4C_Wid, A4C_Area,A2C_Vol, A2C_Len, A2C_Wid, A2C_Area):
    A4C_LA = round(A4C_Area[2],2)
    A2C_LA = round(A2C_Area[2],2)
    A4C_length = round(A4C_Len[2],2)
    
    LAV = (0.85)*(A4C_LA * A2C_LA)/A4C_length
    return round(LAV, 2)

def LAV_vol3(A4C_Len,A2C_Len,A4C_Wid,A2C_Wid):
#    A4C_Area,A2C_Area,
#     A4C_LA = round(A4C_Area,2)
#     A2C_LA = round(A2C_Area,2)
#     A4C_length = round(A4C_Len,2)
    A1 = math.pi* (A4C_Len/2)*(A4C_Wid/2)
    A2 = math.pi* (A2C_Len/2)*(A2C_Wid/2)
    L1 = A4C_Len
    L2 = A2C_Len
    LAV = (8/3*math.pi)*(A1 * A2)/((L1+L2)/2)#+ L2
    #LAV = LAV/1.535
    #LAV = (0.85)*(A4C_Area * A2C_Area)/A4C_Len
    return round(LAV, 2)

# In[2]:


def GLRCS(A4C_LV_all,PSAX_LA_all,PSAX_LV_all):
    #GLS#變化使用LV的length
    GLS = A4C_LV_all[2]
    GLS_value = round(max(GLS)-min(GLS),2)
    #GRS#變化使用psax內圓半徑的變化
    #GRS = PSAX_LV_all[2]
    GRS = [round(c/2,2) for c in  PSAX_LA_all[2]]
    GRS_value = round(max(GRS)-min(GRS),2)
    #GCS#變化使用psax外圍圓周的變化
    GCS = [math.pi*c for c in  PSAX_LV_all[2]]
    GCS_value = round(max(GCS)-min(GCS),2)

    plt.figure()
    print("GLS:",GLS_value)
    plt.subplot(3,1,1)
    plt.title('GLS')
    plt.plot(GLS)
    plt.show()

    print("GRS:",GRS_value)
    plt.subplot(3,1,2)
    plt.title('GRS')
    plt.plot(GRS)
    plt.show()

    print("GCS:",GCS_value)
    plt.subplot(3,1,3)
    plt.title('GCS')
    plt.plot(GCS)
    plt.show()
    
    return GLS_value,GRS_value,GCS_value,GLS,GRS,GCS

import imutils
from numpy import *

def distance(p0, p1):
    return math.sqrt((p0[0] - p1[0])**2 + (p0[1] - p1[1])**2)
def midpoint13(p0, p1):
    #1:2
    x = int((2*p0[0]+p1[0])/3)
    y = int((2*p0[1]+p1[1])/3)
    #print("p0, p1:",p0, p1)
    #print("1/3:",x,y)
    #2:1
    z = int((p0[0]+2*p1[0])/3)
    w = int((p0[1]+2*p1[1])/3)
    #print("2/3:",z,w)
    E = [x,y]
    E = np.asarray(E)
    F = [z,w]
    F = np.asarray(F)
    return E,F
def midpoint(p0, p1):
    x = (p0[0]+p1[0])/2
    y = (p0[1]+p1[1])/2
    return int(x),int(y)

def Arc_length(number,AK):#兩個座標
    Arc_array = []
    n = 0
    for a in range(number):
        if a > 0:
            a = round(math.sqrt((AK[a-1,0] -AK[a,0])**2 + (AK[a-1,1] - AK[a,1])**2),2)
            Arc_array.append(a)
            n = n+a
    return Arc_array,round(n,2)

def length_6_A4C(image):
    #匯入影片
    global box
    #image = cv2.imread("/media/linlab/Seagate Backup Plus Drive/馬交計畫/code/20191003_Label_for_traning/Training _data/Normal/A4C/A4C_LV/9ADBK6OO_52.jpg")
    #灰階
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    #除燥
    blurred = cv2.GaussianBlur(gray, (11, 11), 0)
    canny = cv2.Canny(blurred, 30, 150)
    #找外面區域
    (_, cnts, _) = cv2.findContours(canny.copy(), cv2.RETR_EXTERNAL,
    cv2.CHAIN_APPROX_SIMPLE)
    #把外面區域畫在圖片上
    contours = image.copy()
    #cv2.drawContours(contours, cnts, -1, (0, 255, 0), 2)  
    #畫出最大的框
    cnts = np.asarray(cnts)
    cnts.shape[0]
    cou=[]
    if int(cnts.shape[0])>int(1) :
        for cnt in cnts:
            cou.append(cnt.shape[0])
            if int(cnt.shape[0]) ==max(cou):
                x, y, w, h = cv2.boundingRect(cnt)
                rect = cv2.minAreaRect(cnt)
                box = cv2.cv.Boxpoints() if imutils.is_cv2()else cv2.boxPoints(rect)
                box = np.int0(box)
                cv2.drawContours(contours, [box], 0, (255, 0, 0), 4)
    else:
        for cnt in cnts:
            x, y, w, h = cv2.boundingRect(cnt)

            # 最小外接矩形框，有方向角
            rect = cv2.minAreaRect(cnt)
            box = cv2.cv.Boxpoints() if imutils.is_cv2()else cv2.boxPoints(rect)
            box = np.int0(box)
            cv2.drawContours(contours, [box], 0, (255, 0, 0), 4)

    # plt.imshow(contours)
    # plt.show()

    # 長方形的四個角座標
    A = box[0]
    for k in range(box.shape[0]):
        if box[k][1]<A[1]:
            #print("change")
            A = box[k]
    for k in range(box.shape[0]):
        if A[1] ==box[k][1] and A[0]<box[k][0]:
            #print("change2")
            B = box[k]
    for k in range(box.shape[0]):
        if B[0] ==box[k][0] and B[1]<box[k][1]:
            #print("change3")
            C = box[k]
    for k in range(box.shape[0]):
        if C[1] ==box[k][1] and C[0]>box[k][0]:
            #print("change4")
            D = box[k]
    #print(box.shape[0])
    #橫向中點
    I= array(midpoint(A, B))
    J= array(midpoint(C, D))
    E,F= midpoint13(A, D)
    G,H= midpoint13(B, C)
    K= array(midpoint(E, G))
    M= array(midpoint(F, H))
    #畫出六格線
    cv2.line(contours,tuple(E),tuple(G),(0, 0, 255),4)
    cv2.line(contours,tuple(F),tuple(H),(0, 0, 255),4)
    cv2.line(contours,tuple(I),tuple(J),(0, 0, 255),4)
    # plt.imshow(contours)
    # plt.show()
    # cv2.circle(contours,(K[0],K[1]), 15, (255, 0, 0), 2)
    # plt.imshow(contours)
    # plt.show()
    cnts = np.asarray(cnts.reshape(cnts.shape[1],2))
    #print(cnts.shape)
    AK = []
    IG = []
    EM = []
    KH = []
    FJ = []
    MC = []
    # print(cnts.shape[0])
    #for cnt in cnts:
    for p in range(cnts.shape[0]):
        #AK
        if cnts[p,0]>= A[0] and cnts[p,0]<= K[0] and cnts[p,1]>= A[1]and cnts[p,1]<= K[1]:
    #         print("p",p,"in AK", cnts[p,:],"A0:",A[0],"K0:",K[0])
    #         print("p",p,"in AK", cnts[p,:],"A1:",A[1],"K1:",K[1])
            AK.append(cnts[p,:])
        #IG
        elif cnts[p,0]>= I[0] and cnts[p,0]<= G[0] and cnts[p,1]>= I[1]and cnts[p,1]<= G[1]:
            IG.append(cnts[p,:])
        #EM
        elif cnts[p,0]>= E[0] and cnts[p,0]<= M[0] and cnts[p,1]>= E[1]and cnts[p,1]<= M[1]:
            EM.append(cnts[p,:])
        #KH
        elif cnts[p,0]>= K[0] and cnts[p,0]<= H[0] and cnts[p,1]>= K[1]and cnts[p,1]<= H[1]:
            KH.append(cnts[p,:])

        #FJ
        elif cnts[p,0]>= F[0] and cnts[p,0]<= J[0] and cnts[p,1]>= F[1]and cnts[p,1]<= J[1]:
            FJ.append(cnts[p,:])
        #MC
        elif cnts[p,0]>= M[0] and cnts[p,0]<= C[0] and cnts[p,1]>= M[1]and cnts[p,1]<= C[1]:
            MC.append(cnts[p,:])

    AK = np.asarray(AK)
    IG = np.asarray(IG)
    EM = np.asarray(EM)
    KH = np.asarray(KH)
    FJ = np.asarray(FJ)
    MC = np.asarray(MC)

    # AK = AK.reshape(1,AK.shape[0],1,AK.shape[1])
    # IG = IG.reshape(1,IG.shape[0],1,IG.shape[1])
    # EM = EM.reshape(1,EM.shape[0],1,EM.shape[1])
    # KH = KH.reshape(1,KH.shape[0],1,KH.shape[1])
    # FJ = FJ.reshape(1,FJ.shape[0],1,FJ.shape[1])
    # MC = MC.reshape(1,MC.shape[0],1,MC.shape[1])

    #AK
    for a in range(AK.shape[0]):
        if a ==0:
            cv2.circle(contours,(AK[0,0],AK[0,1]), 5, (255, 0, 0), -1)#-1 = 實心
        else:
            cv2.line(contours,(AK[a-1,0],AK[a-1,1]),(AK[a,0],AK[a,1]),(255, 255, 0),5)
    #IG
    for a in range(IG.shape[0]):
        if a ==0:
            cv2.circle(contours,(IG[0,0],IG[0,1]), 5, (255, 0, 0), -1)#-1 = 實心
        else:
            cv2.line(contours,(IG[a-1,0],IG[a-1,1]),(IG[a,0],IG[a,1]),(0, 255, 255),5)
    #EM
    for a in range(EM.shape[0]):
        if a ==0:
            cv2.circle(contours,(EM[0,0],EM[0,1]), 5, (255, 0, 0), -1)#-1 = 實心
        else:
            cv2.line(contours,(EM[a-1,0],EM[a-1,1]),(EM[a,0],EM[a,1]),(125, 125, 255),5)
    #KH
    for a in range(KH.shape[0]):
        if a ==0:
            cv2.circle(contours,(KH[0,0],KH[0,1]), 5, (255, 0, 0), -1)#-1 = 實心
        else:
            cv2.line(contours,(KH[a-1,0],KH[a-1,1]),(KH[a,0],KH[a,1]),(200, 100, 200),5)
    #FJ
    for a in range(FJ.shape[0]):
        if a ==0:
            cv2.circle(contours,(FJ[0,0],FJ[0,1]), 5, (255, 0, 0), -1)#-1 = 實心
        else:
            cv2.line(contours,(FJ[a-1,0],FJ[a-1,1]),(FJ[a,0],FJ[a,1]),(200, 125, 125),5)
    #MC
    for a in range(MC.shape[0]):
        if a ==0:
            cv2.circle(contours,(MC[0,0],MC[0,1]), 5, (255, 0, 0), -1)#-1 = 實心
        else:
            cv2.line(contours,(MC[a-1,0],MC[a-1,1]),(MC[a,0],MC[a,1]),(50, 125, 200),5)

#     plt.imshow(contours)
#     plt.show()


    AK_Arc_array,AK_n = Arc_length(AK.shape[0],AK)
    IG_Arc_array,IG_n = Arc_length(IG.shape[0],IG)
    EM_Arc_array,EM_n = Arc_length(EM.shape[0],EM)
    KH_Arc_array,KH_n = Arc_length(KH.shape[0],KH)
    FJ_Arc_array,FJ_n = Arc_length(FJ.shape[0],FJ)
    MC_Arc_array,MC_n = Arc_length(MC.shape[0],MC)
    print(AK_n ,IG_n ,EM_n ,KH_n ,FJ_n ,MC_n )
    #六段長度完成,要寫迴圈抓
    return AK_n ,IG_n ,EM_n ,KH_n ,FJ_n ,MC_n
