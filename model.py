#-*- coding:utf-8 -*-
import os
import sys
import cv2
from math import *
import numpy as np
from PIL import Image

sys.path.append(os.getcwd() + '/ctpn')
from ctpn.text_detect import text_detect
from lib.fast_rcnn.config import cfg_from_file
from crnn.keras.model import predict as keras_crnn
# from crnn.pytorch.model import predict as pytorch_crnn
# from angle.predict import predict as angle_detector  # 文字方向检测


def sort_box(box):
    """ 
    对box进行排序: 
        text_recs[index, 0] = x1
        text_recs[index, 1] = y1
        text_recs[index, 2] = x2
        text_recs[index, 3] = y2
        text_recs[index, 4] = x3
        text_recs[index, 5] = y3
        text_recs[index, 6] = x4
        text_recs[index, 7] = y4
    """
    box = sorted(box, key=lambda x: sum([x[1], x[3], x[5], x[7]]))
    return box

def dumpRotateImage(img, degree, pt1, pt2, pt3, pt4):
    height, width = img.shape[:2]
    heightNew = int(width * fabs(sin(radians(degree))) + height * fabs(cos(radians(degree))))
    widthNew = int(height * fabs(sin(radians(degree))) + width * fabs(cos(radians(degree))))
    matRotation = cv2.getRotationMatrix2D((width / 2, height / 2), degree, 1)
    matRotation[0, 2] += (widthNew - width) / 2
    matRotation[1, 2] += (heightNew - height) / 2
    imgRotation = cv2.warpAffine(img, matRotation, (widthNew, heightNew), borderValue=(255, 255, 255))
    pt1 = list(pt1)
    pt3 = list(pt3)

    [[pt1[0]], [pt1[1]]] = np.dot(matRotation, np.array([[pt1[0]], [pt1[1]], [1]]))
    [[pt3[0]], [pt3[1]]] = np.dot(matRotation, np.array([[pt3[0]], [pt3[1]], [1]]))
    ydim, xdim = imgRotation.shape[:2]
    imgOut = imgRotation[max(1, int(pt1[1])) : min(ydim - 1, int(pt3[1])), max(1, int(pt1[0])) : min(xdim - 1, int(pt3[0]))]
    # height, width = imgOut.shape[:2]

    return imgOut

def charRec(img, text_recs, model_type='keras_crnn', adjust=False):
   """
   加载OCR模型，进行字符识别
   """
   results = {}
   xDim, yDim = img.shape[1], img.shape[0]
    
   for index, rec in enumerate(text_recs):
       xlength = int((rec[6] - rec[0]) * 0.1)
       ylength = int((rec[7] - rec[1]) * 0.2)
       if adjust:
           pt1 = (max(1, rec[0] - xlength), max(1, rec[1] - ylength))
           pt2 = (rec[2], rec[3])
           pt3 = (min(rec[6] + xlength, xDim - 2), min(yDim - 2, rec[7] + ylength))
           pt4 = (rec[4], rec[5])
       else:
           pt1 = (max(1, rec[0]), max(1, rec[1]))
           pt2 = (rec[2], rec[3])
           pt3 = (min(rec[6], xDim - 2), min(yDim - 2, rec[7]))
           pt4 = (rec[4], rec[5])
        
       degree = degrees(atan2(pt2[1] - pt1[1], pt2[0] - pt1[0]))  # 图像倾斜角度

       partImg = dumpRotateImage(img, degree, pt1, pt2, pt3, pt4)

       if partImg.shape[0] < 1 or partImg.shape[1] < 1 or partImg.shape[0] > partImg.shape[1]:  # 过滤异常图片
           continue

       image = Image.fromarray(partImg).convert('L')
       if model_type == 'keras_crnn':
           sim_pred = keras_crnn(image)
       # elif model_type == 'pytorch_crnn':
       #     sim_pred = pytorch_crnn(image)
       # else:
           raise ValueError("model_type must be keras_crnn or pytorch_crnn")

       results[index] = [rec]
       results[index].append(sim_pred)  # 识别文字
 
   return results

def model(img, model_type='keras_crnn', adjust=False, angle_detect=False):
    """
    @img: 图片
    @model_type: ocr模型类型
    @adjust: 是否调整文字识别结果
    @angle_detect: 是否检测文字方向
    """
    angle = 0
    # if angle_detect:
    #     angle = angle_detector(img=np.copy(img))  # 文字方向检测
    #     img = Image.fromarray(img)
    #     if angle == 90:
    #         img = img.transpose(Image.ROTATE_90)
    #     elif angle == 180:
    #         img = img.transpose(Image.ROTATE_180)
    #     elif angle == 270:
    #         img = img.transpose(Image.ROTATE_270)
    #     img = np.array(img)
    
    cfg_from_file('./ctpn/ctpn/text.yml')
    text_recs, tmp, img = text_detect(img)
    text_recs = sort_box(text_recs)
    result = charRec(img, text_recs, model_type, adjust)
    return result, tmp, angle

