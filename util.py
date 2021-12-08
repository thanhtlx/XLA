import cv2
import os
import numpy as np
import pandas as pd
from PIL import Image, ImageEnhance
import math
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET

def parseXml(filepath):
    root = ET.parse(filepath).getroot()
    res = []
    ymax = 999999999
    for type_tag in root.findall('object/bndbox'):
        if ymax <= int(type_tag.find("ymin").text):
            yield res
            res = []
        xmin=  int(type_tag.find("xmin").text)
        xmax = int(type_tag.find("xmax").text)
        ymin=int(type_tag.find("ymin").text)
        ymax = int(type_tag.find("ymax").text)
        res.append((xmin, xmax, ymin, ymax))
        # yield xmin.text, xmax.text, ymin.text, ymax.text
    yield res

def showImage(img):
    cv2.imshow("Image", img)
    cv2.waitKey(0)

def display_bgr2rgp(img):
    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.figure(figsize=(10, 10))
    plt.imshow(rgb_img)
    plt.show()


def preprocess(img, factor=2):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = Image.fromarray(img)
    enhancer = ImageEnhance.Sharpness(img).enhance(factor)
    if gray.std() < 30:
        enhancer = ImageEnhance.Contrast(enhancer).enhance(factor)
    return np.array(enhancer)


def cell_in_same_row(c1, c2):
    c1_center = c1[1] + c1[3] / 2
    c2_bottom = c2[1] + c2[3]
    c2_top = c2[1]
    return c2_top < c1_center < c2_bottom


def avg_height_of_center(row):
    centers = [y + h / 2 for x, y, w, h in row]
    return sum(centers) / len(centers)


def binarilize(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (3, 3), 0, 0)
    thresh = cv2.adaptiveThreshold(
        ~blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 5, -2,)
    # remove border noise
    cv2.rectangle(thresh, (0, 0), thresh.shape[::-1], 0, 3)
    return thresh


def remove_border(thresh, REMOVE_SCALE=3):
    # get vertical and horizontal lines and remove them
    vertical = horizontal = thresh.copy()
    img_height, img_width = horizontal.shape

    horizontal_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT, (int(img_width / REMOVE_SCALE), 1))
    horizontally_opened = cv2.morphologyEx(
        thresh, cv2.MORPH_OPEN, horizontal_kernel)
    vertical_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT, (1, int(img_height / REMOVE_SCALE)))
    vertically_opened = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, vertical_kernel)
    both = horizontally_opened + vertically_opened
    both = cv2.dilate(both, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)))

    borderless = thresh - both
    borderless = cv2.morphologyEx(
        borderless, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_CROSS, (2, 2)))

    return borderless


def check_combine_column(a, b, MIN_COLUMN_SPACE=10):
    #x2 > x1
    x1, y1, w1, h1 = a
    x2, y2, w2, h2 = b

    if x2 - x1 + 1 < MIN_COLUMN_SPACE or x2-x1-w1+1 < MIN_COLUMN_SPACE/4:
        new_rect = (x1, min(y1, y2), w2+x2-x1+1,
                    max(y1+h1, y2+h2) - min(y1, y2) + 1)
        return new_rect
    elif w2 < MIN_COLUMN_SPACE:
        new_rect = (x1, min(y1, y2), w2+x2-x1+1,
                    max(y1+h1, y2+h2) - min(y1, y2) + 1)
        return new_rect
    else:
        return False

def reduce_col(rects, MIN_COLUMN_SPACE=10):
    cursor = len(rects) - 1
    while cursor > 0:
        last = rects[cursor]
        next_last = rects[cursor-1] if rects[cursor-1] else None
        if next_last:
            check = check_combine_column(next_last, last, MIN_COLUMN_SPACE)
            if check != False:
                rects.pop(cursor)
                rects.pop(cursor - 1)
                rects.insert(cursor-1, check)
        cursor -= 1
    return rects


def ravel(lst):
    result = []
    for row in lst:
        for cell in row:
            result.append(cell)

    return result


def cal_iou_row(true_row, pred_row, img):
    res = []
    for pred_bnd in pred_row:
        tmp = 0
        for true_bnd in true_row:
            iou = calculate_iou(pred_bnd,true_bnd,img)
            if iou >= tmp:
                tmp = iou
        res.append(tmp)
    return res

def calculate_iou(predict_bndbox, true_bndbox,img):
    """
    Calculate the IOU between the precdict and the true bounding box
    Predict (x, y, w, h)
    """
    # img2 = img.copy()
    # cv2.rectangle(img2, (predict_bndbox[0], predict_bndbox[1]),
    #               (predict_bndbox[0]+predict_bndbox[2], predict_bndbox[1]+predict_bndbox[3]), (255, 255, 255), 2)
    # cv2.rectangle(img2, (true_bndbox[0], true_bndbox[2]),
    #               (true_bndbox[1],true_bndbox[3]), (255,255,0), 2)
    # showImage(img2)
    x, y, w, h = predict_bndbox


    xmin1, ymin1, xmax1, ymax1 = x, y, x+w, y+h
    xmin2, xmax2, ymin2, ymax2 = true_bndbox



    area1 = (xmax1 - xmin1) * (ymax1- ymin1)
    area2 = (xmax2 - xmin2) * (ymax2 - ymin2)
    h = ymax1 - ymin2
    w = xmax1 - xmin2
    # case 1
    if xmax2 < xmin1 :
        return 0
    if xmax1 < xmin2:
        return 0
    if ymax1 < ymin2:
        return 0
    if ymax2 < ymin1:
        return 0
    # if xmax1 < xmin2 and ymax1 < ymin2:
    #     return 0
    # # case 2 
    # if xmax1 < xmin2 and ymax2 < ymin1:
    #     return 0
    # # case 3 
    # if xmax2 < xmin1 and ymax2 < ymin1:
    #     return 0
    # # case 4 
    # if xmax2 < xmin1 and ymin2 < ymax1:
    #     return 0
    xmin = max(xmin1, xmin2)
    xmax = min(xmax1, xmax2)
    ymin = max(ymin1, ymin2)
    ymax = min(ymax1, ymax2)

    w = abs(xmax - xmin)
    h = abs(ymax - ymin)
    # print(area1)
    # print(area2)
    # print(h*w)
    # print("===============")
    img2 = img.copy()
    cv2.rectangle(img2, (predict_bndbox[0], predict_bndbox[1]),
                  (predict_bndbox[0]+predict_bndbox[2], predict_bndbox[1]+predict_bndbox[3]), (255, 255, 255), 2)
    # cv2.rectangle(img2, (true_bndbox[0], true_bndbox[2]),
    #               (true_bndbox[1],true_bndbox[3]), (255,255,0), 2)
    cv2.rectangle(img2, (xmin, ymin), (xmax,ymax), 2)
    # showImage(img2)
    res = round(h*w*1.0/(area1 + area2 - h*w), 4)
    print (res)
    return res
