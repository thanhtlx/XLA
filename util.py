import cv2
import os
import math
import numpy as np
from PIL import Image, ImageEnhance
import xml.etree.ElementTree as ET

def parseXml(filepath):
    root = ET.parse(filepath).getroot()
    for type_tag in root.findall('object/bndbox'):
        xmin=  int(type_tag.find("xmin").text)
        xmax = int(type_tag.find("xmax").text)
        ymin=int(type_tag.find("ymin").text)
        ymax = int(type_tag.find("ymax").text)
        yield (xmin, xmax, ymin, ymax)


def drawOnImage(data,img):
    for cell in data:
        cv2.rectangle(img, (cell[0], cell[1]),
                      (cell[0]+cell[2], cell[1]+cell[3]), 255, 2)
    showImage(img)

def showImage(img):
    cv2.imshow("Image", img)
    cv2.waitKey(0)


def preprocess(img, factor=2):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = Image.fromarray(img)
    enhancer = ImageEnhance.Sharpness(img).enhance(factor)
    if gray.std() < 30:
        enhancer = ImageEnhance.Contrast(enhancer).enhance(factor)
    return np.array(enhancer)


def isInSameRow(c1, c2):
    c1_center = c1[1] + c1[3] / 2
    c2_bottom = c2[1] + c2[3]
    c2_top = c2[1]
    return c2_top < c1_center < c2_bottom


def average(row):
    centers = [y + h / 2 for _, y, _, h in row]
    return sum(centers) / len(centers)


def binarilize(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (3, 3), 0, 0)
    thresh = cv2.adaptiveThreshold(
        ~blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 5, -2,)
    cv2.rectangle(thresh, (0, 0), thresh.shape[::-1], 0, 3)
    return thresh


def remove_border(thresh, REMOVE_SCALE=3):
    horizontal = thresh.copy()
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
    x1, y1, w1, h1 = a
    x2, y2, w2, h2 = b

    if x2 - x1 + 1 < MIN_COLUMN_SPACE or x2-x1-w1+1 < MIN_COLUMN_SPACE/2:
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


def flatten(lst):
    result = []
    for row in lst:
        for cell in row:
            result.append(cell)
    return result


def calculate_iou(predict_bndbox, true_bndbox):
    x, y, w, h = predict_bndbox
    xmin1, ymin1, xmax1, ymax1 = x, y, x+w, y+h
    xmin2, xmax2, ymin2, ymax2 = true_bndbox

    area1 = (xmax1 - xmin1) * (ymax1- ymin1)
    area2 = (xmax2 - xmin2) * (ymax2 - ymin2)
    if xmax2 < xmin1 :
        return 0
    if xmax1 < xmin2:
        return 0
    if ymax1 < ymin2:
        return 0
    if ymax2 < ymin1:
        return 0
    xmin = max(xmin1, xmin2)
    xmax = min(xmax1, xmax2)
    ymin = max(ymin1, ymin2)
    ymax = min(ymax1, ymax2)

    w = abs(xmax - xmin)
    h = abs(ymax - ymin)
    res = round(h*w*1.0/(area1 + area2 - h*w), 4)
    return res
