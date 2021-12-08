import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageEnhance
import math
import matplotlib.pyplot as plt

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
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    blur = cv.GaussianBlur(gray, (3, 3), 0, 0)
    thresh = cv.adaptiveThreshold(
        ~blur, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 5, -2,)
    # remove border noise
    cv.rectangle(thresh, (0, 0), thresh.shape[::-1], 0, 3)
    return thresh


def remove_border(thresh, REMOVE_SCALE=3):
    # get vertical and horizontal lines and remove them
    vertical = horizontal = thresh.copy()
    img_height, img_width = horizontal.shape

    horizontal_kernel = cv.getStructuringElement(
        cv.MORPH_RECT, (int(img_width / REMOVE_SCALE), 1))
    horizontally_opened = cv.morphologyEx(
        thresh, cv.MORPH_OPEN, horizontal_kernel)
    vertical_kernel = cv.getStructuringElement(
        cv.MORPH_RECT, (1, int(img_height / REMOVE_SCALE)))
    vertically_opened = cv.morphologyEx(thresh, cv.MORPH_OPEN, vertical_kernel)
    both = horizontally_opened + vertically_opened
    both = cv.dilate(both, cv.getStructuringElement(cv.MORPH_RECT, (5, 5)))

    borderless = thresh - both
    borderless = cv.morphologyEx(
        borderless, cv.MORPH_OPEN, cv.getStructuringElement(cv.MORPH_CROSS, (2, 2)))

    return borderless


def check_combine_column(a, b, MIN_COLUMN_SPACE=12):
    #x2 > x1
    x1, y1, w1, h1 = a
    x2, y2, w2, h2 = b

    col_space = x2 - (x1 + w1) + 1
    if col_space < MIN_COLUMN_SPACE:
        new_rect = (x1, min(y1, y2), w2+x2-x1+1,
                    max(y1+h1, y2+h2) - min(y1, y2) + 1)
        return new_rect
    else:
        return False

def reduce_col(rects, MIN_COLUMN_SPACE=12):
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


def calculate_iou(predict_bndbox, true_bndbox):
    """
    Calculate the IOU between the precdict and the true bounding box
    Predict (x, y, w, h)
    """

    x, y, w, h = predict_bndbox

    xmin1, ymin1, xmax1, ymax1 = x, y, x+w, y+h
    xmin2, xmax2, ymin2, ymax2 = true_bndbox

    xmin = max(xmin1, xmin2)
    xmax = min(xmax1, xmax2)
    ymin = max(ymin1, ymin2)
    ymax = min(ymax1, ymax2)

    inter_width = max(xmax - xmin + 1, 0)
    inter_height = max(ymax - ymin + 1, 0)

    inter_area = inter_width * inter_height

    pred_area = w * h
    true_area = (xmax2 - xmin2 + 1) * (ymax2 - ymin2 + 1)

    union_area = pred_area + true_area - inter_area

    IOU = round(inter_area / union_area, 3)
    return IOU
