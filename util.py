import cv2 as cv
import numpy as np
import pandas as pd
from PIL import Image, ImageEnhance
import math
import matplotlib.pyplot as plt



def display_bgr2rgp(img):
    rgb_img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    plt.figure(figsize=(10, 10))
    plt.imshow(rgb_img)
    plt.show()


def preprocess(img, factor=2):
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
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
