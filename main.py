import cv2
import chamfer_matching as CM
import numpy as np

def match_sketch(path, templatepath):
    img = cv2.imread(path)
    template = cv2.imread(templatepath)