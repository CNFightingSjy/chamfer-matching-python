import cv2
import chamfer_matching as CM
import numpy as np

def match_sketch(path, templatepath):
    img = cv2.imread(path)
    template = cv2.imread(templatepath)

    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    graytplt = cv2.cvtColor(template, cv2.COLOR_RGB2GRAY)

    imgrev = (255-edge)
    cv2.imshow("rev", imgrev)

    CDTRes = CM.ChamferMatching.Chamfer_Dist_Transform(imgrev)
    cv2.imshow("CDT", CDTRes)

    
    MatchRes = CM.ChamferMatching.MeanConv()
