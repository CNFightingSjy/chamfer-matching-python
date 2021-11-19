import cv2
import chamfer_matching as CM
import numpy as np
import matplotlib.pyplot as plt

def match_sketch(path, templatepath):
    img = cv2.imread(path)
    template = cv2.imread(templatepath)

    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    graytplt = cv2.cvtColor(template, cv2.COLOR_RGB2GRAY)

    # 显示直方图
    # plt.hist(img.ravel(), 256, [0, 256])
    # plt.show()

    # ret, binary = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY)
    # cv2.imshow("二值化图像", binary)
    # imgrev = (255-edge)
    # cv2.imshow("rev", imgrev)

    # CDTRes = CM.ChamferMatching.Chamfer_Dist_Transform(imgrev)
    # cv2.imshow("CDT", CDTRes)

    # MatchRes = CM.ChamferMatching.MeanConv(imgrev, graytplt)

    # row, column = MatchRes.shape
    # position = np.argmin(MatchRes)
    # print(position)
    # m, n = divmod(position, column)
    # canvas = img  

if __name__ == '__main__':
    match_sketch("/data/shijianyang/data/sketch/sketch1.png", "/data/shijianyang/data/sketch/sketch2.png")