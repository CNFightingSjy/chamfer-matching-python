import numpy as np
import cv2

class ChamferMatching:

    def Eucl_Distance(x1, x2, y1, y2) -> float:
        return ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5

    def Chamfer_Dist_Transform(image):
        d1 = 1
        d2 = (d1 * 2) ** 0.5
        # 像素值转换为0-1
        CDTRes = cv2.normalize(image.astype('float'), None, 0.0, 1.0, cv2.NORM_MINMAX)
        CDTRes[CDTRes!=0]=float("inf")
        # 第一阶段：从上到下，从左到右
        for i in range(1, CDTRes.shape[0] - 1):
            for j in range(1, CDTRes.shape[1] - 1):
                # 采用3x3窗口:[1 2 3 0 x 4 5 6 7]
                # 位于x的像素
                BWValCenter = CDTRes[i, j]

                # 像素0 1 2 3
                BWVar0 = CDTRes[i, j - 1]
                dist0 = d1 + BWVar0
                BWVar1 = CDTRes[i - 1, j - 1]
                dist1 = d2 + BWVar1
                BWVar2 = CDTRes[i - 1, j]
                dist2 = d1 + BWVar2
                BWVar3 = CDTRes[i - 1, j + 1]
                dist3 = d2 + BWVar3

                # 获取最小距离，并替换像素x
                fdist = min(BWValCenter, dist0, dist1, dist2, dist3)
                CDTRes[i, j] = fdist

        # 第二阶段：从下到上，从右到左
        for m in range(CDTRes.shape[0] - 2, 0,-1):
            for n in range(CDTRes.shape[1] - 2, 0,-1):
                # 位于x的像素
                BWValCenter = CDTRes[m, n]

                # 像素4 5 6 7
                BWVar4 = CDTRes[m, n + 1]
                dist4 = d1 + BWVar4
                BWVar5 = CDTRes[m + 1, n + 1]
                dist5 = d2 + BWVar5
                BWVar6 = CDTRes[m + 1, n]
                dist6 = d1 + BWVar6
                BWVar7 = CDTRes[m + 1, n - 1]
                dist7 = d2 + BWVar7

                # 获取最小距离，并替换像素x
                fdist = min(BWValCenter, dist4, dist5, dist6, dist7)
                CDTRes[m, n] = fdist
        
        CDTRes = cv2.normalize(CDTRes[1:CDTRes.shape[0]-2,1:CDTRes.shape[1]-2].astype('float'), None, 0.0, 1.0, cv2.NORM_MINMAX)
        return CDTRes

    def MeanConv(image, weight):
        height, width = image.shape
        h, w = weight.shape
        new_h = height - h + 1
        new_w = width - w + 1
        new_image = np.zeros((new_h, new_w), dtype=np.float)
        # 卷积
        for i in range(new_h):
            for j in range(new_w):
                new_image[i, j] = (np.sum(image[i:i + h, j:j + w] * weight))/w/h
        return new_image
        
