import numpy as np
import cv2
import math
import torch 
import matplotlib.pyplot as plt

class ChamferMatching:

    def __init__(self) -> None:
        self.chamfer_distance = 0
        self.n = 0

    def Eucl_Distance(x1, x2, y1, y2) -> float:
        return ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5

    def Chessboard_Distance(x1, x2, y1, y2):
        return max(abs(x2 - x1), abs(y2 - y1))

    def Block_Distance(x1, x2, y1, y2):
        return abs(x1 -x2) + abs(y1 -y2)

    def Chamfer_Dist_Transform(image):
        d1 = 1
        # d2 = (d1 * 2) ** 0.5
        d2 = 2
        # 像素值转换为0-1
        CDTRes = cv2.normalize(image.astype('float'), None, 0.0, 1.0, cv2.NORM_MINMAX)
        # CDTRes[CDTRes!=0]=float("inf")
        # 第一阶段：从上到下，从左到右
        for i in range(1, CDTRes.shape[0] - 1):
            for j in range(1, CDTRes.shape[1] - 1):
                # 采用3x3窗口:[1 2 3 0 x 4 5 6 7]
                # 位于x的像素
                BWValCenter = CDTRes[i, j]

                # 此处可使用欧氏距离等其他如上提供的距离计算方式
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
        
        # CDTRes = cv2.normalize(CDTRes[1:CDTRes.shape[0]-2,1:CDTRes.shape[1]-2].astype('float'), None, 0.0, 1.0, cv2.NORM_MINMAX)
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
        
    # 用img去计算与template的chamfer distance之和
    def Chamfer_Matching(img, template):
        assert img.shape == template.shape, '图片与模板大小不匹配'
        chamfer_distance = 0
        a = 255 * torch.ones(img.shape)
        reverse_img = a + (-1) * img
        binary = cv2.normalize(reverse_img.numpy(), None, 0.0, 1.0, cv2.NORM_MINMAX)
        tensor_binary = torch.tensor(binary)
        n = torch.sum(tensor_binary.view([-1, 1]))
        # print(n)

        # plt.figure()
        # plt.subplot(1,2,1)
        # plt.axis('off')
        # plt.imshow(reverse_img, cmap='Greys_r')
        # plt.subplot(1,2,2)
        # plt.axis('off')
        # plt.imshow(binary, cmap='Greys_r')
        tem = torch.tensor(template)
        Z = torch.mul(tensor_binary, tem)
        h, w = Z.size()
        sum_cd = torch.sum(Z.view([-1, 1]))
        # print(sum_cd)
        # Z.view([])
        # for i in range(0, img.shape[0] - 1):
        #     for j in range(0, img.shape[1] - 1):
        #         if(img[i, j] == 0):
        #             chamfer_distance += (template[i, j] ** 2)
        #             n += 1
                    # print(template[i, j])
        return math.sqrt(sum_cd / n)
