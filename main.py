import cv2
import chamfer_matching as CM
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
import csv
from progress.bar import Bar

# 对path下的图片进行Chamfer Transform
# 使用templatepath下的图片二值化后去匹配path对应图片是否存在相似形状的衣服
# 返回Chamfer Distance
def match_sketch(path, templatepath):
    # print(templatepath)
    img = cv2.imread(path)
    template = cv2.imread(templatepath)

    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    graytplt = cv2.cvtColor(template, cv2.COLOR_RGB2GRAY)

    # 显示直方图
    # plt.hist(img.ravel(), 256, [0, 256])
    # plt.show()

    ret, binary = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY)
    tpltret, tpltbinary = cv2.threshold(graytplt, 50, 255, cv2.THRESH_BINARY)
    # cv2.imshow("二值化图像", binary)
    # plt.figure()
    # plt.subplot(1,2,1)
    # plt.axis('off')
    # plt.imshow(binary, cmap='Greys_r')
    # plt.subplot(1,2,2)
    # plt.axis('off')
    # plt.imshow(tpltbinary, cmap='Greys_r')
    # print(binary)
    # print("二值图像已获取")

    # 进行chamfer transform
    chamfer_transform = CM.ChamferMatching.Chamfer_Dist_Transform(binary)
    # print(type(binary))
    # print(binary.shape)
    # print(chamfer_transform)

    # 计算chamfer distance
    CD = CM.ChamferMatching.Chamfer_Matching(tpltbinary, chamfer_transform)
    # print("Chamfer Distance: ", CD)
    return CD

# 遍历图片进行匹配
def walk_file(file):
    n = 0
    cloth_dirs = []
    # 判断图片文件
    is_image_file = lambda x : any(x.endswith(extension) for extension in ['.png', 'jpg', 'jpeg', '.PNG', '.JPG', '.JPEG'])
    # 进度条
    bar = Bar('Processing', max=109488, suffix='%(percent)d%%')
    for root, dirs, files in os.walk(file):
        # for d in dirs:
        #     cloth_dirs.append(os.path.join(root, d))
            # print(os.path.join(root, d))
        # print(files)
        for f in files:
            if(is_image_file(f)):
                # print(os.path.join(root, f))
                saveimg = cv2.imread(os.path.join(root, f))
                file_name = "/data/shijianyang/data/cloth/clothHD_" + str(n) + ".jpg"
                cv2.imwrite(file_name, saveimg)
                bar.next()
                n += 1
    bar.finish()
    # print(cloth_dirs[1])
    # for path in cloth_dirs[0: 10]:
    #     i = 1
    #     plt.figure()
    #     plt.subplot(1, 10, i)
    #     for root, dirs, files in os.walk(path):
    #         print(os.path.join(root, files[0]))
    #         img = cv2.imread(files[-1])
    #         plt.imshow(img)
    #         files_list.append(os.path.join(root, files[-1])) 
    #         print(os.path.join(root, files[-1]))
    #         files_list = []
    #         for f in files:
    #             if(is_image_file(f)):

    #                 files_list.append(os.path.join(root, f))
    #         print(files_list[-1])
    #         img = cv2.imread(files_list[-1])
    #         plt.imshow(img) 
    #     i += 1
    # print(len(cloth_dirs))
    print('------读取完毕，图片总数：%d------'%(n))

def write_csv(file, encoding, headers, rows):
    with open(file, 'w', encoding=encoding, newline='') as f:
        writer = csv.DictWriter(f, headers)
        writer.writeheader()
        writer.writerows(rows)

if __name__ == '__main__':
    path = "/data/shijianyang/data/sketch"
    headers = ["ID", "sketch", "template", "Chamfer_Distance"]
    id = 1
    rows = []
    # 进度条
    bar = Bar('Processing', max=109488, suffix='%(percent)d%%')
    for root, dirs, files in os.walk(path):
        for f in files:
            # print(os.path.join(root, f))
            sketchpath = os.path.join(root, f)
            templatepath = "/data/shijianyang/data/sketch/sketch1.png"
            # print(templatepath)
            # CD = match_sketch(sketchpath, templatepath)
            CD = match_sketch(templatepath, sketchpath)
            cd_row = dict(ID=id, sketch=sketchpath, template=templatepath, Chamfer_Distance=CD)
            # cd_row['ID'] = id
            # cd_row['sketch'] = sketchpath
            # cd_row['template'] = templatepath
            # cd_row['Chamfer_Distance'] = CD
            rows.append(cd_row)
            bar.next()
            id += 1
            if((id % 1000) == 0):
                write_csv('chamfer_matching_tshirts_exchange.csv', 'utf8', headers, rows)
                print("---------%d has been processed-------"%(id))
            # print(id)
    bar.finish()
    # print(len(rows))
    # for i in range(0, len(rows) - 1):
    #     print(rows[i])
    # write_csv('chamfer_matching_tshirts.csv', 'utf8', headers, rows)
    # walk_file("/data/shijianyang/data/data")
    # print(match_sketch("/data/shijianyang/data/sketch/sketch2.png", "/data/shijianyang/data/sketch/sketch1.png"))