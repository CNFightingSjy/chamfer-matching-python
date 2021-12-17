import csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cv2
from progress.bar import Bar

def normfun(x, mean ,sigma):
    pdf = np.exp(-((x - mean) ** 2) / (2 * sigma ** 2)) / (sigma * np.sqrt(2 * np.pi))
    return pdf

def cd_plot(cd):
    # print(cd)
    cd = cd
    mean = cd.mean()
    std = cd.std()

    # cd_min = cd.min()
    # cd_max = cd.max()

    x = np.arange(0.8, 1, 1)
    y = normfun(x, mean, std)

    plt.plot(x, y)
    plt.hist(cd, density=True)
    plt.show()

def get_sketch(data, threshold):
    sketch = data[data['Chamfer_Distance'] < threshold]
    return sketch

def split_imgname(namestr):
    strlist = namestr.split('/')
    return strlist[-1]

def filter_img(sketch, file):
    img = sketch['sketch'].values

    max = len(img)
    print("total: ", max)
    # print(len(tshirts_sketch['sketch'].values))
    bar = Bar('Processing', max=max, suffix='%(percent)d%%')
    for img in img:
        name = split_imgname(img)
        # print(name)
        path = file + name
        save = cv2.imread(img)
        cv2.imwrite(path, save)
        bar.next()
    bar.finish()
    # img_channel = 1
    # mean, std = [0.5 for _ in range(img_channel)], [0.5 for _ in range(img_channel)]
    # print(mean)
    # print(std)

if __name__=='__main__':
    # chamfer_distance = []
    # with open('chamfer_matching_tshirts.csv') as f:
    #     f_csv = csv.DictReader(f)
    #     for row in f_csv:
    #         chamfer_distance.append(row['Chamfer_Distance'])
    #         # print("chamfer_distance", row['Chamfer_Distance'])
    # data = pd.read_csv('chamfer_matching_tshirts.csv')
    data = pd.read_csv('chamfer_matching_tshirts_exchange.csv')
    cd_plot(data['Chamfer_Distance'])
    sketch = get_sketch(data, 0.86)
    filter_img(sketch, '/data/shijianyang/data/sketch/tshirts_ex/')



