import torch
import numpy as np
import pandas as pd
import torch.utils.data as data
from tqdm import tqdm

def get_class(x):
    cut = x.split('_')
    if len(cut) == 2:
        return cut[0]+ '/' + x[:-1] 
    else:
        class_ = cut[0] + '_' + cut[1]
        return class_ + '/' + x[:-1]

class ModelNetDataset(data.Dataset):
    def __init__(self,
                 method = 'test',
                 class_num = 10,
                 npoints = 5000,
                 data_augmentation=True):
        self.method = method # train or test
        self.class_num = class_num # # 10 or 40
        self.npoints = npoints # number of npoints
        self.data_augmentation = data_augmentation 
        self.data_path = 'modelnet40_normal_resampled/modelnet' + str(class_num) + '_' + str(method) + '.txt' # dataset
        self.class_path = 'modelnet40_normal_resampled/modelnet' + str(class_num) + '_shape_names.txt'

        # 得到分类映射表
        self.class_dict = {}
        with open(self.class_path , 'r') as f1:
            for i , j in enumerate(f1):
                self.class_dict[j.rstrip('\n')] = i

        # 得到文件名
        with open(self.data_path, 'r') as f2:
            point_point = f2.readlines()
        point_point = list(map(lambda x : get_class(x), point_point))
        
        # 读取文件和label
        self.data = [] #存数据 , np.array
        self.label = [] #存标签, np.array
        for file in tqdm(point_point):
            point = 'modelnet40_normal_resampled/' + file + '.txt'
            point = pd.read_csv(point, header=None).iloc[:,0:3].to_numpy(dtype = np.float64)
            lable = self.class_dict[file.split('/')[0]]
            self.label.append(lable)
            self.data.append(point)

        # to_array
        # self.data = np.array(self.data)
        self.label = np.array(self.label)

    def __getitem__(self, id) :
        point = self.data[id]
        # point = pd.read_csv(point,header=None).to_numpy(dtype = np.float64)[:,0:3]
        cls = self.label[id]
        temp = np.zeros(self.class_num)
        temp[cls] = 1
        cls = temp

        #random sample points
        if self.method == 'train':
            choice = np.random.choice(len(point), self.npoints, replace=False)
            point_set = point[choice, :]
        else:
            point_set = point #测试的时候不需要加

        #scale
        point_set = point_set - np.expand_dims(np.mean(point_set, axis=0), 0)  # center
        dist = np.max(np.sqrt(np.sum(point_set ** 2, axis=1)), 0) # sqrt(x ** 2  + y ** 2 + z ** 2) 
        point_set = point_set / dist  # scale

        #data_augmentation
        if self.data_augmentation :
            theta = np.random.uniform(0, np.pi * 2)
            rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
            point_set[:, [0, 2]] = point_set[:, [0, 2]].dot(rotation_matrix)  # random rotation ? 绕着Y轴转?
            point_set += np.random.normal(0, 0.02, size=point_set.shape)  # Gaussian jitter jitter
        
        np.random.shuffle(point_set) #shuffle points
        point_set = torch.from_numpy(point_set.astype(np.float32))
        point_set = point_set.transpose(0 , 1)
        cls = torch.from_numpy(np.array(cls).astype(np.float32))
        return point_set, cls

    def __len__(self):
        return len(self.data)


if __name__ == '__main__':
    train = ModelNetDataset()
    print(train[1])

