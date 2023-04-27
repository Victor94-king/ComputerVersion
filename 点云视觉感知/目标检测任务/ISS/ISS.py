import numpy as np
import pandas as pd
import open3d as o3d

eps = 1e-6

class ISS(object):
    def __init__(self, pc , radius = 0.1, gama21 = 0.7 , gama32 =0.7 , min_lambda3 = 0.0005, min_neighbors = None):
        self.pc = pc #o3dpc
        self.tree = o3d.geometry.KDTreeFlann(pc) # KDTree
        self.point_clound = np.asarray(pc.points) 
        self.num_points = self.point_clound.shape[0]
        self.radius = radius # 计算特征点的直径
        self.gama21 = gama21
        self.gama32 = gama32
        self.min_lambda3 = min_lambda3
        self.min_neighbors = min_neighbors if not min_neighbors else 0.01 * self.num_points #为了筛选噪声和特征点，引入min_negihgbors
        self.radius_neighbor = [] #用于存储每个点的邻居，在NMS的时候用就不必再查了
        self.points_eigns = {
            'id' : [],
            'l1' : [],
            'l2' : [],
            'l3' : [],
                             } # save lambda3 values for each points


    def cal_eigen(self , point , idx_neighbors , w):
        #计算根据记录的加权的radius近邻协方差矩阵
        w = 1 / np.array(w) # K
        dis = self.point_clound[idx_neighbors] - point #每个点距离query_point的相对距离 , K * 3
        conv = np.dot((dis.T * w), dis) / w.sum() # 与其相对距离成反比，与其周围点的个数成反比
        #计算特征值
        values, v = np.linalg.eig(conv)
        values  = values[np.argsort(values)[::-1]]
        return values


    def detect(self):
        num_Neighbors_group = [-1] * self.num_points #用于记录每个点的radiusNN的个数，在计算权重的时候用

        #遍历每个点，对其计算加权的协方差矩阵
        for id , point in enumerate(self.point_clound):
            [num_neighbors, idx_neighbors , dis_neighbors] = self.tree.search_radius_vector_3d(point, self.radius)
            # print(len(idx_neighbors) == num_neighbors , num_neighbors == len(dis_neighbors))
            num_Neighbors_group[id] = num_neighbors
            self.radius_neighbor.append(idx_neighbors)

            if not num_neighbors and num_neighbors < self.min_neighbors: #说明是噪声，就直接跳过
                continue
            
            #遍历其近邻，然后计算其近邻的W权重
            w = []
            for idx in idx_neighbors:
                if num_Neighbors_group[idx] == -1: #之前没有被计算过
                    [a,__,___] = self.tree.search_radius_vector_3d(self.point_clound[idx], self.radius)
                    num_Neighbors_group[idx] = a
                w.append(num_Neighbors_group[idx])

            #计算Radius近邻的特征向量
            v = self.cal_eigen(point , idx_neighbors , w)
            # lambda 必须要足够大
            if self.min_lambda3 < v[2]:
                self.points_eigns['id'].append(id)
                self.points_eigns['l1'].append(v[0])
                self.points_eigns['l2'].append(v[1])
                self.points_eigns['l3'].append(v[2])
        
        # 还没有作过滤的时候可视下
        # self.vis_features()

        #对特征点进行清洗
        self.filter_points()
        # self.vis_features()
        # print(pd.Series(self.points_eigns['l3']).describe())

        #进行NMS
        self.NMS()
        self.vis_features()
        return self.point_clound[self.points_eigns['id']]

    def filter_points(self):
        '''留下满足条件的
        1. lambda2 / lambda1 < gama21
        2. lambda3 / lambda2 < gama32
        3. lambda1 > lambda2 > lambda3
        '''
        tmp = pd.DataFrame(self.points_eigns)
        start = tmp.shape[0]
        tmp['gama21'] = tmp['l2'] / tmp['l1']
        tmp['dev21'] = tmp['l1'] - tmp['l2']
        tmp['gama32'] = tmp['l3'] / tmp['l2']
        tmp['dev32'] = tmp['l2'] - tmp['l3']
        tmp =  tmp[ (tmp['gama21'] > self.gama21) & (tmp['gama32'] > self.gama32) & (tmp['dev21'] > eps) & (tmp['dev32'] > eps)]
        self.points_eigns['id'] = tmp['id']
        self.points_eigns['l1'] = tmp['l1']
        self.points_eigns['l2'] = tmp['l2']
        self.points_eigns['l3'] = tmp['l3']
        end = len(tmp['id'])
        print("Filter points: %d"%(start - end))

    def NMS(self):
        #对特征点按照lambda3的值进行排序
        tmp = pd.DataFrame(self.points_eigns)
        start = len(tmp)
        tmp.sort_values('l3' ,ascending=False, inplace=True)
        tmp = tmp['id'].to_numpy().tolist()

        res = [] #用于保存最后的id
        # print(len(self.radius_neighbor))
        while (tmp):
            #取出lambda3最大的点,并加入结果中，还需要删除掉自身
            query_id = tmp[0] 
            res.append(query_id)
            tmp.remove(query_id)

            #找到query点的Rnn近邻，如果出现在tmp中全部删除
            rnn_list = self.radius_neighbor[query_id]
            jiaoji = set(tmp) & set(rnn_list) 

            for i in list(jiaoji):
                tmp.remove(i)
        
        #更新下特征点的id
        self.points_eigns['id'] = res
        end = len(res)
        print("NMS: %d"%( start - end))
        

    def vis_features(self):
        pc = o3d.geometry.PointCloud() 
        pc.points = o3d.utility.Vector3dVector(self.point_clound[self.points_eigns['id']])
        o3d.visualization.draw_geometries([pc], window_name = "Feature Point Cloud")
        print("FeaturesPoints: %d" %len(self.points_eigns['id']))
        

def read_file(path):
    """read path to get pointcloud & np.array data"""
    np_data = pd.read_csv(path,header=None).to_numpy(dtype = np.float64)[:,0:3]
    pc = o3d.geometry.PointCloud() 
    pc.points = o3d.utility.Vector3dVector(np_data)
    o3d.visualization.draw_geometries([pc], window_name = "Original Cloud")
    print("original points: %d"%np_data.shape[0])
    return np_data , pc 

def main():
    file = r"modelnet40_normal_resampled/person/person_0001.txt"
    print(file)
    np_data , pc = read_file(file)
    detector = ISS(pc)
    feature_point = detector.detect()

if __name__ == '__main__':
    main()