import numpy as np
import open3d as o3d
import pandas as pd
import matplotlib.pyplot as plt

def read_file(path):
    """read path to get pointcloud & np.array data"""
    np_data = pd.read_csv(path,header=None).to_numpy(dtype = np.float64)
    pc = o3d.geometry.PointCloud() 
    pc.points = o3d.utility.Vector3dVector(np_data[:,0:3])
    # o3d.visualization.draw_geometries([pc], window_name = "Original Cloud")
    # print("original points: %d"%np_data.shape[0])
    return np_data , pc 

class PFH(object):
    def __init__(self , radius = 0.05 , B = 10) -> None:
        self.radius = radius
        self.B = B

    def cal_group(self , idx_neighbors , id):
        # idx_neighbors.append(id) #把quiry点加入到整个近邻里
        alpha_group ,phi_group , theta_group = [] ,[] , []

        for id1 in idx_neighbors:
            for id2 in idx_neighbors:
                if id1 == id2:#跳过自己本身
                    continue
                afa , fai , the = self.cal_pair(id1, id2)

                alpha_group.append(afa)
                phi_group.append(fai)
                theta_group.append(the) 
        # print(len(alpha_group) , len(phi_group) , len(theta_group))

        alpha_hist , _ = np.histogram(alpha_group , self.B )
        phi_hist , _   = np.histogram(phi_group   , self.B )
        theta_hist , _ = np.histogram(theta_group , self.B )

        feature = np.hstack((alpha_hist , phi_hist , theta_hist)) / len(alpha_group) #由于每个query的点的近邻个数不一样所以需要归一化
        return feature


    def cal_pair(self , id1 , id2):
        p1 = self.array[id1]
        p2 = self.array[id2]
        n1 = self.normal[id1]
        n2 = self.normal[id2]
        dis = (p2 - p1) / np.linalg.norm((p2 - p1) , ord = 2 )
        
        #计算坐标系 u , v , w ，
        u = n1 
        v = np.cross(u , dis)
        w = np.cross(u ,  v)
        
        #计算其3个夹角alpha , phi , theta
        afa = np.dot(v , n2)
        fai = np.dot(u , (p2 - p1) / sum((p2 - p1) ** 2))
        the = np.arctan2((w * n2),(u * n2))

        return afa , fai , the
        

    def describe(self, point_cloud:o3d.geometry.PointCloud , feature_id:list):
        self.array = point_cloud[0][:,:3]
        self.normal = point_cloud[0][:, 3:]
        self.tree = o3d.geometry.KDTreeFlann(point_cloud[1])

        self.feature_list = []

        for id in feature_id:
            [num_neighbors, idx_neighbors , dis_neighbors] = self.tree.search_radius_vector_3d(self.array[id],self.radius)

            #计算当前query 的特征描述，并返回
            feature = self.cal_group(idx_neighbors , id) 
            self.feature_list.append(feature)

        return self.feature_list
    


class FPFH(object):
    def __init__(self , radius = 0.05 , B = 10) -> None:
        self.radius = radius
        self.B = B

    def cal_SPFH(self , id:int):
        [num_neighbors, idx_neighbors , dis_neighbors] = self.tree.search_radius_vector_3d(self.array[id],self.radius)
        query = self.array[id]
        neigbhors = self.array[idx_neighbors[1:]]
        
        #计算n1 , n2 ,以及dis
        n1 = self.normal[id]
        n2 = self.normal[idx_neighbors[1:]]
        dis = (neigbhors - query) / np.linalg.norm((neigbhors - query) , ord = 2)

        #计算u , v , w
        u = self.normal[id]
        v = np.cross(u , dis)
        w = np.cross(u , v)

        #计算其3个夹角alpha , phi , theta
        afa = (v * n2).sum(axis = 1)
        fai = (u * dis).sum(axis = 1)
        the = np.arctan2((w * n2).sum(axis = 1),(u * n2).sum(axis = 1))
        
        #计算alpah , phi , theta 的直方图
        alpha_hist , _ = np.histogram(afa , self.B)
        phi_hist , _   = np.histogram(fai , self.B)
        theta_hist , _ = np.histogram(the , self.B)

        spfh = np.hstack((alpha_hist , phi_hist , theta_hist)) / (num_neighbors - 1) #由于每个query的点的近邻个数不一样所以需要归一化
        return spfh


    def describe(self, point_cloud:o3d.geometry.PointCloud , feature_id:list):
        self.array = point_cloud[0][:,:3]
        self.normal = point_cloud[0][:, 3:]
        self.tree = o3d.geometry.KDTreeFlann(point_cloud[1])

        self.feature_list = []

        for id in feature_id:
            #计算query点的SPFH特征
            query_SPFH = self.cal_SPFH(id)

            #计算query近邻的SPFH特征
            [k, idx_neighbors , dis_neighbors] = self.tree.search_radius_vector_3d(self.array[id],self.radius) #计算query点的近邻
            weight = 1.0 / np.linalg.norm(self.array[id] - self.array[idx_neighbors[1:]] , ord = 2 , axis = 1) # 权重
            neighbors_SPFH = np.array([self.cal_SPFH(i) for i in idx_neighbors[1:]]) # 近邻的SPFH
            
            #计算query点的FPFH，并归一化
            query_FPFH = query_SPFH + np.dot(weight.reshape((1,-1)) ,neighbors_SPFH).reshape(-1)  / (k - 1)
            query_FPFH = query_FPFH / np.linalg.norm(query_FPFH)
            
            self.feature_list.append(query_FPFH)
        return self.feature_list


def plot_fea(B , feature_list , title):
    for i , fea in enumerate(feature_list):
        plt.plot(range(3*B), fea, ls="-.",marker =",", lw=2, label=f"keypoint{i+1}")
    
    plt.title(title)
    plt.legend()
    plt.show()

def describe(descritor,  point_cloud  ,  feature_id , unfimilar_id , B):
    #初始化描述器
    feature_list = descritor.describe(point_cloud , feature_id)
    unfimilar_fea = descritor.describe(point_cloud , unfimilar_id)

    #plot the feature
    plot_fea(B , feature_list , 'similar')
    # plot_fea(B , unfimilar_fea, 'Unsimilar')

def main():
    ##参数设置
    path = "chair_0001.txt"
    feature_id   = [717 , 3267] #这个id是上一张特征点ISS detection 找到的
    unfimilar_id = [717 , 9999] #同样随机找了一个点，看看特征描述

    B = 10
    radius = 0.05
    point_cloud = read_file(path) # 得到所有点云数据

    #pfh 
    pfh = PFH(B = B , radius= radius)
    describe(pfh,  point_cloud  ,  feature_id , unfimilar_id , B)

    #fpfh
    fpfh = FPFH(B = B , radius= radius)
    describe(fpfh,  point_cloud  ,  feature_id , unfimilar_id , B)
    

if __name__ == "__main__":
    main()
    