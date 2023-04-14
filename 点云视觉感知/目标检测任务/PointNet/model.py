
import torch
import torch.nn as nn
import torch.utils.data
import torch.nn.functional as F


class GlobalFeatures(nn.Module):
    def __init__(self) -> None:
        super(GlobalFeatures , self).__init__()
        #考虑的全局特征所以用Conv
        self.conv1 = nn.Sequential(nn.Conv1d(3 , 64 , 1) , nn.BatchNorm1d(64) , nn.ReLU() )
        self.conv2 = nn.Sequential(nn.Conv1d(64 , 128 , 1) , nn.BatchNorm1d(128) , nn.ReLU())
        self.conv3 = nn.Sequential(nn.Conv1d(128 , 1024 , 1) , nn.BatchNorm1d(1024) , nn.ReLU()) 

    def forward(self, x):
        x = self.conv1(x) # B * 64 * N
        x = self.conv2(x) # B * 128 * N
        x = self.conv3(x) # B * 512 * N
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.squeeze() 
        return x


class PointNet(nn.Module):
    def __init__(self , class_num ) -> None:
        super(PointNet , self).__init__()
        self.class_num = class_num
        self.globalExtra = GlobalFeatures()
        self.fc1 = nn.Sequential( nn.Linear(1024 , 512) , nn.BatchNorm1d(512) ,nn.ReLU())
        self.fc2 = nn.Sequential( nn.Linear(512 , 256) , nn.Dropout(0.3), nn.BatchNorm1d(256) , nn.ReLU())       
        self.fc3 = nn.Sequential( nn.Linear(256 , class_num)  , nn.Softmax(dim=1) )
    
    def forward(self, x):
        x = self.globalExtra(x)  # B * 1024 
        x = self.fc1(x) # B * 512 
        x = self.fc2(x) # B * 256
        output = self.fc3(x) # B * K
        return output


