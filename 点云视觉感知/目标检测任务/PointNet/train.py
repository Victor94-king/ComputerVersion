import torch 
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import numpy as np 
import pandas as pd 
from tqdm import tqdm 
import torch.nn.functional as F
from torchsummary import summary
import random
import os
import matplotlib.pyplot as plt
from dataset import ModelNetDataset
from model import PointNet
import argparse

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_args():
    parase = argparse.ArgumentParser(description='Simple PointNet')
    parase.add_argument("--checkpoint", type = str , default = None ,  help= "Pretrained model")
    parase.add_argument("--output_dir", type = str, default = './checkpoint/' , help = "hash table size")
    parase.add_argument("--epoch", type = int, default = 50 , help = "epoch")
    parase.add_argument("--batch_size", type = int, default = 32 , help = "batch_size")
    parase.add_argument("--class_num", type = int, default = 10 , help = "cls_num")
    parase.add_argument("--npoints", type= int, default = 5000 , help = "Ramdom Dowmsample")
    parase.add_argument('--lr', type = float, default = 1e-3 , help = "learning rate")
    parase.add_argument('--print_log', type = int , default= 40  , help = "nums of print logs")
    parase.add_argument('--seed', type = int , default= 0  , help = "random seed")
    parase.add_argument("--save_epoch", type = int, default = 5 , help = "iter of save checkpoints")
    return parase.parse_args()

def seed_torch(seed):
	random.seed(seed)
	os.environ['PYTHONHASHSEED'] = str(seed) # 为了禁止hash随机化，使得实验可复现
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed(seed)
	torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
	torch.backends.cudnn.benchmark = False
	torch.backends.cudnn.deterministic = True


def test(model, test_loader, criterion , iter):
    print('Testing model!')
    model.eval()
    correct, test_loss  = 0 ,0
    with torch.no_grad():
        for data, target in tqdm(test_loader):
            data, target = data.to(device), target.to(device)
            
            output = model(data)
            test_loss += criterion(output,target).item() # 
            _ , pred = torch.max(output , 1) #返回预测对的位置
            correct += (pred == torch.argmax(target, dim = 1)).sum() #每个batch预测对的数量累加
            
    test_loss /= len(test_loader)
    test_acc = 100. * correct / len(test_loader.dataset)
    print('Test set of iter {}: Average loss: {:.4f}, Accuracy: {}/{} ({:.1f} %)\n'.format(
      iter , test_loss, correct, len(test_loader.dataset), test_acc ) )
    return test_loss , round(test_acc.item(),2)


def plot_loss_and_acc(loss_and_acc_dict):
	fig = plt.figure()
	tmp = list(loss_and_acc_dict.values())
	maxEpoch = len(tmp[0][0])
	stride = np.ceil(maxEpoch / 10)

	maxLoss = max([max(x[0]) for x in loss_and_acc_dict.values()]) + 0.1
	minLoss = max(0, min([min(x[0]) for x in loss_and_acc_dict.values()]) - 0.1)

	for name, lossAndAcc in loss_and_acc_dict.items():
		plt.plot(range(0, maxEpoch), lossAndAcc[0], '-.', label=name)

	plt.xlabel('Epoch')
	plt.ylabel('Loss')
	plt.legend()
	plt.xticks(range(0, maxEpoch + 1, 2))
	plt.axis([0, maxEpoch, minLoss, maxLoss])
	plt.show()
	maxAcc = min(1, max([max(x[1]) for x in loss_and_acc_dict.values()]) + 0.1)
	minAcc = max(0, min([min(x[1]) for x in loss_and_acc_dict.values()]) - 0.1)

	fig = plt.figure()
	for name, lossAndAcc in loss_and_acc_dict.items():
		plt.plot(range(0, maxEpoch), lossAndAcc[1], '-.', label=name)

	plt.xlabel('Epoch')
	plt.ylabel('Accuracy')
	plt.xticks(range(0, maxEpoch + 1, 2))
	plt.legend()
	plt.show()


def main(epoch ,batch_size , class_num , npoints , lr , print_log , save_epoch , output_dir ):
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    train_set = ModelNetDataset(method= 'train' , class_num = class_num , npoints = npoints , data_augmentation = True) 
    test_set = ModelNetDataset(method= 'test' , class_num = class_num , npoints = npoints , data_augmentation = False) 
    train_dataloader = torch.utils.data.DataLoader(train_set, batch_size = batch_size, shuffle=True,)
    test_dataloader = torch.utils.data.DataLoader(test_set, batch_size = batch_size, shuffle=True,)

    #loading model
    model = PointNet(class_num = class_num)
    if checkpoint:
        print('loading model')
        model.load_state_dict(torch.load(checkpoint))
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters() , lr = lr )
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size = 20, gamma = 0.5)  #每过20个epoch, lr 变一半
    criterion = nn.CrossEntropyLoss()

    #保留每个epoch的loss和acc
    Train_loss , Train_acc , Test_loss , Test_acc = [] , [] , [] , [] 
    for it in range(1 , epoch + 1):
        model = model.train()
        iteration_train_loss, iteration_train_acc = [], [] #保留每个batch的loss
        for batch_idx, ( data , target) in enumerate(train_dataloader):
            data , target = data.to(device) , target.to(device) 
            optimizer.zero_grad() #重置梯度
            output = model(data) #得到最终输出
            loss = criterion(output ,target) #计算交叉熵损失
            loss.backward() #梯度计算
            optimizer.step() #更新梯度
        
            #计算acc
            pred = output.max(1, keepdim = True)[1].squeeze() # softmax 
            correct = pred.eq(torch.argmax(target, dim = 1)).cpu().sum() #计算准确率
            acc = correct.item() / float(batch_size) * 100

            #输出loss
            if batch_idx % print_log == 0:
                print( f"Train Epoch: {it} , [{batch_idx * len(data)}/ {len(train_dataloader.dataset)}]\t Loss: {loss.item():.4f} acc: {acc} %")
            
            loss_ = loss.cpu()
            iteration_train_loss.append(loss_.detach().numpy())
            iteration_train_acc.append(np.array(acc))
        
        scheduler.step() #在epoch后面更新学习率！！！！# 打印当前学习率
        print('Train set of iter {}: Learing rate{:.4f} Average loss: {:.4f}, Accuracy: ({:.1f} %) \n'.format(
            it ,optimizer.state_dict()['param_groups'][0]['lr'] , np.mean(iteration_train_loss), np.mean(iteration_train_acc),) )
        it_Testloss , it_Testacc = test(model , test_dataloader , criterion , it)
        Train_loss.append(np.mean(iteration_train_loss)) , Train_acc.append(np.mean(iteration_train_acc)) , Test_loss.append(it_Testloss) , Test_acc.append(it_Testacc) 

        #save model
        if it % save_epoch == 0:
            file_dir = output_dir + 'model_%d.pth' % (it)
            torch.save(model.state_dict(), file_dir )
            print("model saved in {}".format(file_dir))
            
    plot_loss_and_acc(
        {
            'Train': [Train_loss ,Train_acc ],
            'Test': [Test_loss ,Test_acc]
        }
    )

if __name__ == '__main__':
    args = get_args()
    seed = args.seed
    epoch = args.epoch
    batch_size = args.batch_size
    class_num = args.class_num
    npoints = args.npoints
    lr = args.lr
    print_log = args.print_log
    checkpoint = args.checkpoint
    save_epoch = args.save_epoch
    output_dir = args.output_dir

    seed_torch(seed) # 设置下随机种子
    main(epoch ,batch_size , class_num , npoints , lr , print_log , save_epoch , output_dir)

