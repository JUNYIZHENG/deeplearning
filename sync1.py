import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import torchvision
#from PIL import Image
import torch.distributed as dist
import torchvision.transforms as transforms
import os
import subprocess
from mpi4py import MPI
import csv



cmd = "/sbin/ifconfig"
out, err = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE,stderr=subprocess.PIPE).communicate()
ip = str(out).split("inet addr:")[1].split()[0]

name = MPI.Get_processor_name()
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
num_nodes = int(comm.Get_size())

ip = comm.gather(ip)

if rank != 0:
  ip = None

ip = comm.bcast(ip, root=0)

os.environ['MASTER_ADDR'] = ip[0]
os.environ['MASTER_PORT'] = '2222'

backend = 'mpi'
dist.init_process_group(backend, rank=rank, world_size=num_nodes)

dtype = torch.FloatTensor

def conv3x3(in_planes,out_planes,stride = 1):
    return nn.Conv2d(in_planes,out_planes,kernel_size=3,stride = stride,padding = 1,bias = False)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self,in_channels,out_channels,stride =1,downsample = None):
        super(BasicBlock,self).__init__()
        self.conv1 = conv3x3(in_channels,out_channels,stride = stride)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.conv2 = conv3x3(out_channels,out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample
        self.stride = stride

    def forward(self,x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        # print('downsample',self.downsample)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out


class ResNet(nn.Module):

    def __init__(self,basic_block,num_block,num_classes):
        self.inplanes = 32
        super(ResNet,self).__init__()
        self.conv1 = nn.Conv2d(3,32,kernel_size=3,stride = 1,padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu1 = nn.ReLU()
        self.drop1 = nn.Dropout(0.2)
        self.conv2 = self._make_layer(basic_block,32,num_block[0],stride = 1)
        self.conv3 = self._make_layer(basic_block,64,num_block[1],stride = 2)
        self.conv4 = self._make_layer(basic_block,128,num_block[2],stride = 2)
        self.conv5 = self._make_layer(basic_block,256,num_block[3],stride= 2)
        self.maxpool1 = nn.MaxPool2d(3,stride = 2)
        self.fc = nn.Linear(256*basic_block.expansion,num_classes)


    def _make_layer(self,block,planes,blocks,stride = 1):
        downsample = None
        # print('expansion',block.expansion)
        if stride != 1 or self.inplanes != planes * block.expansion:
            # print('inplanes',planes*block.expansion)
            downsample = nn.Sequential(nn.Conv2d(self.inplanes,planes*block.expansion,kernel_size = 1,stride = stride, bias = False),
                                       nn.BatchNorm2d(planes*block.expansion))
        layers = []
        layers.append(block(self.inplanes,planes,stride,downsample))
        self.inplanes = planes*block.expansion
        for i in range(1,blocks):
            layers.append(block(self.inplanes,planes))
        return nn.Sequential(*layers)



    def forward(self,x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.drop1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.maxpool1(x)
        # print(" x shape ", x.size())
        x = x.view(x.size(0),-1)
        x = self.fc(x)
        return x

def train(model):
    model.train()
    counter = 0
    train_accuracy_sum = 0.0
    for i, data in enumerate(train_loader, 0):
        inputs, labels = data
        labels = Variable(labels).cuda()
        optimizer.zero_grad()
        output =  model(Variable(inputs).cuda())
        loss = criterion(output, labels)
        loss.backward()
        for param in model.parameters():
            #print(param.grad.data)
            tensor0 = param.grad.data.cpu()
            dist.all_reduce(tensor0, op=dist.reduce_op.SUM)
            tensor0 /= float(num_nodes)
            param.grad.data = tensor0.cuda()
        if (epoch > 1):
            for group in optimizer.param_groups:
                for p in group['params']:
                    state = optimizer.state[p]
                    if 'step' in state.keys():
                        if (state['step'] >= 1024):
                            state['step'] = 1000
        optimizer.step()
        prediction = output.data.max(1)[1]
        accuracy = (float(prediction.eq(labels.data).sum()) / float(batch_size)) * 100.0
        counter += 1
        train_accuracy_sum = train_accuracy_sum + accuracy
    train_accuracy_ave = train_accuracy_sum / float(counter)
    print('training acc:{:.4f}'.format(train_accuracy_ave))
    return train_accuracy_ave


def test(model, test_loader):
    model.eval()
    counter = 0
    test_accuracy_sum = 0.0
    correct = 0

    for data, target in test_loader:
        target = Variable(target).cuda()
        output =  model(Variable(data).cuda())
        prediction = output.data.max(1)[1]
        accuracy = (float(prediction.eq(target.data).sum()) / float(batch_size)) * 100.0
        counter += 1
        test_accuracy_sum = test_accuracy_sum + accuracy
    test_accuracy_ave = test_accuracy_sum / float(counter)
    print('testing acc:{:.4f}'.format(test_accuracy_ave))
    return test_accuracy_ave

batch_size = 128
num_epochs = 20
learning_rate = 0.001
#read data
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

# For training data
trainset = torchvision.datasets.CIFAR100(root = './data',train = True,download = True,transform = transform_train)
train_loader = torch.utils.data.DataLoader(trainset,batch_size = batch_size, shuffle = True, num_workers = 0)
# For testing data
testset = torchvision.datasets.CIFAR100(root = './data',train = False,download = True,transform = transform_test)
test_loader = torch.utils.data.DataLoader(testset,batch_size = batch_size,shuffle = False, num_workers =0)

model = ResNet(BasicBlock,[2,4,4,2],100)



for param in model.parameters():
    tensor0 = param.data
    dist.all_reduce(tensor0, op=dist.reduce_op.SUM)
    param.data = tensor0/np.sqrt(np.float(num_nodes))

model.cuda()

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer,step_size = 10,gamma = 0.5)

for epoch in range(num_epochs):
    train_acc = train(model)
    scheduler.step()
    test_acc = test(model, test_loader)

