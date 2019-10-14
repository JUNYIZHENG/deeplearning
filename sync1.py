import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import torchvision
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

def conv3x3(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3,
                     stride=stride, padding=1, bias=False)

class BasicBlock(nn.Module):
    def __init__(self,in_channels,out_channels,stride=1,downsample=None):
        super(BasicBlock,self).__init__()
        self.conv1 = conv3x3(in_channels, out_channels, stride)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(out_channels, out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample
        # self.layers = nn.Sequential(nn.Conv2d(in_channels,out_channels,kernel_size=3,stride=stride,padding=1,bias=False),
        #                             nn.BatchNorm2d(num_features=out_channels),
        #                             nn.ReLU(inplace=True),
        #                             nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1),
        #                             nn.BatchNorm2d(num_features=out_channels))
        # self.skip_connection = nn.Sequential()
        # # input channels != output channels, cannot add up F(x) + x
        # if (stride !=1 or in_channels != out_channels):
        #     self.skip_connection = nn.Sequential(nn.Conv2d(in_channels,out_channels,padding=1,stride=stride,bias=False),
        #                                          nn.BatchNorm2d(num_features=out_channels))

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, block,num_blocks):
        super(ResNet, self).__init__()
        self.in_channels = 32
        self.conv = conv3x3(3, 32)
        self.bn = nn.BatchNorm2d(32)
        self.dropout = nn.Dropout(p=0.5)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self.make_layer(block, 32, num_blocks[0])
        self.layer2 = self.make_layer(block, 64, num_blocks[1], 2)
        self.layer3 = self.make_layer(block, 128, num_blocks[2], 2)
        self.layer4 = self.make_layer(block, 256, num_blocks[3], 2)
        self.max_pool = nn.MaxPool2d(4,2)
        self.fc = nn.Linear(256,100)

    def make_layer(self, block, out_channels, num_blocks, stride=1):
        downsample = None
        if (stride != 1) or (self.in_channels != out_channels):
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels, kernel_size=1,
                     stride=stride, bias=False),
                nn.BatchNorm2d(out_channels))
        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels
        for i in range(1, num_blocks):
            layers.append(block(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.max_pool(out)
        # dim : (batchsize，channels，x，y) | x.size(0) = batchsize
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out


num_epochs = 20

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
train_loader = torch.utils.data.DataLoader(trainset,batch_size = 100, shuffle = True, num_workers = 0)
# For testing data
testset = torchvision.datasets.CIFAR100(root = './data',train = False,download = True,transform = transform_test)
test_loader = torch.utils.data.DataLoader(testset,batch_size = 100,shuffle = False, num_workers =0)

model = ResNet(BasicBlock,[2,4,4,2])



for param in model.parameters():
    tensor0 = param.data
    dist.all_reduce(tensor0, op=dist.reduce_op.SUM)
    param.data = tensor0/np.sqrt(np.float(num_nodes))

model.cuda()

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=.001)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer,step_size = 10,gamma = 0.5)

for epoch in range(1, num_epochs+1):
    print('epoch' + str(epoch))
    model.train()
    counter = 0
    for i, data in enumerate(train_loader, 0):
        inputs, labels = data
        labels = Variable(labels).cuda()
        optimizer.zero_grad()
        output = model(Variable(inputs).cuda())
        loss = criterion(output, labels)
        loss.backward()
        for param in model.parameters():
            # print(param.grad.data)
            tensor0 = param.grad.data.cpu()
            dist.all_reduce(tensor0, op=dist.reduce_op.SUM)
            tensor0 /= float(num_nodes)
            param.grad.data = tensor0.cuda()
        if (epoch > 6):
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

    model.eval()
    counter = 0
    test_accuracy_sum = 0.0
    correct = 0

    for data, target in test_loader:
        target = Variable(target).cuda()
        output = model(Variable(data).cuda())
        prediction = output.data.max(1)[1]
        accuracy = (float(prediction.eq(target.data).sum()) / float(batch_size)) * 100.0
        counter += 1
        test_accuracy_sum = test_accuracy_sum + accuracy
    test_accuracy_ave = test_accuracy_sum / float(counter)
    print('testing acc:{:.4f}'.format(test_accuracy_ave))

    scheduler.step()
