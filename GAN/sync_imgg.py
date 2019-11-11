import numpy as np
import torch
import torchvision
from torchvision import models, transforms
from time import *
import torch.nn as nn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
from torch import autograd
from torch.autograd import Variable

class discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=196, kernel_size=3, stride=1, padding=1),
            nn.LayerNorm(normalized_shape=[196, 32, 32]),
            nn.LeakyReLU(0.02, inplace=True)
            )
        self.conv2 = nn.Sequential(
            nn.Conv2d(196, 196, 3, 2, 1),
            nn.LayerNorm(normalized_shape=[196, 16, 16]),
            nn.LeakyReLU(0.02, inplace=True))
        self.conv3 = nn.Sequential(
            nn.Conv2d(196, 196, 3, 1, 1),
            nn.LayerNorm(normalized_shape=[196, 16, 16]),
            nn.LeakyReLU(0.02, inplace=True))
        self.conv4 = nn.Sequential(
            nn.Conv2d(196, 196, 3, 2, 1),
            nn.LayerNorm(normalized_shape=[196, 8, 8]),
            nn.LeakyReLU(0.02, inplace=True))
        self.conv5 = nn.Sequential(
            nn.Conv2d(196, 196, 3, 1, 1),
            nn.LayerNorm(normalized_shape=[196, 8, 8]),
            nn.LeakyReLU(0.02, inplace=True))
        self.conv6 = nn.Sequential(
            nn.Conv2d(196, 196, 3, 1, 1),
            nn.LayerNorm(normalized_shape=[196, 8, 8]),
            nn.LeakyReLU(0.02, inplace=True))
        self.conv7 = nn.Sequential(
            nn.Conv2d(196, 196, 3, 1, 1),
            nn.LayerNorm(normalized_shape=[196, 8, 8]),
            nn.LeakyReLU(0.02, inplace=True))
        self.conv8 = nn.Sequential(
            nn.Conv2d(196, 196, 3, 2, 1),
            nn.LayerNorm(normalized_shape=[196, 4, 4]),
            nn.LeakyReLU(0.02, inplace=True))
        self.pool = nn.MaxPool2d(kernel_size=4)
        self.fc1 = nn.Sequential(
            nn.Linear(196, 1))
        self.fc10 = nn.Sequential(
            nn.Linear(196, 10)
            )
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7(x)
        x = self.conv8(x)
        x = self.pool(x)
        x = x.view(-1, x.size()[1]*x.size()[2]*x.size()[3])
        fc1_out = self.fc1(x)
        fc10_out = self.fc10(x)
        return fc1_out, fc10_out

def plot(samples):
    fig = plt.figure(figsize=(10, 10))
    gs = gridspec.GridSpec(10, 10)
    gs.update(wspace=0.02, hspace=0.02)

    for i, sample in enumerate(samples):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        plt.imshow(sample)
    return fig


transform_test = transforms.Compose([
    transforms.CenterCrop(32),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

batch_size = 128
testset = torchvision.datasets.CIFAR10(root='./', train=False, download=False, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=0)
testloader = enumerate(testloader)

batch_idx, (X_batch, _) = testloader.__next__()
X_batch = Variable(X_batch,requires_grad=True).cuda()

X = X_batch.mean(dim=0)
X = X.repeat(10,1,1,1)

Y = torch.arange(10).type(torch.int64)
Y = Variable(Y).cuda()


model = torch.load('discriminator.model')
model.cuda()
model.eval()

lr = 0.1
weight_decay = 0.001
for i in range(200):
    _, output = model(X)

    loss = -output[torch.arange(10).type(torch.int64),torch.arange(10).type(torch.int64)]
    gradients = torch.autograd.grad(outputs=loss, inputs=X,
                              grad_outputs=torch.ones(loss.size()).cuda(),
                              create_graph=True, retain_graph=False, only_inputs=True)[0]

    prediction = output.data.max(1)[1] # first column has actual prob.
    accuracy = ( float( prediction.eq(Y.data).sum() ) /float(10.0))*100.0
    print(i,accuracy,-loss)

    X = X - lr*gradients.data - weight_decay*X.data*torch.abs(X.data)
    X[X>1.0] = 1.0
    X[X<-1.0] = -1.0

## save new images
samples = X.data.cpu().numpy()
samples += 1.0
samples /= 2.0
samples = samples.transpose(0,2,3,1)

fig = plot(samples)
plt.savefig('visualization/max_class_withgen.png', bbox_inches='tight')
plt.close(fig)
