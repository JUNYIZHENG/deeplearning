import torch
import torch.utils.data
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
##
num_epochs = 100
LR = 0.001
scheduler_step_size = 20
scheduler_gamma = 0.1

# data augmentation
train_augment = transforms.Compose([transforms.RandomCrop(32,4),transforms.RandomHorizontalFlip(),transforms.ToTensor(),
                                    transforms.Normalize((0.4914,0.48216,0.44653),(0.24703, 0.24349, 0.26159)), ])
test_norm = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.4914,0.48216,0.44653),(0.24703, 0.24349, 0.26159)), ])

# load training set & test set
train_set = torchvision.datasets.CIFAR10(root = '~/scratch/',train=True,transform=train_augment,download = True)
test_set = torchvision.datasets.CIFAR10(root = '~/scratch/',train=False,transform=test_norm,download = True)
# how many subprocesses to use for data loading. 0 means that the data will be loaded in the main process.
train_loader = torch.utils.data.DataLoader(train_set, batch_size=128,shuffle=True, num_workers=2)
testloader = torch.utils.data.DataLoader(test_set, batch_size=64, shuffle=False, num_workers=2)
classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')

# construct CNN
class cnn(nn.Module):
    def _init_(self):
        super(cnn,self).__init__()
        # convolution layers : batch norm, maxpool, dropout
        self.conv1 = nn.Sequential(nn.Conv2d(in_channels=3,out_channels=32,kernel_size=3,padding=1),
                                   nn.BatchNorm2d(32),nn.ReLU(inplace=True))
        self.conv2 = nn.Sequential(nn.Conv2d(in_channels=32,out_channels=64,kernel_size=3,padding=1),nn.BatchNorm2d(64),
                                   nn.ReLU(inplace=True),nn.MaxPool2d(kernel_size=2,stride=2),nn.Dropout2d(p=0.25))
        self.conv3 = nn.Sequential(nn.Conv2d(in_channels=64,out_channels=128,kernel_size=3,padding=2),
                                   nn.BatchNorm2d(128),nn.ReLU(inplace=True))
        self.conv4 = nn.Sequential(nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=2),
                                   nn.BatchNorm2d(256),nn.ReLU(inplace=True),nn.MaxPool2d(kernel_size=2,stride=2),
                                   nn.Dropout2d(p=0.35))
        self.conv5 = nn.Sequential(nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=2),
                                   nn.BatchNorm2d(256), nn.ReLU(inplace=True))
        self.conv6 = nn.Sequential(nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1),
                                   nn.BatchNorm2d(512), nn.ReLU(inplace=True), nn.MaxPool2d(kernel_size=2, stride=2),
                                   nn.Dropout2d(p=0.45))
        # fully connected layers
        # dim : 512 * 3 * 3
        self.fc1 = nn.Sequential(nn.Linear(in_features=4608,out_features=576),nn.ReLU(inplace=True))
        self.fc2 = nn.Sequential(nn.Linear(in_features=576,out_features=288),nn.ReLU(inplace=True))
        self.fc3 = nn.Sequential(nn.Linear(in_features=288,out_features=10),nn.Softmax(dim=1))

    # forward propagate
    def forward(self,l):
        l = self.conv1(l)
        l = self.conv2(l)
        l = self.conv3(l)
        l = self.conv4(l)
        l = self.conv5(l)
        l = self.conv6(l)
        l = l.view(-1,4608)
        l = self.fc1(l)
        l = self.fc2(l)
        l = self.fc3(l)
        return l

# CUDA tensor types,utilize GPUs for computation
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = cnn().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(),lr=LR)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer,step_size=scheduler_step_size,gamma=scheduler_gamma)

# train model
for epoch in range(num_epochs):
    print('epoch' + str(epoch))
    scheduler.step()
    correct = 0
    total = 0
    # train model
    model.train()
    for i,dataset in enumerate(train_loader,0):
        # images,labels = dataset
        images = dataset[0].to(device)
        labels = dataset[1].to(device)
        # forward pass
        outputs = model(images)
        # return train loss
        loss = criterion(outputs,labels)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        # bp
        loss.backward()
        if epoch > 6:
            for group in optimizer.param_groups:
                for p in group['params']:
                    state = optimizer.state[p]
                    if 'step' in state.keys():
                        if (state['step'] >= 1024):
                            state['step'] = 1000
        # turn all gardients to zero
        optimizer.zero_grad()
        optimizer.step()
    train_accuracy = float(correct/total) * 100
    print("train accuracy :",train_accuracy)

    model.eval()
    with torch.no_grad():
        correct_test = 0
        total_test = 0
        for images,labels in testloader:
            images = images.to(device)
            labels = labels.to(device)
            output = model(images)
            _, predicted = torch.max(output.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        test_accuracy = float(correct / total) * 100
        print("test accuracy :", test_accuracy)



























