import torch
import torch.nn as nn
import torch.utils.data
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import os
from torch.autograd import Variable

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
        self.fc = nn.Linear(256,200)

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

def create_val_folder(val_dir):
    """
    This method is responsible for separating validation images into separate sub folders
    """
    path = os.path.join(val_dir, 'images')  # path where validation data is present now
    filename = os.path.join(val_dir, 'val_annotations.txt')  # file where image2class mapping is present
    fp = open(filename, "r")  # open file in read mode
    data = fp.readlines()  # read line by line

    # Create a dictionary with image names as key and corresponding classes as values
    val_img_dict = {}
    for line in data:
        words = line.split("\t")
        val_img_dict[words[0]] = words[1]
    fp.close()
    # Create folder if not present, and move image into proper folder
    for img, folder in val_img_dict.items():
        newpath = (os.path.join(path, folder))
        if not os.path.exists(newpath):  # check if folder exists
            os.makedirs(newpath)
        if os.path.exists(os.path.join(path, img)):  # Check if image exists in default directory
            os.rename(os.path.join(path, img), os.path.join(newpath, img))
    return


# data augmentation
transform_train = transforms.Compose([transforms.RandomCrop(32, 4), transforms.RandomHorizontalFlip(),
                                      transforms.ToTensor(),
                                      transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
transform_val = transforms.Compose([transforms.ToTensor()])

train_dir = '/u/training/tra216/scratch/hw4/tiny-imagenet-200/train'
train_dataset = datasets.ImageFolder(train_dir,
         transform=transform_train)
#print(train_dataset.class_to_idx)
train_loader = torch.utils.data.DataLoader(train_dataset,
        batch_size=100, shuffle=True, num_workers=8)
val_dir = '/u/training/tra216/scratch/hw4/tiny-imagenet-200/val/'


if 'val_' in os.listdir(val_dir+'images/')[0]:
    create_val_folder(val_dir)
    val_dir = val_dir+'images/'
else:
    val_dir = val_dir+'images/'


val_dataset = datasets.ImageFolder(val_dir, transform=transforms.ToTensor())
# To check the index for each classes
# print(val_dataset.class_to_idx)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=100, shuffle=False, num_workers=8)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#
# train_dir = '/u/training/tra216/scratch/hw4/tiny-imagenet-200/train/'
# train_dataset = datasets.ImageFolder(train_dir, transform=transform_train)
# train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=100, shuffle=True, num_workers=8)
#
# val_dir = '/u/training/tra216/scratch/hw4/tiny-imagenet-200/val/'
# if 'val_' in os.listdir(val_dir + 'images/')[0]:
#     create_val_folder(val_dir)
#     val_dir = val_dir + 'images/'
# else:
#     val_dir = val_dir + 'images/'

# val_dataset = datasets.ImageFolder(val_dir, transform=transforms.ToTensor())
# val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=100,shuffle=False, num_workers=8)

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = ResNet(BasicBlock,[2,4,4,2]).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=.001)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

epochs = 150
for epoch in range(1, epochs + 1):
    print('epoch' + str(epoch))
    model.train()

    # for i, dataset in enumerate(train_loader):
    #     images = dataset[0].to(device)
    #     labels = dataset[1].to(device)
    for images, labels in train_loader:
        images = Variable(images).to(device)
        labels = Variable(labels).to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

    correct = 0
    total = 0
    model.eval()
    with torch.no_grad():
        # for dataset in val_loader:
        #     images = dataset[0].to(device)
        #     labels = dataset[1].to(device)
        for images, labels in val_loader:
            images = Variable(images).to(device)
            labels = Variable(labels).to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            _, predictions = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predictions == labels).sum().item()
    test_accuracy = float(correct / total) * 100
    print('test accuracy : %.2f' % (test_accuracy))

    scheduler.step()
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# print(device)
#
# model = MyResNet.MyResNet(MyResNet.BasicBlock, [2, 4, 4, 2], 200, 3, 64).to(device)
#
# criterion = nn.CrossEntropyLoss()
# optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
#
# total_step = len(train_loader)
#
# acc_df = pd.DataFrame(columns=["Train Accuracy", "Test Accuracy"])
# for epoch in range(num_epochs):
#     # Train the model
#     train_acc, test_acc = MyUtils.train(epoch, model, train_loader, val_loader,
#                                         device, optimizer, criterion, num_epochs,
#                                         total_step)
#
#     acc_df = acc_df.append({"Train Accuracy": train_acc,
#                             "Test Accuracy": test_acc},
#                            ignore_index=True)
#
#     acc_df.to_csv("./accuracy_tinyimagenet.csv")
#
# print("\nThe accuracy on the test set is: {:.2} %"
#         .format(MyUtils.calculate_accuracy(model, testloader)))
#
# # save the accuracy
# acc_df.to_csv("./accuracy_tinyimagenet.csv")
