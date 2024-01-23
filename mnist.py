import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import gzip
import os

n_epochs = 2
batch_size_train = 2
batch_size_test = 1000
lr = 0.01
momentum = 0.5
log_interval = 10
random_seed = 1
torch.manual_seed(random_seed)

transform = transforms.Compose([
    transforms.ToTensor(),  # 将 PIL 图像或 NumPy ndarray 转换为 FloatTensor。
    transforms.Normalize((0.1307,), (0.3081,))
    ])

    
class DatasetProcessing():
    def __init__(self, folder, data_name, label_name, transform):
        (train_set, train_labels) = self._load_data(folder, data_name, label_name)
        self.train_set = train_set
        self.train_labels = train_labels
        self.transforms = transform
      
    def _load_data(self, data_folder, data_name, label_name):
        with gzip.open(os.path.join(data_folder,label_name), 'rb') as lbpath: # rb表示的是读取二进制数据,lbpath指的就是标签的存放路径。
            y_train = np.frombuffer(lbpath.read(), np.uint8, offset=8)

        with gzip.open(os.path.join(data_folder,data_name), 'rb') as imgpath:
            x_train = np.frombuffer(
            imgpath.read(), np.uint8, offset=16).reshape(len(y_train), 28, 28)
        return (x_train, y_train)
    
    def __getitem__(self, index):

        img, target = self.train_set[index], int(self.train_labels[index])
        img = self.transforms(img)
        return img, target
    
    def __len__(self):
        return len(self.train_set)

train_dataset = DatasetProcessing('./data/',"train-images-idx3-ubyte.gz","train-labels-idx1-ubyte.gz", transform = transform)
test_dataset = DatasetProcessing('./data/',"t10k-images-idx3-ubyte.gz","t10k-labels-idx1-ubyte.gz", transform = transform)

train_loader = torch.utils.data.DataLoader(dataset = train_dataset, batch_size = batch_size_train, 
                                           shuffle = True
                                          )  # 此处的值根据您的数据集进行调整
test_loader = torch.utils.data.DataLoader(dataset = test_dataset, batch_size = batch_size_test, 
                                          shuffle = False
                                          ) # 此处的值根据您的数据集进行调整)

class Conv3x3(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(Conv3x3, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_dim, out_dim, kernel_size=3, bias=True), 
            nn.LeakyReLU(0.1, inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.c1 = Conv3x3(1,8)
        self.c2 = Conv3x3(8,16)
        self.c3 = Conv3x3(16,4)
        self.pool = nn.MaxPool2d(2,2)
        self.rl1 = nn.Linear(100,10)
        

    def forward(self,x):
        x = self.c1(x) # 26x26x8
        x = self.c2(x) # 24x24x16
        x = self.pool(x) # 12x12x16
        x = self.c3(x)# 10x10x4
        x = self.pool(x) #5x5x4
        x = x.view(-1,100)
        x = self.rl1(x)
        return F.log_softmax(x,dim=1)
    

model = Net()
    
optimizer = optim.SGD(model.parameters(),lr = lr, momentum = momentum)

train_losses = []
train_counter = []
test_losses = []
test_counter = [i * len(train_loader.dataset) for i in range(n_epochs + 1)]

def train(epoch):
    model.train()    

    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
            epoch, batch_idx * len(data), len(train_loader.dataset),
            100. * batch_idx / len(train_loader), loss.item()))
            
            train_losses.append(loss.item())
            train_counter.append(
                (batch_idx*64) + ((epoch-1)*len(train_loader.dataset)))
            torch.save(model.state_dict(), './model.pth')
            torch.save(optimizer.state_dict(), './optimizer.pth')
            
def test():
    model.load_state_dict(torch.load('./model.pth'))
    model.eval()
    correct = 0
    
    with torch.no_grad():
        for data,target in test_loader:
            output = model(data)
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).sum()
    print('Accuracy: {}/{} ({:.0f}%)\n'.format(
        correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    


for epoch in range(n_epochs):
    train(epoch+1)
    
test()

