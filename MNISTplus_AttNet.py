
# coding: utf-8

# # MNIST+ Classification with Attention Net
# By Yuheng Zhi, Shanghai Jiao Tong University
# Welcome! Simply run the notebook for results, or you can simply check the recorded results.
# Note:
# 1. Check Block 3 to know how to place the dataset
# 2. Use proper `train` and `validate` function for different models, according to the comments. Sorry I should've avoided overloading the functions, but you know, for efficiency.

# In[114]:


import torch
import numpy as np
import logger
import torch.utils.data as data_utils
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable
import os
import time


# In[300]:


BATCH_SIZE = 32
NUM_WORKERS = 4
CUDA=True
EPOCH_NUM = 100
EPOCH = 1
LR = 0.001
SAVE_PATH = 'saved_model'


# In[158]:


data_num = 60000 #The number of figures
fig_w = 45 

train_data = np.fromfile('mnist/mnist_train/mnist_train_data', dtype=np.uint8)
train_label = np.fromfile('mnist/mnist_train/mnist_train_label', dtype=np.uint8)

test_data = np.fromfile('mnist/mnist_test/mnist_test_data', dtype=np.uint8)
test_label = np.fromfile('mnist/mnist_test/mnist_test_label', dtype=np.uint8)



# In[159]:


miu, sigma = 127.5, 127.5
train_data = (train_data - miu) / sigma
print(np.max(train_data), np.min(train_data))

train_data = np.reshape(train_data, newshape=(data_num, 1, fig_w, fig_w))
train_data = train_data.astype(np.float32)
# print(np.max(train_data), len(train_data))
# print(np.max(train_label), len(train_label))

train_data = torch.from_numpy(train_data)
train_label = torch.from_numpy(train_label)
train_dataset = data_utils.TensorDataset(data_tensor=train_data, target_tensor=train_label)


# In[160]:


test_data = (test_data - miu) / sigma
print(np.max(test_data), np.min(test_data))

test_data = np.reshape(test_data, newshape=(10000, 1, fig_w, fig_w))
test_data = test_data.astype(np.float32)
print(len(test_data))
# print(np.max(train_data), len(train_data))
# print(np.max(train_label), len(train_label))

test_data = torch.from_numpy(test_data)
test_label = torch.from_numpy(test_label)
test_dataset = data_utils.TensorDataset(data_tensor=test_data, target_tensor=test_label)


# In[161]:


train_loader = data_utils.DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, drop_last=True)
test_loader = data_utils.DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, drop_last=True)


# In[162]:


# The baseline model
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, padding=1, kernel_size=3)
        self.conv2 = nn.Conv2d(10, 20, padding=1, kernel_size=3)
        self.conv2_drop_batch = nn.Sequential(
            nn.BatchNorm2d(20),
            nn.Dropout2d(inplace=True)
        )
        self.conv3 = nn.Conv2d(20, 40, padding=1, kernel_size=3)
        self.conv4 = nn.Conv2d(40, 80, padding=1, kernel_size=3)
        self.conv4_drop_batch = nn.Sequential(
            nn.BatchNorm2d(80),
            nn.Dropout2d(inplace=True)
        )
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal(m.weight, mode='fan_out')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant(m.weight, 1)
                nn.init.constant(m.bias, 0)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop_batch(self.conv2(x)), 2))
        x = F.relu(F.max_pool2d(self.conv3(x), 2))
        x = F.relu(F.max_pool2d(self.conv4_drop_batch(self.conv4(x)), 2))
#         print(x.size())
        x = x.view(-1, x.size(1) * x.size(2) * x.size(3))
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return x
    


# In[304]:


# Val function for baseline net and AttNet, but NOT for RecNet
def validate(data_loader, model, criterion):
    cumul_loss = 0.
    cumul_acc = 0.
    for i, (batch_inputs, batch_labels) in enumerate(data_loader):
        batch_inputs = Variable(batch_inputs)
        batch_labels = Variable(batch_labels)
        if CUDA:
            batch_inputs = batch_inputs.cuda()
            batch_labels = batch_labels.cuda()
        
        batch_logits = model(batch_inputs)
        loss = criterion(batch_logits, batch_labels)
        _, pred = torch.max(batch_logits, dim=1)
#         print("Pred, ", pred)
        acc = torch.mean(torch.eq(pred, batch_labels).float())
#         print("Acc, ", acc)
        
        cumul_loss += loss
        cumul_acc += acc
    
    avg_loss = cumul_loss / len(data_loader)
    avg_acc = cumul_acc / len(data_loader)
#     print("Hey: ", avg_loss.data.cpu().numpy()[0], avg_acc)
    
    return avg_loss.data.cpu().numpy()[0], avg_acc.data.cpu().numpy()[0]


# In[305]:


# Training function for baseline net and AttNet, but NOT for RecNet
def train(train_loader, model, criterion, optimizer, epoch):
    load_time = 0
    batch_time = 0
        
    cumul_loss = 0
    cumul_acc = 0
    end = time.time()
    for i, (batch_inputs, batch_labels) in enumerate(train_loader):
        batch_inputs = Variable(batch_inputs)
        batch_labels = Variable(batch_labels)
        if CUDA:
            batch_inputs = batch_inputs.cuda()
            batch_labels = batch_labels.cuda()
        load_time += time.time() - end
        
        batch_logits = model(batch_inputs)
        loss = criterion(batch_logits, batch_labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        _, pred = torch.max(batch_logits, dim=1)
        acc = torch.mean(torch.eq(pred, batch_labels).float())
        
        cumul_loss += loss
        cumul_acc += acc
        
        batch_time += time.time() - end
        end = time.time()
    
    avg_loss = cumul_loss / len(train_loader)
    avg_loss = avg_loss.data.cpu().numpy()[0]
    avg_acc = cumul_acc / len(train_loader)
    avg_acc = avg_acc.data.cpu().numpy()[0]
    
    return avg_loss, avg_acc, load_time, batch_time
    
    


# In[311]:


base_net = Net().cuda()
# batch_inputs, batch_labels = next(iter(train_loader))
# batch_inputs = Variable(batch_inputs)
# batch_labels = Variable(batch_labels)
# if CUDA:
#     batch_inputs = batch_inputs.cuda()
#     batch_labels = batch_labels.cuda()
# print(torch.max(base_net(batch_inputs), dim=1)[0])


# In[312]:


criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(base_net.parameters(), lr=LR)
test_interval = 3
decay_interval = 10


# In[313]:


for epoch in range(EPOCH, EPOCH_NUM+1):
    decay_times = epoch // decay_interval
    lr = LR * 0.1 ** decay_times
    for para_group in optimizer.param_groups:
        para_group['lr'] = lr
    
    avg_loss, avg_acc, load_time, batch_time = train(train_loader, base_net, criterion, optimizer, epoch)
    print("Epoch {}, Lr {}, Loss: {}, Acc: {}, load_time: {}, batch_time: {}".format(epoch, lr, avg_loss, avg_acc, load_time, batch_time))
    
    if epoch % test_interval == 0:
        save_path = os.path.join(SAVE_PATH, 'base_net')
        if not os.path.isdir(save_path):
            os.makedirs(save_path)
        file_name = os.path.join(save_path, 'base_net_' + str(epoch)+'.pth')
        torch.save({
            'epoch': epoch,
            'lr': LR,
            'model_state': base_net.state_dict(),
            'optimizer_state': optimizer.state_dict()
        }, file_name)
        print('Model saved at '+file_name)

        print("Testing...")
        test_loss, test_acc = validate(test_loader, base_net, criterion)
        print("Loss {}, Acc {}".format(test_loss, test_acc))
    

    


# In[314]:


class RecNet(nn.Module):
    def __init__(self):
        super(RecNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, padding=1, kernel_size=3)
        self.conv2 = nn.Conv2d(10, 20, padding=1, kernel_size=3)
        self.conv2_drop_batch = nn.Sequential(
            nn.BatchNorm2d(20),
            nn.Dropout2d(inplace=True)
        )
        self.conv3 = nn.Conv2d(20, 40, padding=1, kernel_size=3)
        self.conv4 = nn.Conv2d(40, 80, padding=1, kernel_size=3)
        self.conv4_drop_batch = nn.Sequential(
            nn.BatchNorm2d(80),
            nn.Dropout2d(inplace=True)
        )
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)
        
        self.deconv1 = nn.ConvTranspose2d(80, 40, padding=0, stride=2, kernel_size=3)
        self.deconv1_bn = nn.BatchNorm2d(40)
        self.deconv2 = nn.ConvTranspose2d(40, 20, padding=0, stride=2, kernel_size=3)
        self.deconv3 = nn.ConvTranspose2d(20, 10, padding=0, stride=2, kernel_size=2)
        self.deconv3_bn = nn.BatchNorm2d(10)
        self.deconv4 = nn.ConvTranspose2d(10, 1, padding=0, stride=2, kernel_size=3)
        
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal(m.weight, mode='fan_out')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant(m.weight, 1)
                nn.init.constant(m.bias, 0)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop_batch(self.conv2(x)), 2))
        x = F.relu(F.max_pool2d(self.conv3(x), 2))
        x = F.relu(F.max_pool2d(self.conv4_drop_batch(self.conv4(x)), 2))
#         print(x.size())
        x_ft = x.view(-1, x.size(1) * x.size(2) * x.size(3))
        x_ft = F.relu(self.fc1(x_ft))
        x_ft = F.dropout(x_ft, training=self.training)
        logits = self.fc2(x_ft)
        
        x_rec = F.relu(self.deconv1_bn(self.deconv1(x)))
        x_rec = F.relu(self.deconv2(x_rec))
        x_rec = F.relu(self.deconv3_bn(self.deconv3(x_rec)))
        x_rec = F.tanh(self.deconv4(x_rec))
        return logits, x_rec


# In[367]:


rec_net = RecNet().cuda()
# batch_inputs, batch_labels = next(iter(train_loader))
# batch_inputs = Variable(batch_inputs)
# batch_labels = Variable(batch_labels)
# if CUDA:
#     batch_inputs = batch_inputs.cuda()
#     batch_labels = batch_labels.cuda()
# print(rec_net(batch_inputs))


# In[368]:


class RecCrossEntropyLoss(nn.Module):
    def __init__(self, rec_ratio):
        super(RecCrossEntropyLoss, self).__init__()
        self.rec_ratio = rec_ratio
    
    def forward(self, rec, inputs, logits, targets):
        rec_loss = nn.MSELoss()
        cls_loss = nn.CrossEntropyLoss()
        
        return cls_loss(logits, targets) + self.rec_ratio * rec_loss(rec, inputs)

rec_ratio = 0.5
criterion = RecCrossEntropyLoss(rec_ratio)
rec_optimizer = torch.optim.Adam(rec_net.parameters(), lr=LR)
test_interval = 3
decay_interval = 10


# In[369]:


# Training function for RecNet, but NOT for baseline net and AttNet
def train(train_loader, model, criterion, optimizer, epoch):
    load_time = 0
    batch_time = 0
        
    cumul_loss = 0
    cumul_acc = 0
    end = time.time()
    for i, (batch_inputs, batch_labels) in enumerate(train_loader):
        batch_inputs = Variable(batch_inputs)
        batch_labels = Variable(batch_labels)
        if CUDA:
            batch_inputs = batch_inputs.cuda()
            batch_labels = batch_labels.cuda()
        load_time += time.time() - end
        
        batch_logits, batch_recs = model(batch_inputs)
        loss = criterion(batch_recs, batch_inputs, batch_logits, batch_labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        _, pred = torch.max(batch_logits, dim=1)
        acc = torch.mean(torch.eq(pred, batch_labels).float())
        
        cumul_loss += loss
        cumul_acc += acc
        
        batch_time += time.time() - end
        end = time.time()
    
    avg_loss = cumul_loss / len(train_loader)
    avg_loss = avg_loss.data.cpu().numpy()[0]
    avg_acc = cumul_acc / len(train_loader)
    avg_acc = avg_acc.data.cpu().numpy()[0]
    
    return avg_loss, avg_acc, load_time, batch_time


# In[370]:


# Val function for RecNet, but NOT for baseline net and AttNet
def validate(data_loader, model, criterion):
    cumul_loss = 0.
    cumul_acc = 0.
    for i, (batch_inputs, batch_labels) in enumerate(data_loader):
        batch_inputs = Variable(batch_inputs)
        batch_labels = Variable(batch_labels)
        if CUDA:
            batch_inputs = batch_inputs.cuda()
            batch_labels = batch_labels.cuda()
        
        batch_logits, batch_recs = model(batch_inputs)
        loss = criterion(batch_recs, batch_inputs, batch_logits, batch_labels)
        _, pred = torch.max(batch_logits, dim=1)
#         print("Pred, ", pred)
        acc = torch.mean(torch.eq(pred, batch_labels).float())
#         print("Acc, ", acc)
        
        cumul_loss += loss
        cumul_acc += acc
    
    avg_loss = cumul_loss / len(data_loader)
    avg_acc = cumul_acc / len(data_loader)
#     print("Hey: ", avg_loss.data.cpu().numpy()[0], avg_acc)
    
    return avg_loss.data.cpu().numpy()[0], avg_acc.data.cpu().numpy()[0]


# In[371]:


for epoch in range(EPOCH, EPOCH_NUM+1):
    decay_times = epoch // decay_interval
    lr = LR * 0.1 ** decay_times
    for para_group in rec_optimizer.param_groups:
        para_group['lr'] = lr
    
    avg_loss, avg_acc, load_time, batch_time = train(train_loader, rec_net, criterion, rec_optimizer, epoch)
    print("Epoch {}, Lr {}, Loss: {}, Acc: {}, load_time: {}, batch_time: {}".format(epoch, lr, avg_loss, avg_acc, load_time, batch_time))
    
    if epoch % test_interval == 0:
        save_path = os.path.join(SAVE_PATH, 'rec_net')
        if not os.path.isdir(save_path):
            os.makedirs(save_path)
        file_name = os.path.join(save_path, 'rec_net_' + str(epoch)+'.pth')
        torch.save({
            'epoch': epoch,
            'lr': LR,
            'model_state': base_net.state_dict(),
            'optimizer_state': optimizer.state_dict()
        }, file_name)
        print('Model saved at '+file_name)

        print("Testing...")
        test_loss, test_acc = validate(test_loader, rec_net, criterion)
        print("Loss {}, Acc {}".format(test_loss, test_acc))


# In[301]:


class AttentionNet(nn.Module):
    def __init__(self):
        super(AttentionNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, padding=1, kernel_size=3)
        self.attention1 = nn.Conv2d(1, 1, padding=1, kernel_size=3)
        self.conv2 = nn.Conv2d(10, 20, padding=1, kernel_size=3)
        self.attention2 = nn.Conv2d(10, 1, padding=1, kernel_size=3)
        self.conv2_drop_batch = nn.Sequential(
            nn.BatchNorm2d(20),
            nn.Dropout2d(inplace=True)
        )
        self.conv3 = nn.Conv2d(20, 40, padding=1, kernel_size=3)
        self.attention3 = nn.Conv2d(20, 1, padding=1, kernel_size=3)
        self.conv4 = nn.Conv2d(40, 80, padding=1, kernel_size=3)
        self.attention4 = nn.Conv2d(40, 1, padding=1, kernel_size=3)
        self.conv4_drop_batch = nn.Sequential(
            nn.BatchNorm2d(80),
            nn.Dropout2d(inplace=True)
        )
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal(m.weight, mode='fan_out')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant(m.weight, 1)
                nn.init.constant(m.bias, 0)

    def forward(self, x):
        
        attention = F.sigmoid(self.attention1(x))
        x = self.conv1(x)
        x = x * attention
        x = F.relu(F.max_pool2d(x, 2))
        
        attention = F.sigmoid(self.attention2(x))
        x = self.conv2(x)
        x = x * attention
        x = F.relu(F.max_pool2d(self.conv2_drop_batch(x), 2))
        
        attention = F.sigmoid(self.attention3(x))
        x = self.conv3(x)
        x = x * attention
        x = F.relu(F.max_pool2d(x, 2))
        
        attention = F.sigmoid(self.attention4(x))
        x = self.conv4(x)
        x = x * attention
        x = F.relu(F.max_pool2d(self.conv4_drop_batch(x), 2))
#         print(x.size())

        x_ft = x.view(-1, x.size(1) * x.size(2) * x.size(3))
        x_ft = F.relu(self.fc1(x_ft))
        x_ft = F.dropout(x_ft, training=self.training)
        logits = self.fc2(x_ft)
        
        return logits


# In[308]:


att_net = AttentionNet().cuda()
# batch_inputs, batch_labels = next(iter(train_loader))
# batch_inputs = Variable(batch_inputs)
# batch_labels = Variable(batch_labels)
# if CUDA:
#     batch_inputs = batch_inputs.cuda()
#     batch_labels = batch_labels.cuda()
# print(att_net(batch_inputs))


# In[309]:


criterion = nn.CrossEntropyLoss()
att_optimizer = torch.optim.Adam(att_net.parameters(), lr=LR)
test_interval = 10
decay_interval = 30


# In[310]:


for epoch in range(EPOCH, EPOCH_NUM+1):
    decay_times = epoch // decay_interval
    lr = LR * 0.1 ** decay_times
    for para_group in optimizer.param_groups:
        para_group['lr'] = lr
    
    avg_loss, avg_acc, load_time, batch_time = train(train_loader, att_net, criterion, att_optimizer, epoch)
    print("Epoch {}, Lr {}, Loss: {}, Acc: {}, load_time: {}, batch_time: {}".format(epoch, lr, avg_loss, avg_acc, load_time, batch_time))
    
    if epoch % test_interval == 0:
        save_path = os.path.join(SAVE_PATH, 'att_net')
        if not os.path.isdir(save_path):
            os.makedirs(save_path)
        file_name = os.path.join(save_path, 'att_net_' + str(epoch)+'.pth')
        torch.save({
            'epoch': epoch,
            'lr': LR,
            'model_state': base_net.state_dict(),
            'optimizer_state': optimizer.state_dict()
        }, file_name)
        print('Model saved at '+file_name)

        print("Testing...")
        test_loss, test_acc = validate(test_loader, att_net, criterion)
        print("Loss {}, Acc {}".format(test_loss, test_acc))
    


# In[358]:


import matplotlib.pyplot as plt

def visualize_attention(att_net, test_loader):
    for (batch_inputs, batch_labels) in test_loader:
        batch_inputs = Variable(batch_inputs)
        batch_labels = Variable(batch_labels)
        if CUDA:
            batch_inputs = batch_inputs.cuda()
            batch_labels = batch_labels.cuda()
        plt.figure(figsize=[22, 10])
        for i in range(5):
            plt.subplot(5, 11, 1 + 11*i)
            if i == 0:
                plt.title('input')
            plt.axis('off')
            plt.imshow(-batch_inputs[i, 0].data.cpu().numpy(), cmap='Greys')
            oris, atts = get_attention(att_net, batch_inputs[i:(i+1)])
            for j, (ori, att) in enumerate(zip(oris, atts)):
                plt.subplot(5, 11, 2*j + 11 * i + 2)
                if i == 0:
                    plt.title('Original')
                tmp = ori.data.cpu().numpy()
                plt.axis('off')
                plt.imshow(tmp[0, 0], cmap='Greys')
                
                plt.subplot(5, 11, 2*j + 11 * i + 3)
                if i == 0:
                    plt.title('Attented')
                tmp = att.data.cpu().numpy()
                    #       print(tmp.shape)
                plt.axis('off')
                plt.imshow(tmp[0, 0], cmap='Greys')
        plt.savefig('attention.pdf', bbox_inches='tight')
        break
        
def get_attention(att_net, x):
        
    attention1 = F.sigmoid(att_net.attention1(x))
    x1_ori = att_net.conv1(x)
    x1 = x1_ori * attention1
    x = F.relu(F.max_pool2d(x1, 2))

    attention2 = F.sigmoid(att_net.attention2(x))
    x2_ori = att_net.conv2(x)
    x2 = x2_ori * attention2
    x = F.relu(F.max_pool2d(att_net.conv2_drop_batch(x2), 2))

    attention3 = F.sigmoid(att_net.attention3(x))
    x3_ori = att_net.conv3(x)
    x3 = x3_ori * attention3
    x = F.relu(F.max_pool2d(x3, 2))

    attention4 = F.sigmoid(att_net.attention4(x))
    x4_ori = att_net.conv4(x)
    x4 = x4_ori * attention4
    x = F.relu(F.max_pool2d(att_net.conv4_drop_batch(x4), 2))
#         print(x.size())

#     x_ft = x.view(-1, x.size(1) * x.size(2) * x.size(3))
#     x_ft = F.relu(att_net.fc1(x_ft))
#     x_ft = F.dropout(x_ft, training=self.training)
#     logits = self.fc2(x_ft)

    return [x1_ori, x2_ori, x3_ori, x4_ori], [x1, x2, x3, x4]


# In[359]:


get_ipython().magic(u'matplotlib inline')
get_ipython().magic(u"config InlineBackend.figure_format = 'svg'")
visualize_attention(att_net, test_loader)


# In[364]:


def visualize_rec(model, test_loader):
    for (batch_inputs, batch_labels) in test_loader:
        batch_inputs = Variable(batch_inputs)
        batch_labels = Variable(batch_labels)
        if CUDA:
            batch_inputs = batch_inputs.cuda()
            batch_labels = batch_labels.cuda()
        logits, batch_recs = model(batch_inputs)
        _, pred = torch.max(logits, dim=1)
        plt.figure(figsize=[10, 4])
        for i in range(5):
            plt.subplot(2, 5, i+1)
            plt.title(pred[i].data.cpu().numpy()[0])
            plt.axis('off')
            plt.imshow(-batch_inputs[i, 0].data.cpu().numpy(), cmap='Greys')
            plt.subplot(2, 5, i+6)
            plt.axis('off')
            plt.imshow(-batch_recs[i, 0].data.cpu().numpy(), cmap='Greys')
        plt.savefig('rec.pdf', bbox_inches='tight')
        break


# In[372]:


visualize_rec(rec_net, test_loader)

