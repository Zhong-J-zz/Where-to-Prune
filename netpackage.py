import torch
import torch.nn as nn
import torch.nn.functional as F 
import torch.autograd as autograd 
import copy
import numpy as np
import os
n_pool = 5
class BasicConv2d(nn.Module):

    def __init__(self, in_chanel, out_planes, kernel_size=3, stride=1, padding=1):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_chanel, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding) 
        self.bn = nn.BatchNorm2d(out_planes,
                                 eps=0.001, 
                                 momentum=0.1, 
                                 affine=True)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class LSTM_PRUNE(nn.Module):
    def __init__(self, in_dim, hidden_dim,n_layer, out_dim = 2):
        super(LSTM_PRUNE, self).__init__()
        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTM(in_dim, hidden_dim,n_layer,dropout = 0.8)
        self.hidden2dec = nn.Linear(hidden_dim, out_dim) 
        self.hidden = self.init_hidden()

    def init_hidden(self):
        return (autograd.Variable(torch.zeros(2, 1, self.hidden_dim).cuda()),
                autograd.Variable(torch.zeros(2, 1, self.hidden_dim).cuda()))

    def forward(self, sequence):    
        lstm_out, self.hidden = self.lstm(
            sequence.view(len(sequence), 1, -1), self.hidden)       
        dec = self.hidden2dec(lstm_out.view(len(sequence), -1))
        dec_scores = F.log_softmax(dec)
        return dec_scores, dec   
        
class CIFAR(nn.Module):
    def __init__(self, features, n_channel, num_classes):
        super(CIFAR, self).__init__()
        assert isinstance(features, nn.Sequential), type(features)
        self.features = features
        self.classifier = nn.Sequential(
            nn.Linear(n_channel, num_classes)
        )
        print(self.features)
        print(self.classifier)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x
        
class PRUNE_RESNET_CNN(nn.Module):
    def __init__(self,net, to_add_conv, add_tuple):
        super(PRUNE_RESNET_CNN, self).__init__()
        self.features = net.features
        
        self.linear =net.linear
        self.pool = nn.MaxPool2d(2,2)           
        self.to_add_conv = to_add_conv 
        basic_block = []
        for in_chanels,out_channels in add_tuple:
            basic_block.append(BasicConv2d(in_chanels, out_channels, kernel_size=1, stride=1, padding=0))
        self.basic_conv2d = nn.Sequential(*basic_block)
        #print(self.basic_conv2d)
    def forward(self, y):
        save = False
        num_conv = 0
        add_layer = None
        basic_conv_index = 0
        
        for x in self.features:
            y = x(y)
            if isinstance(x, torch.nn.ReLU):                
                
                if num_conv-1 in self.to_add_conv:

                    if add_layer.size()[1] != y.size()[1]:
                        add_layer = self.basic_conv2d[basic_conv_index](add_layer)
                        basic_conv_index += 1
                    if add_layer.size()[2] != y.size()[2]:
                        add_layer = self.pool(add_layer)
                    y = y + add_layer
                if num_conv in self.to_add_conv: 
                    add_layer = y
                    
                num_conv += 1  
        n = self.num_flat_features(y)        
        y = y.view(-1,n)
        y = self.linear(y)
        y = F.softmax(y)  
        return y
        
    def num_flat_features(self,x):
        size = x.size()[1:]        
        num_features = 1
        for s in size:
            num_features *= s
        return num_features   
        
class CNN(nn.Module):
    def __init__(self,features,classification):
        super(CNN, self).__init__()
        self.features = features
        self.linear =classification
    def forward(self, y):

        y = self.features(y)
        n = self.num_flat_features(y)        
        y = y.view(-1,n)
        y = self.linear(y)
        y = F.softmax(y)  
        return y
        
    def num_flat_features(self,x):
        size = x.size()[1:]        
        num_features = 1
        for s in size:
            num_features *= s
        return num_features      

def make_layers(network, n_pool = 5):
    features = []
    classification = []    
    in_channels = 3
    units = 0
    for x in network:
        if 'C:' in x:
            out_channels = int(x.split(':')[1])
            features.append(BasicConv2d(in_channels,out_channels))
            in_channels = out_channels
        elif 'P' == x:
            features.append(nn.MaxPool2d(2,2))
        elif 'F:' in x:
            if units ==0:
                units = int(in_channels*32*32/pow(2,n_pool)/pow(2,n_pool))
            classification += [nn.Linear(units,int(x.split(':')[1])), nn.BatchNorm1d(int(x.split(':')[1]),eps=0.001, momentum=0.1, affine=True),nn.ReLU(inplace=False)]
            units = int(x.split(':')[1])
    if units == 0:
        units = int(in_channels*32*32/pow(2,n_pool)/pow(2,n_pool))
    classification += [nn.Linear(units,int(x.split('S:')[1]))] 

    return nn.Sequential(*features),nn.Sequential( *classification)
    
def prepare_sequence_double(network,num =1):
    sequence = []
    for x in network:
        if 'C:' in x:
            sequence.append([0,int(x.split(':')[1])])
        elif 'F:' in x:
            sequence.append([1,int(x.split(':')[1])]) 
        elif 'P' in x:
            sequence.append([2,2])
    if num==2:
        for i,x in enumerate(sequence[:len(sequence)-1]):
            sequence[i] += sequence[i+1]
        sequence[len(sequence)-1] += [1,10]
        for i in range(len(sequence))[::-1]:
            if sequence[i][0]==2:
                sequence.pop(i)
    tensor = torch.FloatTensor(sequence)    
    return autograd.Variable(tensor.cuda())  

def get_network_history(path):
    network_dict = []
    f = open(path,'rb')
    f.seek(0,os.SEEK_END)
    l = f.tell()
    f.seek(0,os.SEEK_SET)
    while(1):
        if f.tell() == l:
            break
        x = np.load(f)
        network_dict.append(x)
    return network_dict