#-*- coding:utf-8 -*-
import torch
import torch.autograd as autograd 
import torch.nn as nn             
import torch.optim as optim    
import torchvision
import torch.nn.functional as F 
import torchvision.transforms as transforms
import numpy as np
import torch.backends.cudnn as cudnn
import random
import argparse
import sys
import torch.optim.lr_scheduler as lr_scheduler
import netpackage
from netpackage import CIFAR,CNN
import copy
import os
import shutil
import prunepackage

vgg19 = ['C:64','C:64','P','C:128','C:128','P','C:256','C:256','C:256','C:256','P','C:512','C:512','C:512','C:512','P','C:512','C:512','C:512','C:512','P','S:10']
prune_percent = 0.8
network_dict = []
this_network = vgg19
def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--cnn_batch_size',type=int,default=128)
    parser.add_argument('--cnn_epoch_size',type = int, default = 30)
    parser.add_argument('--lstm_topk_size',type=int,default=5)
    parser.add_argument('--lstm_epoch_size',type = int,default = 50000)
    parser.add_argument('--data_dir',type = str,default = '/data/')
    parser.add_argument('--pretrain',action='store_true', default = False)
    parser.add_argument('--pretrained_model',type = str, default = '/1down_params_lstm_prune.pkl')
    parser.add_argument('--use_teacher', action='store_true', default = True)
    parser.add_argument('--teacher_dir',type = str, default = 'model0.pkl')
    parser.add_argument('--history_network_npy',type = str,default = '/network_prune.npy')
    parser.add_argument('--use_history_network',action='store_true', default = False)
    parser.add_argument('--lr',type = float,default = 0.001)
    parser.add_argument('--train_model',type = str,choices=['lstm', 'lstm_fullcov', 'lstm_prune'], default = 'lstm_prune')
    parser.add_argument('--save_lstm_path',type = str, default = '/lstm_prune.pkl')
    parser.add_argument('--save_history_npy',type = str, default = '/network_prune.npy')
    parser.add_argument('--save_log',type = str, default = '/network_prune.txt')
    parser.add_argument('--prune',action = 'store_true', default = True)
    return parser.parse_args(argv)    
    

def data_process(data_dir,batch_size):

    transform_train=transforms.Compose([                                  
                                  transforms.RandomHorizontalFlip(),
                                  transforms.RandomCrop(32,padding = 4),       
                                  transforms.ToTensor(),
                                  transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                                 ])
    transform_test=transforms.Compose([
                                  transforms.ToTensor(),
                                  transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                                 ])   
                                  

    train_dataset = torchvision.datasets.CIFAR10(root=data_dir, train=True, 
                download=False, transform=transform_train)

    valid_dataset = torchvision.datasets.CIFAR10(root=data_dir, train=True, 
                download=False, transform=transform_test)

    test_dataset = torchvision.datasets.CIFAR10(root=data_dir, train=False, 
                download=False, transform=transform_test)
    num_train = len(train_dataset)
    indices = list(range(num_train))
    split = int(np.floor(0.1 * num_train))
    np.random.seed(0)
    np.random.shuffle(indices)

    train_idx, valid_idx = indices[split:], indices[:split]
    train_sampler = torch.utils.data.sampler.SubsetRandomSampler(train_idx)
    valid_sampler = torch.utils.data.sampler.SubsetRandomSampler(valid_idx)

    train_loader = torch.utils.data.DataLoader(train_dataset, 
                    batch_size=batch_size, sampler=train_sampler, 
                    num_workers=2)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, 
                    batch_size=batch_size, sampler=valid_sampler, 
                    num_workers=2)
    test_loader = torch.utils.data.DataLoader(test_dataset, 
                    batch_size=batch_size, 
                    num_workers=2)    
    return train_loader,valid_loader, test_loader
    
def gen_trainset(network_dict, batch_lstm):
    if(len(network_dict)<batch_lstm):
        return [[network_dict[i][1],network_dict[i][2]] for i in range(len(network_dict))]
    else:     
        trainset = [[network_dict[i][1],network_dict[i][2]] for i in range(batch_lstm)]       
        return trainset
    
def generate_network(dec_scores, this_network):
    network = copy.deepcopy(this_network)
    _,top_dec = dec_scores.data.topk(1)
    top_dec = top_dec.cpu().numpy()
    index = -1
    pos = 1
    out_channels = 32
    for i in range(len(this_network)):
        if this_network[i] != 'P' and 'S' not in this_network[i]:  
            index += 1                
            if top_dec[index]==1: 
                if this_network[i].split(':')[0]=='C':
                    c = int(int(network[i+pos-1].split(':')[1])*prune_percent)
                    network[i+pos-1] = 'C:'+str(c)
                           
    new = []
    global n_pool
    n_pool =0
    for x in network:
        if 'P' not in x:
            new.append(x)
            
        elif 'P' in x:
            new.append(x)
            n_pool += 1   
    return new 
def compute_complex(net,network):
    w_l = []
    num_pool = 0
    for x in network:
        if 'C' in x:
            w_l.append(int(32/pow(2,num_pool)))
            out_channels = int(x.split(':')[1])
        if 'P' in x:
            num_pool += 1    

    total_complex = 0
    index = 0
    for x in net.features:
        if isinstance(x, nn.Conv2d):
            conv = list(x.parameters())[0].size()
            per = 1
            for s in conv:
               per = per * s 
            total_complex += per *w_l[index]*w_l[index] 
            index += 1 
    
    return total_complex

    
def  calculate_loss(reward,zero_reward, dec_scores,dec):
    
    reward_array =  reward*np.array(zero_reward)
    reward_array = autograd.Variable(torch.from_numpy(np.array([[reward_array[i],reward_array[i] ]for i in range(len(reward_array))]))).float().cuda()
    mask_dec = torch.cat([x != x.max() for x in dec]).resize(dec.size()[0],dec.size()[1]).float()
    mask_dec1 = torch.cat([x == x.max() for x in dec]).resize(dec.size()[0],dec.size()[1]).float()
    
    if reward<0:
        loss = (reward_array*dec_scores*mask_dec).sum()/mask_dec.sum()
    else:
        loss = -(reward_array*dec_scores*mask_dec1).sum()/mask_dec1.sum()
    
    return loss
    

def get_pre_sum_weights(module):
    sum_weights = []    
    for i in range(module.weight.size()[1]):
        weights = np.sum(torch.pow(module.weight[:,i],2).cpu().data.numpy())    
        sum_weights.append(weights)
    return sum_weights  
    
def get_filter_prune(sum_weights, prune_num, largest):
    sum_weights = torch.from_numpy(np.array(sum_weights))
    _,index_array = sum_weights.topk(prune_num,0,largest = largest)
    index_array = sorted(index_array.numpy(),reverse = True)
    return index_array    
   
def change_network(net, target, del_filters, which_nn):    
    for layer_index, filter_index ,f in target:
        if which_nn == 'features':
            if f  == -1:
                net = prunepackage.prune_per_conv_layer(net,layer_index, filter_index)     
    return net
    
    
def pruning(index,network,del_filters,cnn_epoch_size,lr, use_teacher,trainloader,testloader, finalloader, model_teacher):
    net = torch.load('model'+str(index)+'.pkl')
    filter_prune = {}
    layers_prune = []
    prune_target = []   
    index = 0
    pre_bn_weights = 1
    for layer, module in enumerate(net.features):     
        if isinstance(module, nn.Conv2d):
            if index>0 and del_filters[index-1] > 1:
                filter_weights =  get_pre_sum_weights(module)                 
                filter_prune[layers_prune[-1]] =get_filter_prune(filter_weights,del_filters[index-1], False)
                pre_bn_weights = 1
            if del_filters[index] > 1:           
                pre_bn_weights = 0
                layers_prune.append(layer)
            index += 1

    if pre_bn_weights!=1:
        linear_weights = get_pre_sum_weights(net.linear[0]) 
        linear_weights = [ 0.9*linear_weights[i] for i in range(len(linear_weights))]
        if del_filters[index-1] > 1:        
            filter_prune[layers_prune[-1]] =get_filter_prune(linear_weights,del_filters[index-1], False)

    for l in layers_prune:
        for x in filter_prune[l]:
            prune_target.append((l, x, -1))
    target = sorted(prune_target, reverse = True)
    net = change_network(net, target, del_filters,'features')   
    reward, acc ,net= finetune(net, network, cnn_epoch_size,lr, use_teacher,trainloader,testloader, finalloader, model_teacher)
    return net, reward, acc
      
def get_del_filters(dec_scores, this_network,out_scores = 'WIDE'):
    del_filters = []   
    top_out = out_scores
    _,top_dec = dec_scores.data.topk(1)
    top_dec = top_dec.cpu().numpy()
    index = 0
    zero_reward = []
    for x in this_network:
        if 'P' not in x and 'S' not in x:
            if top_out == 'WIDE':
                if top_dec[index] == 0:
                    del_filters.append(0)
                    zero_reward.append(1)
                elif top_dec[index] == 1: 
                    del_filters.append(int(x.split(':')[1])-int(int(x.split(':')[1])*prune_percent))
                    zero_reward.append(min(abs(int(x.split(':')[1])-int(int(x.split(':')[1])*prune_percent))-0.5,1))     
            index += 1
    if zero_reward ==[]:
        zero_reward = [1 for x in range(len(del_filters))]
    return del_filters  , zero_reward              
                    
def finetune(net, network, cnn_epoch_size,lr, use_teacher,trainloader,testloader, finalloader, model_teacher):
    accuracy_list = []   
    net.cuda()
    cudnn.benchmark = True 
    Complex = compute_complex(net, network)
    print('complex:',Complex)                                                                                                                                                 
    # net = nn.DataParallel(net, device_ids=range(torch.cuda.device_count()))
    criterion = nn.CrossEntropyLoss()
    criterion_distill = nn.MSELoss()
    learning_rate = lr
    optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9,weight_decay=5e-4)
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 'max',factor = 0.5,patience=3)
    for epoch in range(cnn_epoch_size):
        net.train()
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            inputs, labels = autograd.Variable(inputs.cuda(),requires_grad=True), autograd.Variable(labels.cuda())            
            optimizer.zero_grad()           
            outputs = net(inputs)
            soft_target = F.softmax(model_teacher(inputs)).detach()
            loss = criterion(outputs, labels)
            if use_teacher:
                loss_soft = criterion_distill(outputs,soft_target)
                loss = loss_soft+ 0.1*loss
            loss.backward()  
            optimizer.step()
        acc = testCNN(net,testloader) 
        print('epoch:'+str(epoch)+'  acc:'+str(acc)+'  lr:'+str(optimizer.param_groups[0]['lr']))
        accuracy_list.append(acc)
    print('test acc:',testCNN(net,finalloader) )
    reward = -Complex* 4e-10+max(accuracy_list)
    return reward, max(accuracy_list), net
       
def testCNN(net,testloader):
    net.eval()
    correct = 0
    total = 0
    for data in testloader:
        images, labels = data
        outputs = net(autograd.Variable(images.cuda()))
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted.cpu() == labels).sum()    
    acc = correct/total
    return acc

def train(args):
    continue_networks = 0
    model = netpackage.LSTM_PRUNE(4, 100, 2, 2)
        
    model.cuda()
    if args.pretrain:
        model.load_state_dict(torch.load(args.pretrained_model))
    optimizer_lstm = optim.Adam(model.parameters(), lr=0.001,weight_decay=0)
    if args.use_teacher:
        model_teacher = torch.load(args.teacher_dir)
        model_teacher.cuda()
        model_teacher.eval()
    trainloader,testloader, finalloader = data_process(args.data_dir,args.cnn_batch_size) 
    if args.use_history_network:
        network_dict = netpackage.get_network_history(args.history_network_npy)
        network_dict.sort(key = lambda x:x[2],reverse=True)
    else:
        global this_network     
        Complex = compute_complex(model_teacher,this_network)                
        print(Complex)
        acc = 0.99
        this_reward = acc-4e-10*Complex
        network_dict=[[this_network, this_network,this_reward,acc]]  
        print('this reward:'+str(this_reward))
        
    network_num = len(network_dict)
    for epoch in range(args.lstm_epoch_size): 
        train_set = gen_trainset(network_dict, args.lstm_topk_size)
        old_net = []
        new_net = []
        new_net_index = 0
        for index,data in enumerate(train_set):
            this_network, best_reward = data
            old_net.append(best_reward)
            optimizer_lstm.zero_grad()
            model.hidden = model.init_hidden() 
            if args.train_model == 'lstm_prune': 
                input = netpackage.prepare_sequence_double(this_network,2)
                dec_scores,dec = model(input)
                new_network = generate_network(dec_scores, this_network)
                del_filters, zero_reward = get_del_filters(dec_scores, this_network)
            if (this_network,new_network) in [(network_dict[i][0],network_dict[i][1]) for i in range(len(network_dict))] :
                this_reward = network_dict[[(network_dict[i][0],network_dict[i][1]) for i in range(len(network_dict))].index((this_network,new_network))][2]
                acc = network_dict[[(network_dict[i][0],network_dict[i][1]) for i in range(len(network_dict))].index((this_network,new_network))][3]
                continue_networks += 1
            elif new_network ==this_network:
                this_reward = best_reward
                acc = ''
                continue_networks += 1
            else:            
                print(new_network,del_filters)               
                netcnn,this_reward, acc = pruning(index,new_network, del_filters,args.cnn_epoch_size,args.lr, args.use_teacher,trainloader,testloader, finalloader, model_teacher)
                if this_reward <= best_reward:
                	continue_networks += 1  
                else:
                	continue_networks = 0                       
                network_dict.append([this_network, new_network,this_reward,acc])
                network_num += 1
                torch.save(netcnn,'1model'+str(new_net_index)+'.pkl') 
                new_net.append(this_reward)
                new_net_index += 1
            reward = (this_reward - best_reward)*10-0.00001
            if args.train_model == 'lstm_prune':            
                loss_back = calculate_loss(reward,zero_reward, dec_scores,dec)
            print('epoch: %d no. of network: %d this_reward: %f,best_reward:%f\n  loss:'%(epoch,network_num,this_reward,best_reward),loss_back)
            loss_back.backward()
            optimizer_lstm.step()
            network_dict.sort(key = lambda x:x[2],reverse=True)
        torch.save(model.state_dict(), (args.save_lstm_path))       
        l = old_net + new_net
        l = torch.from_numpy(np.array(l))
        lenth = min(len(l), args.lstm_topk_size)
        _,id_list = l.topk(lenth)
        for i in range(lenth):
            if(id_list[i]+1 > len(old_net)):
                shutil.copy('1model'+str(id_list[i] - len(old_net))+'.pkl', '2model'+str(i)+'.pkl')
            else:
                shutil.copy('model'+str(id_list[i])+'.pkl', '2model'+str(i)+'.pkl')
        for i in range(lenth):
            shutil.copy('2model'+str(i)+'.pkl', 'model'+str(i)+'.pkl')
        if continue_networks >=10:
        	break
    
if __name__ == '__main__':
    train(parse_arguments(sys.argv[1:]))