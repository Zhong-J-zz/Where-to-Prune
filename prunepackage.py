import torch
import torch.nn as nn  
import numpy as np

def replace_layers(model, i, indexes, layers):
    if i in indexes:
        return layers[indexes.index(i)]
    return model[i]          
    
def prune_per_conv_layer(net,layer_index, filter_index):
    conv = net.features[layer_index]
    batch_norm = net.features[layer_index+1]
    new_batch_norm = nn.BatchNorm2d(conv.out_channels - 1,
                                 eps=0.001,
                                 momentum=0.1, 
                                 affine=False)
    
    next_conv = None
    offset = 1
    while layer_index + offset <  len(net.features):
        res =  net.features[layer_index+offset]
        if isinstance(res, nn.modules.conv.Conv2d):
            next_conv = res
            break
        offset = offset + 1
    
    new_conv = \
        nn.Conv2d(in_channels = conv.in_channels, \
            out_channels = conv.out_channels - 1,
            kernel_size = conv.kernel_size, \
            stride = conv.stride,
            padding = conv.padding,
            dilation = conv.dilation,
            groups = conv.groups,bias = False)

    old_weights = conv.weight.data.cpu().numpy()
    new_weights = new_conv.weight.data.cpu().numpy()
    new_weights[: filter_index, :, :, :] = old_weights[: filter_index, :, :, :]
    new_weights[filter_index : , :, :, :] = old_weights[filter_index + 1 :, :, :, :]
    new_conv.weight.data = torch.from_numpy(new_weights).cuda()
    bn_running_mean = new_batch_norm.running_mean.cpu().numpy()
    bn_old_running_mean = batch_norm.running_mean.cpu().numpy()    
    bn_running_mean[:filter_index] = bn_old_running_mean[: filter_index]
    bn_running_mean[filter_index :] =  bn_old_running_mean[filter_index + 1 :]
    new_batch_norm.running_mean =  torch.from_numpy(bn_running_mean).cuda()
    bn_running_var = new_batch_norm.running_var.cpu().numpy()
    bn_old_running_var = batch_norm.running_var.cpu().numpy()    
    bn_running_var[:filter_index] = bn_old_running_var[: filter_index]
    bn_running_var[filter_index :] =  bn_old_running_var[filter_index + 1 :]
    new_batch_norm.running_var =  torch.from_numpy(bn_running_var).cuda()    

    if not next_conv is None:
        next_new_conv = \
            torch.nn.Conv2d(in_channels = next_conv.in_channels - 1,\
                out_channels =  next_conv.out_channels, \
                kernel_size = next_conv.kernel_size, \
                stride = next_conv.stride,
                padding = next_conv.padding,
                dilation = next_conv.dilation,
                groups = next_conv.groups,bias = False)

        old_weights = next_conv.weight.data.cpu().numpy()
        new_weights = next_new_conv.weight.data.cpu().numpy()
        new_weights[:, : filter_index, :, :] = old_weights[:, : filter_index, :, :]
        new_weights[:, filter_index : , :, :] = old_weights[:, filter_index + 1 :, :, :]
        
        for i in range(len(new_weights)):
            old_sum = np.sum(pow(old_weights[i],2))
            alpha = 0.8
            if  np.sum(pow(old_weights[i,filter_index],2))/old_sum > alpha/len(old_weights):
                new_weights[i] = old_sum*new_weights[i]/np.sum(pow(new_weights[i],2))
        next_new_conv.weight.data = torch.from_numpy(new_weights).cuda()       
        features = torch.nn.Sequential(
                *(replace_layers(net.features, i, [layer_index,layer_index+1 ,layer_index+offset], \
                    [new_conv, new_batch_norm,next_new_conv]) for i, _ in enumerate(net.features)))
   
    else:
        features = torch.nn.Sequential(
                *(replace_layers(net.features, i, [layer_index, layer_index+1], \
                    [new_conv, new_batch_norm]) for i, _ in enumerate(net.features)))  
        layer_index = 0
        old_linear_layer = None
        for _, module in net.linear._modules.items():
            if isinstance(module, torch.nn.Linear):
                old_linear_layer = module
                break
            layer_index = layer_index  + 1
            
        params_per_input_channel = int(old_linear_layer.in_features / conv.out_channels)

        new_linear_layer = \
            torch.nn.Linear(old_linear_layer.in_features - params_per_input_channel, 
                old_linear_layer.out_features,bias = False)       
        old_weights = old_linear_layer.weight.data.cpu().numpy()
        new_weights = new_linear_layer.weight.data.cpu().numpy()        
        new_weights[:, : filter_index * params_per_input_channel] = \
            old_weights[:, : filter_index * params_per_input_channel]
        new_weights[:, filter_index * params_per_input_channel :] = \
            old_weights[:, (filter_index + 1) * params_per_input_channel :]

        new_linear_layer.weight.data = torch.from_numpy(new_weights).cuda()
        linear = torch.nn.Sequential(
            *(replace_layers(net.linear, i, [layer_index], \
                [new_linear_layer]) for i, _ in enumerate(net.linear)))
        del net.linear
        net.linear = linear    
    del net.features    
    del conv
    net.features = features      
    return net
 