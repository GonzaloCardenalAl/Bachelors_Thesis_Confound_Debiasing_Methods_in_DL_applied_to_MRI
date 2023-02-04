import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from metadatanorm import MetadataNorm
from PMDNLayer import PMDNLayer
#from CNNpipeline.PMDNLayer import PMDNLayer
import math
from collections import OrderedDict
from _model_resnet import *
        

import math
from functools import partial

import os, sys
from os.path import join

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
#####################      
        

class FCN_3D(nn.Module):
    '''A class to build 3D CNN (fully-convolutional networks) model architectures on-the-fly
        
        Args::
            convs: A list that specifies (1) number of layers and 
                                          (2) number of conv-channels per layer in the model architecture.
                Ex: [16, 32, 64] creates 3 layer FCN_3D with 16, 32 and 64 conv. channels resp. followed 
                by a final convolutional layer to create class predictions.
                Each layer consists of a block of *Convolution-BatchNorm-ELU*.
                
            pools (optional): Can be 'max', 'avg' or False/None/''. Can be a list or a single value 
            in which case the same pooling is applied at the end of all conv layers in the network.
                    
            kernels (optional): kernel size to use at each convolutional layers. Can either be a list of 
                with same length as number of layers (convs) or a single kernel_size accepted by pytorch's conv
                layers.
                    
            dropout (optional): additionally add a dropout layer before each *Convolution-BatchNorm-ELU* block.
                The value between [0.,1.] represents the amount of 3D dropout to perform.
                the length of this list should be smaller than the length of 'convs' (obviously).
                To add dropout only before the first n layers, give a smaller list of len(dropout) < len(convs).
            
            in_shape (optional): The input shape of the images of format (im_x, im_y, im_z)
            out_class (optional): The number of output classes in the classification task
            debug_print (optional): prints shapes at every layer of the conv model for debugging
    '''
    def __init__(self, convs, pools='max', kernels=3, dropout=[], 
                 in_shape=(96, 114, 96), out_classes=2, 
                 debug_print=False):
        
        super().__init__()
        
        self.out_classes = out_classes
        # if a single kernel is given, make it a list of len equal to no. of layers 
        # with same kernel for all the layers
        if isinstance(kernels, int) or (isinstance(kernels,(list,tuple)) and len(kernels)==3 and isinstance(kernels[0],int)):
            kernels = [kernels]*len(convs)
        assert len(kernels)==len(convs), f"number of kernels given ({len(kernels)}) is not equal to number of layers i.e. len(convs)={len(convs)}"
        self.dropout = dropout        
        if isinstance(pools, str):
            pools = [pools]*len(convs)
        if len(pools)<len(convs):
            pools += ([False]*(len(convs)-len(pools)))
        
        self.debug_print = debug_print
        
        # build the convolutional layers
        self.convs = nn.ModuleList([])
        self.pools = [] # not a nn.ModuleList()
        self._conv_out_shape = np.array(in_shape)     
        
        for i, (cin, cout) in enumerate(zip([1]+convs, convs)):
            
            kernel = kernels[i]
            layers = []
            # add dropout layer if requested
            if i<len(self.dropout) and self.dropout[i]>0: 
                layers.append(nn.Dropout3d(p=self.dropout[i]))    
                                            
            layers.extend([nn.Conv3d(cin, cout, kernel_size=kernel),
                           nn.BatchNorm3d(cout),                     
                           nn.ELU()
                          ])
            self.convs.append(nn.Sequential(*layers))
                
            # dynamically calculate the output shape of convolutions + pooling
            self._conv_out_shape = (self._conv_out_shape-((np.array(kernel))-1)) # equations from https://medium.com/mlreview/a-guide-to-receptive-field-arithmetic-for-convolutional-neural-networks-e0f514068807   
            if self.debug_print: print(f"output shape = {self._conv_out_shape} \t after layer-{i} conv (cout={cout}, kernel={kernel})")
            
            # add the pooling layer
            if pools[i]=='max':
                self.pools.append(nn.MaxPool3d(kernel_size=2, stride=2)) # TODO kernel size can be made configurable too
                self._conv_out_shape //= 2
            elif pools[i]=='avg':                
                self.pools.append(nn.AvgPool3d(kernel_size=2, stride=2)) 
                self._conv_out_shape //= 2
            else:
                self.pools.append(False)
                
            if self.debug_print and pools[i]: print(f"output shape = {self._conv_out_shape} \t layer-{i} pool ({pools[i]}, kernel=2)")
            assert np.all(self._conv_out_shape>0), f"output shape at layer {i} has 0 or lower value/s = {self._conv_out_shape}"

        # set the last FCN layer kernel such that it produces a single output prediction
        self.finalconv = nn.Conv3d(convs[-1], self.out_classes, kernel_size=self._conv_out_shape)
        
        
    def _forward_convs(self, t):
        # loop over all convolution layers
        for i, (conv, pool) in enumerate(zip(self.convs, self.pools)):
            #  covolution + non-linear Elu operation
            t = conv(t)
            if self.debug_print: print("conv{}>{}".format(i,list(t.shape)))
            if pool:
                # perform maxpool with (2x2x2) kernels
                t = pool(t)
                if self.debug_print: print("pool{}>{}".format(i,list(t.shape)))
        return t
        
        
    def forward(self, t):
        # pass through all conv layers
        t = self._forward_convs(t)
        # final layer
        t = self.finalconv(t)
        
        if self.out_classes>1:
            t = t.reshape(-1, self.out_classes)
        else:
            # for BCEWithLogits compatibility convert to float
            t = t.reshape(-1).float() 
            
        if self.debug_print: 
            print("final>{}".format(list(t.shape)))
            self.debug_print = False   
        # no activations in the last layer since 
        # we use cross_entropy_loss with logits
        return t
        
###############################################################################    

class FCN_3D_hooked(FCN_3D):
    '''An additional 3-variable linear layer 'features_3D' is attached to FCN_3D just
    before the final prediction layer. This layer acts as a 3D feature space bottleneck
    that can be visualized. The output activations of this layer will be stored in an
    array 'self.features_3D_outputs' when 'set_hook'=True
    '''
    def __init__(self, convs, **kwargs):
        
        super().__init__(convs=convs, **kwargs) 
        
        # delete the previous final layer and replace with 2 layers
        self.features_3D = nn.Conv3d(convs[-1], 3, kernel_size=self._conv_out_shape)
        self.finalconv = nn.Conv3d(3, self.out_classes, kernel_size=1)
        
        self._reset_hook()
    
    
    def forward(self, t):
        # pass through all conv layers
        t = self._forward_convs(t)
        if self.debug_print: print("conv_out>{}".format(list(t.shape)))
        # apply the feature_3D layer before the final layer
        t = self.features_3D(t)
        if self.debug_print: print("features_3D>{}".format(list(t.shape)))
        t = self.finalconv(t)
        
        if self.out_classes>1:
            t = t.reshape(-1, self.out_classes)
        else:
            # for BCEWithLogits compatibility convert to float
            t = t.reshape(-1).float() 
            
        if self.debug_print: 
            print("final>{}".format(list(t.shape)))
            self.debug_print = False   
        return t
    
    
    def _hook_viz_layer(self, hook_func_lambda=None):
        # implements a hook (https://pytorch.org/docs/stable/generated/torch.nn.modules.module.register_module_forward_hook.html) on features_3D layer    
        def hook_func(model, input, output):
            # save output to an array
            self.features_3D_outputs = np.append(self.features_3D_outputs, 
                     output.detach().cpu().numpy().reshape(-1,3), axis=0)
            # if additional custom hook function operations are provided then also perform them
            if hook_func_lambda is not None: hook_func_lambda(model, input, output)
            
        self._hook_handler = self.features_3D.register_forward_hook(hook_func)
        
    
    def _reset_hook(self):
        if hasattr(self, '_hook_handler'):
            self._hook_handler.remove()
        self.features_3D_outputs = np.empty(shape=(0,3))
        
############################
        
        
class PMDN_model(nn.Module):

    def __init__(self, dataset_size, batch_size, convs, num_metadata = 1, pools='max', kernels=3, dropout=[], 
                 in_shape=(96, 114, 96), out_classes=2, debug_print=False):
        
        
        super().__init__()
        
        self.N = batch_size
        self.dataset_size = dataset_size
        self.number_of_input_channels = in_shape
        
        self.out_classes = out_classes
        self.cfs = nn.Parameter(torch.randn(batch_size, num_metadata), requires_grad=False) # confounders per batch
        #self.loss_terms = 0 # aggregates all losses
        self.use_pmdn_labels = False

        

        # if a single kernel is given, make it a list of len equal to no. of layers 
        # with same kernel for all the layers
        if isinstance(kernels, int) or (isinstance(kernels,(list,tuple)) and len(kernels)==3 and isinstance(kernels[0],int)):
            kernels = [kernels]*len(convs)
        assert len(kernels)==len(convs), f"number of kernels given ({len(kernels)}) is not equal to number of layers i.e. len(convs)={len(convs)}"
        self.dropout = dropout        
        if isinstance(pools, str):
            pools = [pools]*len(convs)
        if len(pools)<len(convs):
            pools += ([False]*(len(convs)-len(pools)))
        
        self.debug_print = debug_print
        
        # build the convolutional layers
        self.convs = nn.ModuleList([])
        self.pools = [] # not a nn.ModuleList()
        self._conv_out_shape = np.array(in_shape)     
        
        for i, (cin, cout) in enumerate(zip([1]+convs, convs)):
            
            kernel = kernels[i]
            layers = []
            
            self._conv_out_shape = (self._conv_out_shape-((np.array(kernel))-1)) # equations from https://medium.com/mlreview/a-guide-to-receptive-field-arithmetic-for-convolutional-neural-networks-e0f514068807 
            # add dropout layer if requested
            if i<len(self.dropout) and self.dropout[i]>0: 
                layers.append(("dropout", nn.Dropout3d(p=self.dropout[i])))    

            layers.extend([("conv1" , nn.Conv3d(cin, cout, kernel_size=kernel)), # full padding
                           ("layernorm1" , nn.LayerNorm(self._conv_out_shape)),                 
                           ("pmdn1", PMDNLayer(self.N, num_metadata,  np.prod(self._conv_out_shape) * cout)),
                           ("elu1", nn.ELU(inplace=False))])
            self.convs.append(nn.Sequential(OrderedDict(layers)))

                
            # dynamically calculate the output shape of convolutions + pooling
            
            if self.debug_print: print(f"output shape = {self._conv_out_shape} \t after layer-{i} conv (cout={cout}, kernel={kernel})")
            
            # add the pooling layer
            if pools[i]=='max':
                self.pools.append(nn.MaxPool3d(kernel_size=2, stride=2)) # TODO kernel size can be made configurable too
                self._conv_out_shape //= 2
            elif pools[i]=='avg':                
                self.pools.append(nn.AvgPool3d(kernel_size=2, stride=2)) 
                self._conv_out_shape //= 2
            else:
                self.pools.append(False)
                
            if self.debug_print and pools[i]: print(f"output shape = {self._conv_out_shape} \t layer-{i} pool ({pools[i]}, kernel=2)")
            assert np.all(self._conv_out_shape>0), f"output shape at layer {i} has 0 or lower value/s = {self._conv_out_shape}"

        # set the last FCN layer kernel such that it produces a single output prediction
        self.finalconv = nn.Conv3d(convs[-1], self.out_classes, kernel_size=self._conv_out_shape)
        
        
    def _forward_convs(self, t):
        
        loss_terms = 0
        # loop over all convolution layers
        for i, (conv, pool) in enumerate(zip(self.convs, self.pools)):
            #  covolution + non-linear Elu operation
            
            
            t =  conv._modules["conv1"](t)
            t =  conv._modules["layernorm1"](t)
            conv._modules["pmdn1"].cfs = self.cfs
            conv._modules["pmdn1"].use_labels = self.use_pmdn_labels
            t, loss_term =  conv._modules["pmdn1"](t)
            loss_terms += loss_term
            t =  conv._modules["elu1"](t)
            
            #t = conv(t)
            if self.debug_print: print("conv{}>{}".format(i,list(t.shape)))
            if pool:
                # perform maxpool with (2x2x2) kernels
                t = pool(t)
                if self.debug_print: print("pool{}>{}".format(i,list(t.shape)))
        return t, loss_terms
        
    """    
    def forward(self, t):
        
        self.loss_terms = 0
        # pass through all conv layers
        t, loss_terms = self._forward_convs(t)
        self.loss_terms += loss_terms
        # final layer
        t = self.finalconv(t)
        if self.debug_print: print("final>{}".format(list(t.shape)))
            
        t = t.reshape(-1, self.out_classes)
        # no activations in the last layer since 
        # we use cross_entropy_loss with logits
        if self.debug_print: self.debug_print = False      
            
            
         
        # Avereage loss
        self.loss_terms = 1/len(convs) * self.loss_terms   
            
        
        return t
      """
    
    
    
    
 ###############################################################################    

class PMDN_model_hooked(PMDN_model):

    def __init__(self, convs, **kwargs):
        
        super().__init__(convs=convs, **kwargs) 
        
        # delete the previous final layer and replace with 2 layers
        self.features_3D = nn.Conv3d(convs[-1], 3, kernel_size=self._conv_out_shape)
        self.finalconv = nn.Conv3d(3, self.out_classes, kernel_size=1)
        self.loss_terms = 0 # aggregates all losses
        self.num_conv_blocks = len(convs)
        self._reset_hook()
    
    
    def forward(self, t):
        self.loss_terms = 0
        # pass through all conv layers
        t , loss_terms= self._forward_convs(t)
        self.loss_terms += loss_terms
        if self.debug_print: print("conv_out>{}".format(list(t.shape)))
        # apply the feature_3D layer before the final layer
        t = self.features_3D(t)
        if self.debug_print: print("features_3D>{}".format(list(t.shape)))
        t = self.finalconv(t)
        
        if self.out_classes>1:
            t = t.reshape(-1, self.out_classes)
        else:
            # for BCEWithLogits compatibility convert to float
            t = t.reshape(-1).float() 
            
        if self.debug_print: 
            print("final>{}".format(list(t.shape)))
            self.debug_print = False   
            
        self.loss_terms = 1/self.num_conv_blocks * self.loss_terms  
        return t
    
    
    def _hook_viz_layer(self, hook_func_lambda=None):
        # implements a hook (https://pytorch.org/docs/stable/generated/torch.nn.modules.module.register_module_forward_hook.html) on features_3D layer    
        def hook_func(model, input, output):
            # save output to an array
            self.features_3D_outputs = np.append(self.features_3D_outputs, 
                     output.detach().cpu().numpy().reshape(-1,3), axis=0)
            # if additional custom hook function operations are provided then also perform them
            if hook_func_lambda is not None: hook_func_lambda(model, input, output)
            
        self._hook_handler = self.features_3D.register_forward_hook(hook_func)
        
    
    def _reset_hook(self):
        if hasattr(self, '_hook_handler'):
            self._hook_handler.remove()
        self.features_3D_outputs = np.empty(shape=(0,3))   
        
####################################################
#This one just have 1 PMDN Layer implemented
class PMDNResNet50DeepRepViz(nn.Module):  
    def __init__(self, out_classes=1, 
                 task_type='classif', 
                 freeze_feature_extractor=False, 
                 pretrained_model='', 
                 debug_print=False,
                 dataset_size = None,
                 batch_size = 12,
                 num_metadata = 3,
                 in_shape=(96, 114, 96)):
        
      #this PMDNResNet only works with this in_shape, if you want to change shape, change shape in PMDN layer
        """Args::
            out_classes: Number of classes your label has, for regression 'out_classes'=1. *Warning: for classif_binary the out_classes should also be 1.
            
            task_type: 'classif_binary','classif','regression'.
            
            freeze_feature_extractor(optional): If True, It will freeze all the layers in the ResNet except for the last linear layer.
            
            pretrained_model (optional):'Add a path to a file and loads parameters from a trained model, 
                                          e.g pretrained_model ='/ritter/share/data/UKBB_2020/trained_model/r3d50_K_200ep.pth'
            
            debug_print(optional): If True, It will print all the layers name and weigths freeze.
        """
        super(PMDNResNet50DeepRepViz, self).__init__()
        
        #Paramters for the PMDN layer
        
        self.N = batch_size
        self.dataset_size = dataset_size
        self.in_shape = in_shape
        self.num_metadata = num_metadata

        self.out_classes =  out_classes
        self.feature_extractor = PMDNResNet(Bottleneck, [3, 4, 6, 3], get_inplanes(), out_classes, 
                                          in_shape=in_shape,batch_size=batch_size,dataset_size=dataset_size)
        self.task_type = task_type
        self.pretrained_model = pretrained_model
        self.n=256
        
        self._reset_hook()
        
        # Loading pretrained weights if provided
        if self.pretrained_model: 
            
            state_dict = torch.load(self.pretrained_model)
            
            #In case the paremeters dictionary containing the state_dict is written with more info about the training
            if 'state_dict' in state_dict:
                state_dict = torch.load(self.pretrained_model)['state_dict']
                
            # pretrained model expect 3 input_channels [R,G,B] if it was trained on videos 
            n_in_chns = state_dict['conv1.weight'].shape[1]
            # a work around for this is to squeeze the 3 channels into one as suggested in
            # https://stackoverflow.com/a/54777347/3293428
            if n_in_chns==3:
                state_dict['conv1.weight']= state_dict['conv1.weight'].sum(dim=1, keepdim=True)
                
            log = self.feature_extractor.load_state_dict(state_dict, strict=False)
            if debug_print:
                if log.missing_keys: print(f"missing_keys in current model: {log.missing_keys}")
                if log.unexpected_keys: print(f"unexpected_keys in pretrained model: {log.unexpected_keys}")
            
        # Adding linear layer
        block = Bottleneck
        block_inplanes = get_inplanes()
        self.features_3D = nn.Linear(block_inplanes[3] * block.expansion, self.n)
        self.relu = nn.ReLU(inplace=True)
        self.classifier = nn.Linear(self.n, out_classes)
    
        ####Freezing the pretrain weights
        if freeze_feature_extractor == True:
            for name, layer in self.feature_extractor.named_parameters():
                layer.requires_grad = False
  
                if debug_print == True and "bias" not in name and "bn" not in name: 
                    print(f"layer {name.replace('.weight','')}({list(layer.shape)}) was frozen")

    def forward(self, x):
        self.loss_terms = 0
        h, loss_terms = self.feature_extractor(x)
        t = self.features_3D(h)
        t = self.relu(t)
        out = self.classifier(t)
        if self.task_type=='regression' or self.out_classes<=2: 
            out = out.squeeze()
        if self.task_type=='regression':
            out = out.float()
            
        self.loss_terms += loss_terms  #if I want to get more pmdn layers I should divide them into the number of pmdn layer
        return out
    
    def _hook_viz_layer(self, hook_func_lambda=None):
        # implements a hook (https://pytorch.org/docs/stable/generated/torch.nn.modules.module.register_module_forward_hook.html) on features_3D layer    
        def hook_func(model, input, output):
            # save output to an array
            self.features_3D_outputs = np.append(self.features_3D_outputs, 
                     output.detach().cpu().numpy().reshape(-1,self.n), axis=0)
            # if additional custom hook function operations are provided then also perform them
            if hook_func_lambda is not None: hook_func_lambda(model, input, output)
            
        self._hook_handler = self.features_3D.register_forward_hook(hook_func)
        
    
    def _reset_hook(self):
        if hasattr(self, '_hook_handler'):
            self._hook_handler.remove()
        self.features_3D_outputs = np.empty(shape=(0,self.n))
        
#######################

class PMDNResNet(nn.Module):

    def __init__(self,
                 block,
                 layers,
                 block_inplanes,
                 out_classes=1,
                 n_input_channels=1,
                 conv1_t_size=7,
                 conv1_t_stride=1,
                 no_max_pool=False,
                 shortcut_type='B',
                 widen_factor=1.0,
                 
                 batch_size= 12, 
                 dataset_size = 1000,
                 num_metadata = 3,
                 in_shape=(96,114,96)):
        super().__init__()

        block_inplanes = [int(x * widen_factor) for x in block_inplanes]
        
        self.N = batch_size
        self.dataset_size = dataset_size
        self.in_shape = in_shape
        self.num_metadata = num_metadata
        
        self.cfs = nn.Parameter(torch.randn(batch_size, num_metadata), requires_grad=False) # confounders per batch
        #self.loss_terms = 0 # aggregates all losses
        self.use_pmdn_labels = False
        
        self.in_shape = in_shape
        self.in_planes = block_inplanes[0]
        self.no_max_pool = no_max_pool

        self.conv1 = nn.Conv3d(n_input_channels,
                               self.in_planes,
                               kernel_size=(conv1_t_size, 7, 7),
                               stride=(conv1_t_stride, 2, 2),
                               padding=(conv1_t_size // 2, 3, 3),
                               bias=False)
        
        self.bn1 = nn.BatchNorm3d(self.in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, block_inplanes[0], layers[0],
                                       shortcut_type)
        self.layer2 = self._make_layer(block,
                                       block_inplanes[1],
                                       layers[1],
                                       shortcut_type,
                                       stride=2)
        self.layer3 = self._make_layer(block,
                                       block_inplanes[2],
                                       layers[2],
                                       shortcut_type,
                                       stride=2)
        self.layer4 = self._make_layer(block,
                                       block_inplanes[3],
                                       layers[3],
                                       shortcut_type,
                                       stride=2)
        
        ###aqui tiene que ir la PMDN Layer
        self.ln1 = nn.LayerNorm(normalized_shape=[6, 4, 3])
        self.pmdn1 = PMDNLayer(batch_size, num_metadata,  np.prod([6, 4, 3]) * self.in_planes)  

        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        #self.fc = nn.Linear(block_inplanes[3] * block.expansion, out_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight,
                                        mode='fan_out',
                                        nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _downsample_basic_block(self, x, planes, stride):
        out = F.avg_pool3d(x, kernel_size=1, stride=stride)
        zero_pads = torch.zeros(out.size(0), planes - out.size(1), out.size(2),
                                out.size(3), out.size(4))
        if isinstance(out.data, torch.cuda.FloatTensor):
            zero_pads = zero_pads.cuda()

        out = torch.cat([out.data, zero_pads], dim=1)

        return out

    def _make_layer(self, block, planes, blocks, shortcut_type, stride=1):
        downsample = None
        if stride != 1 or self.in_planes != planes * block.expansion:
            if shortcut_type == 'A':
                downsample = partial(self._downsample_basic_block,
                                     planes=planes * block.expansion,
                                     stride=stride)
            else:
                downsample = nn.Sequential(
                    conv1x1x1(self.in_planes, planes * block.expansion, stride),
                    nn.BatchNorm3d(planes * block.expansion))

        layers = []
        layers.append(
            block(in_planes=self.in_planes,
                  planes=planes,
                  stride=stride,
                  downsample=downsample))
        self.in_planes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.in_planes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        loss_terms = 0
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        if not self.no_max_pool:
            x = self.maxpool(x)
    
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.ln1(x)
        self.pmdn1.cfs = self.cfs
        self.pmdn1.use_labels = self.use_pmdn_labels
        x, loss_term = self.pmdn1(x)
        loss_terms += loss_term
        
        x = self.avgpool(x)

        x = x.view(x.size(0), -1)
        #x = self.fc(x)

        return x, loss_terms #.squeeze()