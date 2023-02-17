import torch
import torch.nn as nn
import torch.nn.functional as F

from StarterKit.models.resnet18 import BasicBlock



# class BasicResnetBlock(nn.Module):
#     expansion = 1

#     def __init__(self, in_planes, planes, stride=1):
#         super(BasicBlock, self).__init__()
#         self.conv1 = nn.Conv2d(
#             in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
#         self.bn1 = nn.BatchNorm2d(planes)
#         self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
#                                stride=1, padding=1, bias=False)
#         self.bn2 = nn.BatchNorm2d(planes)

#         self.shortcut = nn.Sequential()
#         if stride != 1 or in_planes != self.expansion*planes:
#             self.shortcut = nn.Sequential(
#                 nn.Conv2d(in_planes, self.expansion*planes,
#                           kernel_size=1, stride=stride, bias=False),
#                 nn.BatchNorm2d(self.expansion*planes)
#             )

#     def forward(self, x):
#         out = F.relu(self.bn1(self.conv1(x)))
#         out = self.bn2(self.conv2(out))
#         out += self.shortcut(x)
#         out = F.relu(out)
#         return out





## we are creating our own residual block..this will generate a layer or layer with a residual connection
class customResidualBlock(nn.Module):
    def __init__(self,in_planes,planes, res_block=None):
        super(customResidualBlock, self).__init__()
        
        self.layer=self._make_layer(in_planes,planes)

        self.res_block=None
        if not res_block is None:
            self.res_block=nn.Sequential(
                res_block(planes,planes) ## we will call the residual connection of the actual Resnet model as passed in the arguments
            )

    def _make_layer(self,in_channels,out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels,out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.MaxPool2d(2,2),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()            
        )
        
       

    def forward(self,x):
        x=self.layer(x)
        #print("sdd")
        if not self.res_block is None:
            x=x+self.res_block(x)
        return x





class customResnet(nn.Module):
    
    def __init__(self, custom_block, resnet_block):
        super(customResnet, self).__init__()
        #self.custom_block=customResidualBlock
        
        
        #customResidualBlock
        
        self.prep_layer=nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU()            
        )
        
        self.block_layers=nn.Sequential(
            ## the following three functions will be done in the custom block
            #X = Conv 3x3 (s1, p1) >> MaxPool2D >> BN >> RELU [128k]
            #R1 = ResBlock( (Conv-BN-ReLU-Conv-BN-ReLU))(X) [128k] 
            #Add(X, R1)
            custom_block(64,128,res_block=resnet_block),
            
            #X = Conv 3x3 (s1, p1) >> MaxPool2D >> BN >> RELU [256]
            custom_block(128,256),
            
            #X = Conv 3x3 (s1, p1) >> MaxPool2D >> BN >> RELU [512k]
            # R2 = ResBlock( (Conv-BN-ReLU-Conv-BN-ReLU))(X) [512k]
            # Add(X, R2)
            custom_block(256,512,res_block=resnet_block)
            
        ) 
        
        
        
        self.pool=nn.MaxPool2d(4, 4)
        self.fc=nn.Linear(512,10,bias=False)
        
    def forward(self,x):
        ## add the prep layer
        x=self.prep_layer(x)
        
        # ## add the blocks
        x=self.block_layers(x)
        
        ## add the pool
        x=self.pool(x)
        
        ## add the linear
        x=x.view(x.size(0),-1)
        x=self.fc(x) 
        
        return x

def CustomResidualModel():
    return customResnet(customResidualBlock, BasicBlock)