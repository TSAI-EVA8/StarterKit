import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.nn.functional as F

## we are creating our own residual block..this will generate a layer or layer with a residual connection
class UltimusBlock(nn.Module):
    def __init__(self,device):
        super(UltimusBlock, self).__init__()
        
        self.fc_k=nn.Linear(48,8)
        self.fc_q=nn.Linear(48,8)
        self.fc_v=nn.Linear(48,8)
        self.scale=torch.sqrt(torch.FloatTensor([8])).to(device)
        self.fc_zout=nn.Linear(8,48)
        
        

    def forward(self,x):
        x = x.view(-1, 48)
        #print("x.shapein ultima",x.shape)
        K=self.fc_k(x)
        Q=self.fc_q(x)
        V=self.fc_v(x)
        #print("Q<K<V",Q.shape,K.shape,V.shape)
        
        #scale=np.sqrt(8)
        AM=torch.softmax(torch.matmul(Q.T,K)/self.scale,dim=-1)
        #print(AM.shape)
        Z=torch.matmul(V,AM)
        #print("Z",Z.shape)
        out=self.fc_zout(Z)
        #print("out shape",out.shape)
        return out
        

class UltimusModel(nn.Module):

    def __init__(self,device):
        """ This function instantiates all the model layers """

        super(UltimusModel, self).__init__()
    
        self.convblock1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1),  # Input: 32x32x3 | Output: 32x32x32 | RF: 3x3
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3,padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Conv2d(in_channels=32, out_channels=48, kernel_size=3,padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(48)
           
        )
        
        self.gap = nn.Sequential(
            nn.AdaptiveAvgPool2d(1)
        ) 

        self.ultimusLayer1=nn.Sequential(
            UltimusBlock(device)
        )
        
        self.ultimusLayer2=nn.Sequential(
            UltimusBlock(device)
        )
        
        self.ultimusLayer3=nn.Sequential(
            UltimusBlock(device)
        )
        
        self.ultimusLayer4=nn.Sequential(
            UltimusBlock(device)
        )
        
         # Input: 4x4x64 | Output: 1x1x64 | RF: 108x108

        self.fc = nn.Sequential(
             nn.Linear(48, 10)
        )

        

    def forward(self, x):
        """ This function defines the network structure """
        #print("shape before conv",x.shape)
        x = self.convblock1(x)
        #print("shape after conv",x.shape)
        x = self.gap(x)
        #print("shape after gap",x.shape)
        x = self.ultimusLayer1(x)
        x = self.ultimusLayer2(x)
        x = self.ultimusLayer3(x)
        x = self.ultimusLayer4(x)
#         #x = x.view(-1, 48)
        x=self.fc(x)
        return x
        #x = self.fc(x)
        #return x