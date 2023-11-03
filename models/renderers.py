import torch 
import torch.nn as nn 
import torch.nn.functional as F
import einops 

class FCN(nn.Module):
    def __init__(self):
        super(FCN, self).__init__()
        self.fc1 = (nn.Linear(10, 512))
        self.fc2 = (nn.Linear(512, 1024))
        self.fc3 = (nn.Linear(1024, 2048))
        self.fc4 = (nn.Linear(2048, 4096))
        self.conv1 = (nn.Conv2d(16, 32, 3, 1, 1))
        self.conv2 = (nn.Conv2d(32, 32, 3, 1, 1))
        self.conv3 = (nn.Conv2d(8, 16, 3, 1, 1))
        self.conv4 = (nn.Conv2d(16, 16, 3, 1, 1))
        self.conv5 = (nn.Conv2d(4, 8, 3, 1, 1))
        self.conv6 = (nn.Conv2d(8, 4, 3, 1, 1))
        self.pixel_shuffle = nn.PixelShuffle(2)

    def forward(self, x):
        # x is shape [batch, 10]
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = x.view(-1, 16, 16, 16)
        x = F.relu(self.conv1(x)) # [N, 32, 16, 16]
        x = self.pixel_shuffle(self.conv2(x)) # [N, 8, 32, 32]
        x = F.relu(self.conv3(x)) # [N, 16, 32, 32]
        x = self.pixel_shuffle(self.conv4(x)) # [N, 4, 64, 64]
        x = F.relu(self.conv5(x)) # [N, 8, 64, 64]
        x = self.pixel_shuffle(self.conv6(x)) # CONV6 -> # [N, 4, 64, 64], PIXEL SHUFFLE -> [N, 1, 128, 128]
        x = torch.sigmoid(x)
        return 1 - x.view(-1, 128, 128) 


class FCN_2outputs(nn.Module):
    def __init__(self, num_inputs):
        super(FCN_2outputs, self).__init__()
        self.fc1 = (nn.Linear(num_inputs, 512))
        self.fc2 = (nn.Linear(512, 1024))
        self.fc3 = (nn.Linear(1024, 2048))
        self.fc4 = (nn.Linear(2048, 4096))

        # Binary mask 
        self.conv1 = (nn.Conv2d(16, 32, 3, 1, 1))
        self.conv2 = (nn.Conv2d(32, 32, 3, 1, 1))
        self.conv3 = (nn.Conv2d(8, 16, 3, 1, 1))
        self.conv4 = (nn.Conv2d(16, 16, 3, 1, 1))
        self.conv5 = (nn.Conv2d(4, 8, 3, 1, 1))
        self.conv6 = (nn.Conv2d(8, 4, 3, 1, 1))
        
        # Texture branch
        self.conv1t = (nn.Conv2d(16+1, 32, 3, 1, 1))
        self.conv2t = (nn.Conv2d(32, 32, 3, 1, 1))
        self.conv3t = (nn.Conv2d(8, 16, 3, 1, 1))
        self.conv4t = (nn.Conv2d(16, 16, 3, 1, 1))
        self.conv5t = (nn.Conv2d(4, 8, 3, 1, 1))
        self.conv6t = (nn.Conv2d(8, 4, 3, 1, 1))
        
        self.pixel_shuffle = nn.PixelShuffle(2)


    def forward(self, x, idx_cls=1.0):
        
        # x is shape [batch, num_inputs]
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))

        base = x.view(-1, 16, 16, 16)
        #idx_cls = einops.repeat(idx_cls, 'm -> m n k l', n=1, k=16, l=16)
        #base_text = torch.cat((base, idx_cls), dim=1)

        # Binary branch 
        x = F.relu(self.conv1(base))
        #print('x shape after conv 1: ', x.shape)
        x = self.pixel_shuffle(self.conv2(x))
        #print('x shape after conv 2: ', x.shape)
        x = F.relu(self.conv3(x))
        #print('x shape after conv 3: ', x.shape)
        x = self.pixel_shuffle(self.conv4(x))
        #print('x shape after conv 4: ', x.shape)
        x = F.relu(self.conv5(x))
        #print('x shape after conv 5: ', x.shape)
        x = self.pixel_shuffle(self.conv6(x))
        #print('x shape after conv 6: ', x.shape)
        
        # Texture branch 
        """
        xt = F.relu(self.conv1t(base_text))
        xt = self.pixel_shuffle(self.conv2t(xt))
        xt = F.relu(self.conv3t(xt))
        xt = self.pixel_shuffle(self.conv4t(xt))
        xt = F.relu(self.conv5t(xt))
        xt = self.pixel_shuffle(self.conv6t(xt))
        xt = torch.sigmoid(xt)
        """
        x = torch.sigmoid(x)
        
        return 1 - x.view(-1, 128, 128) #, 1 - xt # 1 - sigmoid flips the range [0, 0.25, 0.5, 0.75, 1] becomes [1, 0.75, 0.5, 0.25, 0]]