import torch
import torch.nn as nn

class C3D_UCF101(nn.Module):
    
    def __init__(self, d=3, architecture='HOMOGENEOUS', mode=None):
        super(C3D_UCF101, self).__init__()
        
        if architecture == 'HOMOGENEOUS':
            depths = [d] * 5
        elif architecture == 'VARYING':                
            if mode == 'increasing':
                depths = [3, 3, 5, 5, 7]
            elif mode == 'decreasing':
                depths = [7, 5, 5, 3, 3]
            else:
                raise ValueError("mode must be "
                                 "'increasing' or 'decreasing'")
        else:
            raise ValueError("architecture must be "
                             "'HOMOGENEOUS' or 'VARYING'")
                
        filters = [64, 128, 256, 256, 256]
        
        # Conv Layer
        self.conv1 = nn.Conv3d(3, filters[0], depths[0], 1, 1)
        self.conv2 = nn.Conv3d(filters[0], filters[1], depths[1], 1, 1)
        self.conv3 = nn.Conv3d(filters[1], filters[2], depths[2], 1, 1)
        self.conv4 = nn.Conv3d(filters[2], filters[3], depths[3], 1, 1)
        self.conv5 = nn.Conv3d(filters[3], filters[4], depths[4], 1, 1)
        
        # MaxPool Layer
        self.pool1 = nn.MaxPool3d((1,2,2), 1)
        self.pool2 = nn.MaxPool3d((2,2,2), 1)
        self.pool3 = nn.MaxPool3d((2,2,2), 1)
        self.pool4 = nn.MaxPool3d((2,2,2), 1)
        self.pool5 = nn.MaxPool3d((2,2,2), 1)
        
        # Fully Connected Layer
        self.fc1 = nn.Linear(4096, 2048)
        self.fc2 = nn.Linear(2048, 2048)
        self.fc3 = nn.Linear(2048, 101)
        
    def forward(self, x):
        # Conv&MaxPool Blocks
        x = self.pool1(self.conv1(x))
        x = self.pool2(self.conv2(x))
        x = self.pool3(self.conv3(x))
        x = self.pool4(self.conv4(x))
        x = self.pool5(self.conv5(x))
        
        # Concatenate
        x = x.view(-1, self.num_flat_features(x))
        
        # Fully Connected
        x = self.fc1(x)
        x = self.fc2(x)
        
        # Output
        out = self.fc3(x)
        return out
        
    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features
