import torch
import torch.nn as nn

#########################################
#       Improve this basic model!       #
#########################################
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.layer1 = nn.Sequential(
            nn.Conv2d(12, 32, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))  # 32 channel, 64x64 images
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))  # 64 channel, 32x32 images
        self.drop_out = nn.Dropout()
        self.fc1 = nn.Linear(32 * 32 * 64, 1000)
        self.fc2 = nn.Linear(1000, 24 * 64 * 64)
        
        #super().__init__()

        #self.layer1 = nn.Linear(in_features=12 * 128 * 128, out_features=256)
        #self.layer2 = nn.Linear(in_features=256, out_features=256)
        #self.layer3 = nn.Linear(in_features=256, out_features=24 * 64 * 64)

    def forward(self, features):
        x = features.view(-1, 12, 128, 128) / 1024.0
        x = self.layer1(x)
        x = self.layer2(x)
        x = x.reshape(x.size(0), -1)
        x = self.drop_out(x)
        x = self.fc1(x)
        x = self.fc2(x)
        
        return x.view(-1, 24, 64, 64) * 1024.0
        
        #x = features.view(-1, 12 * 128 * 128) / 1024.0
        #x = torch.relu(self.layer1(x))
        #x = torch.relu(self.layer2(x))
        #x = torch.relu(self.layer3(x))

        #return x.view(-1, 24, 64, 64) * 1024.0

# class BaseModel(nn.Module):
#     def __init__(self):
#         super().__init__()

#         self.layer1 = nn.Linear(in_features=12 * 128 * 128, out_features=256)
#         self.layer2 = nn.Linear(in_features=256, out_features=256)
#         self.layer3 = nn.Linear(in_features=256, out_features=24 * 64 * 64)

#     def forward(self, features):
#         x = features.view(-1, 12 * 128 * 128) / 1024.0
#         x = torch.relu(self.layer1(x))
#         x = torch.relu(self.layer2(x))
#         x = torch.relu(self.layer3(x))

#         return x.view(-1, 24, 64, 64) * 1024.0
