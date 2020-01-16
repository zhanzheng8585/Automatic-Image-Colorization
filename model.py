import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class ColorizationNet(nn.Module):
    def __init__(self, midlevel_input_size=128, global_input_size=512):
        super(ColorizationNet, self).__init__()
        # Fusion layer to combine midlevel and global features
        self.midlevel_input_size = midlevel_input_size
        self.global_input_size = global_input_size
        # self.fusion = nn.Linear(midlevel_input_size + global_input_size, midlevel_input_size)
        self.fusion = nn.Linear(midlevel_input_size + global_input_size, midlevel_input_size + global_input_size)

        # self.bn1 = nn.BatchNorm1d(midlevel_input_size)
        self.bn1 = nn.BatchNorm1d(midlevel_input_size + global_input_size)

        # Convolutional layers and upsampling
        # self.deconv1_new = nn.ConvTranspose2d(midlevel_input_size, 128, kernel_size=4, stride=2, padding=1)
        self.deconv1_new = nn.ConvTranspose2d(midlevel_input_size + global_input_size, midlevel_input_size + global_input_size, kernel_size=4, stride=2, padding=1)
        # self.conv1 = nn.Conv2d(midlevel_input_size, 128, kernel_size=3, stride=1, padding=1)
        # self.bn2 = nn.BatchNorm2d(128)
        # self.conv2 = nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1)
        # self.bn3 = nn.BatchNorm2d(64)
        # self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        # self.bn4 = nn.BatchNorm2d(64)
        # self.conv4 = nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1)
        # self.bn5 = nn.BatchNorm2d(32)
        # self.conv5 = nn.Conv2d(32, 2, kernel_size=3, stride=1, padding=1)
        # self.upsample = nn.Upsample(scale_factor=2)
        self.conv1 = nn.Conv2d(midlevel_input_size + global_input_size, 512, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(512)
        self.conv2 = nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        self.conv3 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        self.conv4 = nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1)
        self.bn5 = nn.BatchNorm2d(128)
        self.conv5 = nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1)
        self.bn6 = nn.BatchNorm2d(64)
        self.conv6 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn7 = nn.BatchNorm2d(64)
        self.conv7 = nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1)
        self.bn8 = nn.BatchNorm2d(32)
        self.conv8 = nn.Conv2d(32, 2, kernel_size=3, stride=1, padding=1)
        self.upsample = nn.Upsample(scale_factor=2)
        print('Loaded colorization net.')

    def forward(self, midlevel_input, global_input):

        # Convolutional layers and upsampling
        x = torch.cat((midlevel_input,global_input), 1) 
        # x = F.relu(self.bn1(self.fusion(x)))
        x = F.relu(self.bn1(self.deconv1_new(x)))
        x = self.upsample(x)
        x = F.relu(self.bn2(self.conv1(x)))
        x = self.upsample(x)
        x = F.relu(self.bn3(self.conv2(x)))
        x = F.relu(self.bn4(self.conv3(x)))
        x = self.upsample(x)
        x = F.relu(self.bn5(self.conv4(x)))
        x = self.upsample(x)
        x = F.relu(self.bn6(self.conv5(x)))
        x = F.relu(self.conv6(x))
        x = self.upsample(x)
        x = F.relu(self.conv7(x))
        x = self.upsample(self.conv8(x))
        return x


class ColorNet(nn.Module):
    def __init__(self):
        super(ColorNet, self).__init__()
        
        # Build ResNet and change first conv layer to accept single-channel input
        # resnet_gray_model = models.resnet18(num_classes=365)
        resnet_gray_model = models.resnet18(num_classes=365)
        # print(resnet_gray_model)
        resnet_gray_model.conv1.weight = nn.Parameter(resnet_gray_model.conv1.weight.sum(dim=1).unsqueeze(1).data)
        # print(resnet_gray_model)

        # Only needed if not resuming from a checkpoint: load pretrained ResNet-gray model
        if torch.cuda.is_available(): # and only if gpu is available
            resnet_gray_weights = torch.load('pretrained/resnet_gray_weights.pth.tar') #torch.load('pretrained/resnet_gray.tar')['state_dict']
            resnet_gray_model.load_state_dict(resnet_gray_weights)
            print('Pretrained ResNet-gray weights loaded')

        # Extract midlevel and global features from ResNet-gray
        self.midlevel_resnet = nn.Sequential(*list(resnet_gray_model.children())[0:6])
        self.global_resnet = nn.Sequential(*list(resnet_gray_model.children())[0:9])
        self.fusion_and_colorization_net = ColorizationNet()

    def forward(self, input_image):

        # Pass input through ResNet-gray to extract features
        midlevel_output = self.midlevel_resnet(input_image)
        global_output = self.global_resnet(input_image)

        new_global_output = global_output.expand(-1, -1, 28, 28)
        # x = torch.cat((midlevel_output,new_global_output), 1)
        # print(x.size())
        # print(midlevel_output.size())
        # print(new_global_output.size())
        # Combine features in fusion layer and upsample
        output = self.fusion_and_colorization_net(midlevel_output, new_global_output)
        return output
