import timm 
import torch.nn as nn

'''
Author: Kayed Mahra
Wrapper class to create a eNet model using ImageNet pretrained weights.
Output dimension is changed to match the number of classes in the task.
Backbone could specify e.g eNetb0, eNetb1 etc.
'''
class CNN(nn.Module):
    def __init__(self, backbone, output_dimension):
        super(CNN, self).__init__()
        self.model = timm.create_model(backbone, pretrained=True, num_classes=output_dimension)

    def forward(self, x):
        x = self.model(x)
        return x