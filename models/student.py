import torch
import torch.nn as nn
from torchvision import models

from utils.seed import  set_seed
from torchvision.models.vision_transformer import vit_b_16,vit_l_16,ViT_B_16_Weights
from torchvision.models import alexnet,AlexNet_Weights,resnet50,resnet152,ResNet50_Weights
from torchvision.models.swin_transformer import swin_s

import torch
import torch.nn as nn
from torchvision import models
from torchvision.models.vision_transformer import vit_b_16,vit_l_16,ViT_B_16_Weights
from torchvision.models import alexnet,AlexNet_Weights,resnet50,resnet152,ResNet50_Weights
from torchvision.models.swin_transformer import swin_s


set_seed(42)

class StudentModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        name = cfg["model"]["student"]
        num_classes = cfg["model"]["num_classes"]
        pretrained = cfg["model"]["pretrained"]
        set_seed(42)
        if name == "vit_b_16":
            vitbasedistillation = vit_b_16(ViT_B_16_Weights,image_size=224)
            vitbasedistillation.heads.head = nn.Linear(vitbasedistillation.heads.head.in_features, num_classes)
            vitbasedistillation.num_classes=num_classes
            self.backbone = vitbasedistillation
        elif name == "vit_l_16":
            self.backbone = models.vit_l_16(pretrained=pretrained)
            in_features = self.backbone.heads.head.in_features
            self.backbone.heads.head = nn.Linear(in_features, num_classes)

        elif name == "swin_s":
            self.backbone = models.swin_s(pretrained=pretrained)
            in_features = self.backbone.head.in_features
            self.backbone.head = nn.Linear(in_features, num_classes)

        else:
            raise ValueError(f"Unsupported student model: {name}")

    def forward(self, x):
        """
        Returns logits only (KD features handled via hooks later)
        """
        return self.backbone(x)