import torch
import torch.nn as nn
from torchvision import models


class TeacherModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        name = cfg.model.teacher.name
        num_classes = cfg.model.num_classes
        pretrained = cfg.model.teacher.pretrained

        if name == "resnet50":
            self.backbone = models.resnet50(pretrained=pretrained)
            in_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Linear(in_features, num_classes)
       
 
        elif name == "resnet101":
            self.backbone = models.resnet101(pretrained=pretrained)
            in_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Linear(in_features, num_classes)

        elif name == "resnet152":
            self.backbone = models.resnet152(pretrained=pretrained)
            in_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Linear(in_features, num_classes)


        else:
            raise ValueError(f"Unsupported teacher model: {name}")

        # Teacher is frozen by default
        for p in self.parameters():
            p.requires_grad = False

        self.eval()

    def forward(self, x):
        return self.backbone(x)