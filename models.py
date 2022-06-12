# =============================================================================
# Import required libraries
# =============================================================================
import torch
from torch import nn
import torchvision
import timm

import numpy as np


# =============================================================================
# CNNs
# =============================================================================
class ResNet101(nn.Module):
    def __init__(self, pretrained, with_fc=False, sam=False):
        super(ResNet101, self).__init__()
        self.path = './checkpoints/ResNet101_Corel-5K.pth'
        self.name = 'ResNet101'
        self.with_fc = with_fc
        self.sam = sam  # semantic attention module

        resnet = torchvision.models.resnet101(pretrained=pretrained)
        #
        self.features = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2,
            resnet.layer3,
            resnet.layer4,
        )
        self.pooling = nn.AdaptiveAvgPool2d(1)
        if with_fc:
            self.fc = nn.Sequential(
                nn.Dropout(p=0.3),
                nn.Linear(in_features=2048, out_features=1024)
            )

    def forward(self, img):
        feature = self.features(img)
        if self.sam is False:
            feature = self.pooling(feature)
            feature = torch.flatten(feature, 1)
        if self.with_fc:
            feature = self.fc(feature)
        return feature


class ResNeXt50(nn.Module):
    def __init__(self, pretrained, with_fc=False, sam=False):
        super(ResNeXt50, self).__init__()
        self.path = './checkpoints/ResNext50_Corel-5K.pth'
        self.name = 'ResNeXt50'
        self.with_fc = with_fc
        self.sam = sam  # semantic attention module

        resnext = torchvision.models.resnext50_32x4d(pretrained=pretrained)
        #
        self.features = nn.Sequential(
            resnext.conv1,
            resnext.bn1,
            resnext.relu,
            resnext.maxpool,
            resnext.layer1,
            resnext.layer2,
            resnext.layer3,
            resnext.layer4,
        )
        self.pooling = nn.AdaptiveAvgPool2d(1)
        if with_fc:
            self.fc = nn.Sequential(
                nn.Dropout(p=0.3),
                nn.Linear(in_features=2048, out_features=1024)
            )

    def forward(self, img):
        feature = self.features(img)
        if self.sam is False:
            feature = self.pooling(feature)
            feature = torch.flatten(feature, 1)
        if self.with_fc:
            feature = self.fc(feature)
        return feature


class Xception(nn.Module):
    def __init__(self, pretrained, with_fc=False, sam=False):
        super(Xception, self).__init__()
        self.path = './checkpoints/Xception_Corel-5K.pth'
        self.name = 'Xception'
        self.with_fc = with_fc
        self.sam = sam  # semantic attention module

        xception = timm.create_model('xception', pretrained=pretrained)
        #
        self.features = nn.Sequential(
            *(list(xception.children())[:-2]),
        )
        self.pooling = nn.AdaptiveAvgPool2d(1)
        if with_fc:
            self.fc = nn.Sequential(
                nn.Dropout(p=0.3),
                nn.Linear(in_features=2048, out_features=1024)
            )

    def forward(self, img):
        feature = self.features(img)
        if self.sam is False:
            feature = self.pooling(feature)
            feature = torch.flatten(feature, 1)
        if self.with_fc:
            feature = self.fc(feature)
        return feature


# =============================================================================
# GCN
# =============================================================================
class GraphConvolution(nn.Module):
    def __init__(self, in_features, out_features, bias=False):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.Tensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(1, 1, out_features))
        else:
            self.register_parameter('bias', None)
        self.init_parameters()

    def init_parameters(self):
        stdv = 1. / np.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.matmul(input, self.weight)
        output = torch.matmul(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output


# =============================================================================
# CNN-GCN
# =============================================================================
class GCNCNN(nn.Module):
    def __init__(self, model, input_d=300, middle_d=1024, output_d=2048):
        super(GCNCNN, self).__init__()
        self.gcn1_path = './checkpoints/' + model.name + '_GCN1_Corel-5K.pth'
        self.gcn2_path = './checkpoints/' + model.name + '_GCN2_Corel-5K.pth'

        # CNN model
        self.cnn = model

        self.gc1 = GraphConvolution(input_d, middle_d)
        self.gc2 = GraphConvolution(middle_d, output_d)
        self.leakyrelu = nn.LeakyReLU(0.2)

        if torch.cuda.is_available():
            self.cnn.cuda()
            self.gc1.cuda()
            self.gc2.cuda()

    def forward(self, img, emb, adj):
        if torch.cuda.is_available():
            emb = emb.cuda()
            adj = adj.cuda()

        feature = self.cnn(img)

        new_emb = self.gc1(emb, adj)
        new_emb = self.leakyrelu(new_emb)
        new_emb = self.gc2(new_emb, adj)

        new_emb = new_emb.transpose(0, 1)
        output = torch.matmul(feature, new_emb)
        return output

    def get_emb(self, emb, adj):
        if torch.cuda.is_available():
            emb = emb.cuda()
            adj = adj.cuda()
        new_emb = self.gc1(emb, adj)
        new_emb = self.leakyrelu(new_emb)
        new_emb = self.gc2(new_emb, adj)
        return new_emb

    def get_config_optim(self, lr):
        return [
            {'params': self.cnn.parameters(), 'lr': lr * 0.5},
            {'params': self.gc1.parameters(), 'lr': lr},
            {'params': self.gc2.parameters(), 'lr': lr},
        ]

    def save(self, path_CNN):
        torch.save(self.cnn.state_dict(), path_CNN)
        torch.save(self.gc1.state_dict(), self.gcn1_path)
        torch.save(self.gc2.state_dict(), self.gcn2_path)
