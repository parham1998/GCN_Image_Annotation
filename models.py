# =============================================================================
# Install necessary packages
# =============================================================================
# pip install inplace-abn
# pip install timm


# =============================================================================
# Import required libraries
# =============================================================================
import numpy as np
import torch
from torch import nn
import timm


# =============================================================================
# CNN
# =============================================================================
class TResNet(nn.Module):
    def __init__(self, args, pretrained):
        super(TResNet, self).__init__()
        self.path = args.save_dir + 'TResNet_Corel-5k.pth'

        tresnet = timm.create_model('tresnet_m', pretrained=pretrained)
        self.features = nn.Sequential(
            tresnet.body,
            tresnet.head.global_pool,
        )

    def forward(self, x):
        feature = self.features(x)
        feature = torch.flatten(feature, 1)
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
    def __init__(self,
                 args,
                 cnn_model,
                 input_d=300,
                 middle_d=1024,
                 output_d=2048):
        super(GCNCNN, self).__init__()
        self.gcn1_path = args.save_dir + 'GCN1_Corel-5K.pth'
        self.gcn2_path = args.save_dir + 'GCN2_Corel-5K.pth'

        # CNN model (features)
        self.cnn = cnn_model

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
            {'params': self.cnn.parameters(), 'lr': lr},
            {'params': self.gc1.parameters(), 'lr': lr},
            {'params': self.gc2.parameters(), 'lr': lr},
        ]
