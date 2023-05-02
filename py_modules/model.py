import torch
import torch.nn as nn
import os
import torch.nn.functional as F
import pdb

### https://github.com/dl4sits/BreizhCrops/tree/6de796ed36a457c8520322d6110b8f2862fd8c25/breizhcrops/models
### TempCNN

class TempCNN(torch.nn.Module):
    def __init__(self, num_classes=8, kernel_size=5, hidden_dims=64, dropout=0.5):
        super(TempCNN, self).__init__()
        self.hidden_dims = hidden_dims

        self.conv_bn_relu1 = Conv1D_BatchNorm_Relu_Dropout(hidden_dims, kernel_size=kernel_size,
                                                           drop_probability=dropout)
        self.conv_bn_relu2 = Conv1D_BatchNorm_Relu_Dropout(hidden_dims, kernel_size=kernel_size,
                                                           drop_probability=dropout)
        self.conv_bn_relu3 = Conv1D_BatchNorm_Relu_Dropout(hidden_dims, kernel_size=kernel_size,
                                                           drop_probability=dropout)
        self.flatten = Flatten()
        self.dense = FC_BatchNorm_Relu_Dropout(4 * hidden_dims, drop_probability=dropout)
        self.linear = nn.LazyLinear(num_classes)

    def forward(self, x):
        # require NxTxD
        x = self.conv_bn_relu1(x)
        x = self.conv_bn_relu2(x)
        x = self.conv_bn_relu3(x)
        x = self.flatten(x)
        x = self.dense(x)
        return self.linear(x)

class s1TempCNN(torch.nn.Module):
    def __init__(self, num_classes=8, kernel_size2d=3, hidden_dims=64, dropout=0.5):
        super(s1TempCNN, self).__init__()
        self.hidden_dims = hidden_dims

        
        self.conv2d_bn_relu1 = Conv2D_BatchNorm_Relu_Dropout(hidden_dims*4, kernel_size=kernel_size2d,
                                                           drop_probability=dropout)
        self.conv2d_bn_relu2 = Conv2D_BatchNorm_Relu_Dropout(hidden_dims*4, kernel_size=kernel_size2d,
                                                           drop_probability=dropout)
        self.conv2d_bn_relu3 = Conv2D_BatchNorm_Relu_Dropout(hidden_dims*4, kernel_size=kernel_size2d,
                                                           drop_probability=dropout)
        
        self.gap2D =  nn.AdaptiveAvgPool2d(1)
        
        self.flatten = Flatten()
        self.dense = FC_BatchNorm_Relu_Dropout(4 * hidden_dims, drop_probability=dropout)
        self.linear = nn.LazyLinear(num_classes)

    def forward(self, s1):
        s1 = self.conv2d_bn_relu1(s1)
        s1 = self.conv2d_bn_relu2(s1)
        s1 = self.conv2d_bn_relu3(s1)
        s1 = self.gap2D(s1)
        
        #pdb.set_trace()
        x = self.dense(torch.squeeze(s1))
        return self.linear(x)

class mmTempCNN(torch.nn.Module):
    def __init__(self, num_classes=8, kernel_size1d=5, kernel_size2d=3, hidden_dims=64, dropout=0.5):
        super(mmTempCNN, self).__init__()
        self.hidden_dims = hidden_dims

        self.conv1d_bn_relu1 = Conv1D_BatchNorm_Relu_Dropout(hidden_dims, kernel_size=kernel_size1d,
                                                           drop_probability=dropout)
        self.conv1d_bn_relu2 = Conv1D_BatchNorm_Relu_Dropout(hidden_dims, kernel_size=kernel_size1d,
                                                           drop_probability=dropout)
        self.conv1d_bn_relu3 = Conv1D_BatchNorm_Relu_Dropout(hidden_dims, kernel_size=kernel_size1d,
                                                           drop_probability=dropout)
        
        self.conv2d_bn_relu1 = Conv2D_BatchNorm_Relu_Dropout(hidden_dims*4, kernel_size=kernel_size2d,
                                                           drop_probability=dropout)
        self.conv2d_bn_relu2 = Conv2D_BatchNorm_Relu_Dropout(hidden_dims*4, kernel_size=kernel_size2d,
                                                           drop_probability=dropout)
        self.conv2d_bn_relu3 = Conv2D_BatchNorm_Relu_Dropout(hidden_dims*4, kernel_size=kernel_size2d,
                                                           drop_probability=dropout)
        
        self.gap2D =  nn.AdaptiveAvgPool2d(1)
        
        self.flatten = Flatten()
        self.dense = FC_BatchNorm_Relu_Dropout(4 * hidden_dims, drop_probability=dropout)
        self.linear_fused = nn.LazyLinear(num_classes)
        self.linear_s2 = nn.LazyLinear(num_classes)
        self.linear_s1= nn.LazyLinear(num_classes)

    def forward(self, s2, s1):
        # require NxTxD
        #x = x.transpose(1,2)
        s2 = self.conv1d_bn_relu1(s2)
        s2 = self.conv1d_bn_relu2(s2)
        s2 = self.conv1d_bn_relu3(s2)
        s2 = self.flatten(s2)
        
        s1 = self.conv2d_bn_relu1(s1)
        s1 = self.conv2d_bn_relu2(s1)
        s1 = self.conv2d_bn_relu3(s1)
        s1 = self.gap2D(s1)
        
        feats = torch.cat((s2, torch.squeeze(s1) ), dim=1)
        
        #pdb.set_trace()
        x = self.dense(feats)
        return self.linear_fused(x), self.linear_s2(s2), self.linear_s1(torch.squeeze(s1))

class Conv1D_BatchNorm_Relu_Dropout(torch.nn.Module):
    def __init__(self, hidden_dims, kernel_size=5, drop_probability=0.5):
        super(Conv1D_BatchNorm_Relu_Dropout, self).__init__()

        self.block = nn.Sequential(
            nn.LazyConv1d(hidden_dims, kernel_size, padding=(kernel_size // 2)),
            nn.BatchNorm1d(hidden_dims),
            nn.ReLU(),
            nn.Dropout(p=drop_probability)
        )

    def forward(self, X):
        return self.block(X)
    
class Conv2D_BatchNorm_Relu_Dropout(torch.nn.Module):
    def __init__(self, hidden_dims, kernel_size=3, drop_probability=0.5):
        super(Conv2D_BatchNorm_Relu_Dropout, self).__init__()

        self.block = nn.Sequential(
            nn.LazyConv2d(hidden_dims, kernel_size, padding=(kernel_size // 2)),
            nn.BatchNorm2d(hidden_dims),
            nn.ReLU(),
            nn.Dropout(p=drop_probability)
        )

    def forward(self, X):
        return self.block(X)


class FC_BatchNorm_Relu_Dropout(torch.nn.Module):
    def __init__(self, hidden_dims, drop_probability=0.5):
        super(FC_BatchNorm_Relu_Dropout, self).__init__()

        self.block = nn.Sequential(
            nn.LazyLinear(hidden_dims),
            nn.BatchNorm1d(hidden_dims),
            nn.ReLU(),
            nn.Dropout(p=drop_probability)
        )

    def forward(self, X):
        return self.block(X)


class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)
    
    
### Inception Time version 2

class InceptionLayer(nn.Module):
    # PyTorch translation of the Keras code in https://github.com/hfawaz/dl-4-tsc
    def __init__(self, nb_filters=32, use_bottleneck=True,
                 bottleneck_size=32, kernel_size=40):
        super(InceptionLayer, self).__init__()

        # self.in_channels = in_channels
        kernel_size_s = [5, 9, 13] # = [40, 20, 10]
        kernel_size_s = [x+1 for x in kernel_size_s] # Avoids warning about even kernel_size with padding="same"
        self.bottleneck_size = bottleneck_size
        self.use_bottleneck = use_bottleneck


        # Bottleneck layer
        self.bottleneck = nn.LazyConv1d(self.bottleneck_size, kernel_size=1,
                                    stride=1, padding="same", bias=False)
        self.max_pool = nn.MaxPool1d(kernel_size=3, stride=1, padding=1)
        self.bottleneck_conv = nn.LazyConv1d(nb_filters, kernel_size=1,
                                         stride=1, padding="same", bias=False)

        # Convolutional layer (several filter lenghts)
        self.conv_list = nn.ModuleList([])
        for i in range(len(kernel_size_s)):
            # Input size could be self.in_channels or self.bottleneck_size (if bottleneck was applied)
            self.conv_list.append(nn.LazyConv1d(nb_filters, kernel_size=kernel_size_s[i],
                                            stride=1, padding='same', bias=False))

        self.bn = nn.BatchNorm1d(4*self.bottleneck_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.2)

    def forward(self, input):
        in_channels = input.shape[-2]
        if self.use_bottleneck and int(in_channels) > self.bottleneck_size:
            input_inception = self.bottleneck(input)
        else:
            input_inception = input

        max_pool = self.max_pool(input)
        output = self.bottleneck_conv(max_pool)
        for conv in self.conv_list:
            output = torch.cat((output,conv(input_inception)),dim=1)

        output = self.bn(output)
        output = self.relu(output)
        output = self.dropout(output)

        return output


class Inception(nn.Module):
    # PyTorch translation of the Keras code in https://github.com/hfawaz/dl-4-tsc
    def __init__(self, nb_classes, nb_filters=16, use_residual=True,
                 use_bottleneck=True, bottleneck_size=16, depth=6, kernel_size=40):
        super(Inception, self).__init__()

        self.use_residual = use_residual

        # Inception layers
        self.inception_list = nn.ModuleList(
            [InceptionLayer(nb_filters,use_bottleneck, bottleneck_size, kernel_size) for _ in range(depth)])
        # Explicit input sizes (i.e. without using Lazy layers). Requires n_var passed as a constructor input
        # self.inception_list = nn.ModuleList([InceptionLayer(n_var, nb_filters,use_bottleneck, bottleneck_size, kernel_size) for _ in range(depth)])
        # for _ in range(1,depth):
        #     inception = InceptionLayer(4*nb_filters,nb_filters,use_bottleneck, bottleneck_size, kernel_size)
        #     self.inception_list.append(inception)

        # Fully-connected layer
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.LazyLinear(nb_classes),
            # nn.Softmax(dim=1) # already performed inside CrossEntropyLoss
        )

        # Shortcut layers
        # First residual layer has n_var channels as inputs while the remaining have 4*nb_filters
        self.conv = nn.ModuleList([
            nn.LazyConv1d(4*nb_filters, kernel_size=1,
                            stride=1, padding="same", bias=False)
            for _ in range(int(depth/3))
        ])
        self.bn = nn.ModuleList([nn.BatchNorm1d(4*nb_filters) for _ in range(int(depth/3))])
        self.relu = nn.ModuleList([nn.ReLU() for _ in range(int(depth/3))])
        

    def _shortcut_layer(self, input_tensor, out_tensor, id):
        shortcut_y = self.conv[id](input_tensor)
        shortcut_y = self.bn[id](shortcut_y)
        x = torch.add(shortcut_y, out_tensor)
        x = self.relu[id](x)
        return x

    def forward(self, x):
        input_res = x

        for d, inception in enumerate(self.inception_list):
            x = inception(x)

            # Residual layer
            if self.use_residual and d % 3 == 2:
                x = self._shortcut_layer(input_res,x, int(d/3))
                input_res = x

        gap_layer = self.gap(x)
        return self.fc(gap_layer)

### SpADANN


class Conv1dBloc(nn.Module):
    """Elementary 1D-convolution block for TempCNN encoder"""
    def __init__(self, filters_nb, kernel_size, drop_val):
        super(Conv1dBloc, self).__init__()
        self.conv1D = nn.LazyConv1d(filters_nb, kernel_size, padding="same")
        self.batch_norm = nn.BatchNorm1d(filters_nb)
        self.act = nn.ReLU()
        self.output_ = nn.Dropout(drop_val)

    def forward(self, inputs):
        conv1D = self.conv1D(inputs)
        batch_norm = self.batch_norm(conv1D)
        act = self.act(batch_norm)
        return self.output_(act)


class TempCnnEncoder(nn.Module):
    """Encoder of SITS on temporal dimension"""
    def __init__(self, drop_val=0.5):
        super(TempCnnEncoder, self).__init__()
        self.conv_bloc1 = Conv1dBloc(64, 5, drop_val)
        self.conv_bloc2 = Conv1dBloc(64, 5, drop_val)
        self.conv_bloc3 = Conv1dBloc(64, 5, drop_val)
        self.flatten = nn.Flatten()

    def forward(self, inputs):
        conv1 = self.conv_bloc1(inputs)
        conv2 = self.conv_bloc2(conv1)
        conv3 = self.conv_bloc3(conv2)
        flatten = self.flatten(conv3)
        return flatten


class Classifier(nn.Module):
    """Generic classifier"""
    def __init__(self, nb_class, nb_units, drop_val=0.5):
        super(Classifier, self).__init__()
        self.dense = nn.LazyLinear(nb_units)
        self.batch_norm = nn.BatchNorm1d(nb_units)
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(drop_val)
        self.output_ = nn.LazyLinear(nb_class)
        
    def forward(self, inputs):
        dense = self.dense(inputs)
        batch_norm = self.batch_norm(dense)
        act = self.act(batch_norm)
        dropout = self.dropout(act)
        return self.output_(dropout)

class GradReverse(torch.autograd.Function):
    """Gradient reversal layer (GRL)"""
    @staticmethod
    def forward(ctx, x, lamb_da):
        ctx.lamb_da = lamb_da
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.lamb_da, None


class SpADANN(nn.Module):
    """SpADANN model is composed of
    a feature extractor: encoder
    a label predictor/classifier: labelClassif
    a domain predictor/classifier: domainClassif
    a GRL to connect feature extractor and domain predictor/classifier
    """
    def __init__(self, nb_class, drop_val=0.5):
        super(SpADANN, self).__init__()
        self.encoder = TempCnnEncoder(drop_val)
        self.labelClassif = Classifier(nb_class, 256, drop_val)
        self.grl = GradReverse()
        self.domainClassif = Classifier(2, 256, drop_val)

    def forward(self, inputs, lamb_da=1.0):
        enc_out = self.encoder(inputs)
        grl = self.grl.apply(enc_out, lamb_da)
        return enc_out, self.labelClassif(enc_out), self.domainClassif(grl)
