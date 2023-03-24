import torch
import torch.nn as nn
import os
import torch.nn.functional as F
### https://github.com/dl4sits/BreizhCrops/tree/6de796ed36a457c8520322d6110b8f2862fd8c25/breizhcrops/models
### TempCNN

class TempCNN(torch.nn.Module):
    def __init__(self, input_dim=10, num_classes=8, sequencelength=73, kernel_size=5, hidden_dims=64, dropout=0.5):
        super(TempCNN, self).__init__()
        self.modelname = f"TempCNN_input-dim={input_dim}_num-classes={num_classes}_sequencelenght={sequencelength}_" \
                         f"kernelsize={kernel_size}_hidden-dims={hidden_dims}_dropout={dropout}"

        self.hidden_dims = hidden_dims

        self.conv_bn_relu1 = Conv1D_BatchNorm_Relu_Dropout(input_dim, hidden_dims, kernel_size=kernel_size,
                                                           drop_probability=dropout)
        self.conv_bn_relu2 = Conv1D_BatchNorm_Relu_Dropout(hidden_dims, hidden_dims, kernel_size=kernel_size,
                                                           drop_probability=dropout)
        self.conv_bn_relu3 = Conv1D_BatchNorm_Relu_Dropout(hidden_dims, hidden_dims, kernel_size=kernel_size,
                                                           drop_probability=dropout)
        self.flatten = Flatten()
        self.dense = FC_BatchNorm_Relu_Dropout(hidden_dims * sequencelength, 4 * hidden_dims, drop_probability=dropout)
        self.linear = nn.Linear(4 * hidden_dims, num_classes)
        #self.logsoftmax = nn.Sequential(nn.Linear(4 * hidden_dims, num_classes), nn.LogSoftmax(dim=-1))

    def forward(self, x):
        # require NxTxD
        #x = x.transpose(1,2)
        x = self.conv_bn_relu1(x)
        x = self.conv_bn_relu2(x)
        x = self.conv_bn_relu3(x)
        x = self.flatten(x)
        x = self.dense(x)
        return self.linear(x)

    def save(self, path="model.pth", **kwargs):
        print("\nsaving model to " + path)
        model_state = self.state_dict()
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(dict(model_state=model_state, **kwargs), path)

    def load(self, path):
        print("loading model from " + path)
        snapshot = torch.load(path, map_location="cpu")
        model_state = snapshot.pop('model_state', snapshot)
        self.load_state_dict(model_state)
        return snapshot


class Conv1D_BatchNorm_Relu_Dropout(torch.nn.Module):
    def __init__(self, input_dim, hidden_dims, kernel_size=5, drop_probability=0.5):
        super(Conv1D_BatchNorm_Relu_Dropout, self).__init__()

        self.block = nn.Sequential(
            nn.Conv1d(input_dim, hidden_dims, kernel_size, padding=(kernel_size // 2)),
            nn.BatchNorm1d(hidden_dims),
            nn.ReLU(),
            nn.Dropout(p=drop_probability)
        )

    def forward(self, X):
        return self.block(X)


class FC_BatchNorm_Relu_Dropout(torch.nn.Module):
    def __init__(self, input_dim, hidden_dims, drop_probability=0.5):
        super(FC_BatchNorm_Relu_Dropout, self).__init__()

        self.block = nn.Sequential(
            nn.Linear(input_dim, hidden_dims),
            nn.BatchNorm1d(hidden_dims),
            nn.ReLU(),
            nn.Dropout(p=drop_probability)
        )

    def forward(self, X):
        return self.block(X)


class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)
    
### InceptionTime

class InceptionTime(nn.Module):
    
    def __init__(self,num_classes, input_dim=10,num_layers=6, hidden_dims=128,use_bias=False, use_residual= True, device=torch.device("cpu")):
        super(InceptionTime, self).__init__()
        self.modelname = f"InceptionTime_input-dim={input_dim}_num-classes={num_classes}_" \
                         f"hidden-dims={hidden_dims}_num-layers={num_layers}"
        self.num_layers = num_layers
        self.use_residual = use_residual

        self.inception_modules_list = nn.ModuleList([InceptionModule(input_dim = input_dim,kernel_size=40, num_filters=hidden_dims//4,
                                                       use_bias=use_bias, device=device)])
        for i in range(num_layers-1):
          self.inception_modules_list.append(InceptionModule(input_dim = hidden_dims, kernel_size=40, num_filters=hidden_dims//4,
                                                       use_bias=use_bias, device=device))
        
        self.shortcut_layer_list = nn.ModuleList([ShortcutLayer(input_dim,hidden_dims,stride = 1, bias = False)])
        for i in range(num_layers//3):
          self.shortcut_layer_list.append(ShortcutLayer(hidden_dims,hidden_dims,stride = 1, bias = False))
        
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.outlinear = nn.Linear(hidden_dims,num_classes)

        self.to(device)

    def forward(self,x):
        # N x T x D -> N x D x T
        #x = x.transpose(1,2)
        input_res = x
        
        for d in range(self.num_layers):
            x = self.inception_modules_list[d](x)

            if self.use_residual and d % 3 == 2:
                x = self.shortcut_layer_list[d//3](input_res, x)
                input_res = x
        x = self.avgpool(x).squeeze(2)
        x = self.outlinear(x)
        #logprobabilities = F.log_softmax(x, dim=-1)
        return x
    
class InceptionModule(nn.Module):
    def __init__(self,input_dim=32, kernel_size=40, num_filters= 32, residual=False, use_bias=False, device=torch.device("cpu")):
        super(InceptionModule, self).__init__()

        self.residual = residual
        self.bottleneck = nn.Conv1d(input_dim, num_filters , kernel_size = 1, stride=1, padding= 0,bias=use_bias)
        
        # the for loop gives 40, 20, and 10 convolutions
        kernel_size_s = [kernel_size // (2 ** i) for i in range(3)]
        self.convolutions = [nn.Conv1d(num_filters, num_filters, kernel_size=kernel_size+1, stride=1, bias= False, padding=kernel_size//2).to(device) for kernel_size in kernel_size_s] #padding is 1 instead of kernel_size//2
        
        self.pool_conv = nn.Sequential(
            nn.MaxPool1d(kernel_size=3, stride=1, padding=1),
            nn.Conv1d(input_dim, num_filters,kernel_size=1, stride = 1,padding=0, bias=use_bias) 
        )

        self.bn_relu = nn.Sequential(
            nn.BatchNorm1d(num_filters*4),
            nn.ReLU()
        )

        self.to(device)

    def forward(self, input_tensor):
        # collapse feature dimension

        input_inception = self.bottleneck(input_tensor)
        features = [conv(input_inception) for conv in self.convolutions]
        features.append(self.pool_conv(input_tensor.contiguous()))
        features = torch.cat(features, dim=1) 
        features = self.bn_relu(features)

        return features

class ShortcutLayer(nn.Module):
    def __init__(self, in_planes, out_planes, stride, bias):
        super(ShortcutLayer, self).__init__()
        self.sc = nn.Sequential(nn.Conv1d(in_channels=in_planes,
                                          out_channels=out_planes,
                                          kernel_size=1,
                                          stride=stride,
                                          bias=bias),
                                nn.BatchNorm1d(num_features=out_planes))
        self.relu = nn.ReLU()

    def forward(self, input_tensor, out_tensor):
        x = out_tensor + self.sc(input_tensor)
        x = self.relu(x)

        return x