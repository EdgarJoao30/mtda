import torch
import torch.nn as nn

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


# class GradientReversal(torch.autograd.Function):
#     @staticmethod
#     def forward(ctx, x, lamb_da):
#         ctx.save_for_backward(x, lamb_da)
#         return x
    
#     @staticmethod
#     def backward(ctx, grad_output):
#         grad_input = None
#         _, lamb_da = ctx.saved_tensors
#         if ctx.needs_input_grad[0]:
#             grad_input = - lamb_da*grad_output
#         return grad_input, None

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
