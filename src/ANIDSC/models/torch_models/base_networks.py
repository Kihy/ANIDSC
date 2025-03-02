
import torch
import numpy as np
import importlib


def _instantiate_class(module_name: str, class_name: str):
    module = importlib.import_module(module_name)
    class_ = getattr(module, class_name)
    return class_()

class ConvNet(torch.nn.Module):
    """Convolutional Network

    Args:
        n_features (int):
            number of input data features
        kernel_size (int):
            kernel size (Default=1)
        n_hidden (int):
            number of hidden units in hidden layers (Default=8)
        n_layers (int):
            number of layers (Default=5)
        activation (str):
            name of activation layer,
            activation should be implemented in torch.nn.module.activation
            (Default='ReLU')
        bias (bool):
            use bias or not
            (Default=False)

    """
    def __init__(self, n_features, kernel_size=1, n_hidden=8, n_layers=5,
                 activation='ReLU', bias=False):
        super(ConvNet, self).__init__()

        self.layers = []

        in_channels = n_features
        for i in range(n_layers+1):
            self.layers += [
                torch.nn.Conv1d(in_channels, n_hidden,
                                kernel_size=kernel_size,
                                bias=bias)
            ]
            if i != n_layers:
                self.layers += [
                    # torch.nn.LeakyReLU(inplace=True)
                    _instantiate_class(module_name="torch.nn.modules.activation",
                                       class_name=activation)
                ]
            in_channels = n_hidden

        self.net = torch.nn.Sequential(*self.layers)

        return

    def forward(self, x):
        return self.net(x)


class MlpAE(torch.nn.Module):
    """MLP-based AutoEncoder"""
    def __init__(self, n_features, n_hidden='500,100', n_emb=20, activation='ReLU',
                 bias=False, batch_norm=False,
                 skip_connection=None, dropout=None
                 ):
        super(MlpAE, self).__init__()

        if type(n_hidden)==int:
            n_hidden = [n_hidden]
        if type(n_hidden)==str:
            n_hidden = n_hidden.split(',')
            n_hidden = [int(a) for a in n_hidden]
        num_layers = len(n_hidden)

        self.encoder_layers = []
        for i in range(num_layers+1):
            in_channels = n_features if i == 0 else n_hidden[i-1]
            out_channels = n_emb if i == num_layers else n_hidden[i]
            self.encoder_layers += [LinearBlock(in_channels, out_channels,
                                                bias=bias, batch_norm=batch_norm,
                                                activation=activation if i != num_layers else None,
                                                skip_connection=skip_connection if i != num_layers else 0,
                                                dropout=dropout if i != num_layers else None)]

        self.decoder_layers = []
        for i in range(num_layers+1):
            in_channels = n_emb if i == 0 else n_hidden[num_layers-i]
            out_channels = n_features if i == num_layers else n_hidden[num_layers-1-i]
            self.decoder_layers += [LinearBlock(in_channels, out_channels,
                                                bias=bias, batch_norm=batch_norm,
                                                activation=activation if i != num_layers else None,
                                                skip_connection=skip_connection if i != num_layers else 0,
                                                dropout=dropout if i != num_layers else None)]

        self.encoder = torch.nn.Sequential(*self.encoder_layers)
        self.decoder = torch.nn.Sequential(*self.decoder_layers)

    def forward(self, x):
        enc = self.encoder(x)
        xx = self.decoder(enc)
        return xx, enc


class MLPnet(torch.nn.Module):
    """MLP-based Representation Network"""
    def __init__(self, n_features, n_hidden='500,100', n_output=20, mid_channels=None,
                 activation='ReLU', bias=False, batch_norm=False,
                 skip_connection=None, dropout=None):
        super(MLPnet, self).__init__()
        self.skip_connection = skip_connection
        self.n_output = n_output

        if type(n_hidden)==int:
            n_hidden = [n_hidden]
        if type(n_hidden)==str:
            n_hidden = n_hidden.split(',')
            n_hidden = [int(a) for a in n_hidden]
        num_layers = len(n_hidden)

        # for only use one kind of activation layer
        if type(activation) == str:
            activation = [activation] * num_layers
            activation.append(None)

        assert len(activation) == len(n_hidden)+1, 'activation and n_hidden are not matched'

        self.layers = []
        for i in range(num_layers+1):
            in_channels, out_channels = self.get_in_out_channels(i, num_layers, n_features,
                                                                 n_hidden, n_output, skip_connection)
            self.layers += [
                LinearBlock(in_channels, out_channels,
                            mid_channels=mid_channels,
                            bias=bias, batch_norm=batch_norm,
                            activation=activation[i],
                            skip_connection=skip_connection if i != num_layers else 0,
                            dropout=dropout if i !=num_layers else None)
            ]
        self.network = torch.nn.Sequential(*self.layers)

    def forward(self, x):
        x = self.network(x)
        return x

    def get_in_out_channels(self, i, num_layers, n_features, n_hidden, n_output, skip_connection):
        if skip_connection is None:
            in_channels = n_features if i == 0 else n_hidden[i-1]
            out_channels = n_output if i == num_layers else n_hidden[i]
        elif skip_connection == 'concat':
            in_channels = n_features if i == 0 else np.sum(n_hidden[:i])+n_features
            out_channels = n_output if i == num_layers else n_hidden[i]
        else:
            raise NotImplementedError('')
        return in_channels, out_channels


class LinearBlock(torch.nn.Module):
    """Linear Block"""
    def __init__(self, in_channels, out_channels, mid_channels=None,
                 activation='Tanh', bias=False, batch_norm=False,
                 skip_connection=None, dropout=None):
        super(LinearBlock, self).__init__()

        self.skip_connection = skip_connection

        self.linear = torch.nn.Linear(in_channels, out_channels, bias=bias)

        # Tanh, ReLU, LeakyReLU, Sigmoid
        if activation is not None:
            self.act_layer = _instantiate_class("torch.nn.modules.activation", activation)
        else:
            self.act_layer = torch.nn.Identity()

        self.dropout = dropout
        if dropout is not None:
            self.dropout_layer = torch.nn.Dropout(p=dropout)

        self.batch_norm = batch_norm
        if batch_norm is True:
            dim = out_channels if mid_channels is None else mid_channels
            self.bn_layer = torch.nn.BatchNorm1d(dim, affine=bias)

    def forward(self, x):
        x1 = self.linear(x)
        x1 = self.act_layer(x1)

        if self.batch_norm is True:
            x1 = self.bn_layer(x1)

        if self.dropout is not None:
            x1 = self.dropout_layer(x1)

        if self.skip_connection == 'concat':
            x1 = torch.cat([x, x1], axis=1)

        return x1