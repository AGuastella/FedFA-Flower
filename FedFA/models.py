"""CNN model architecture, training, and testing functions for MNIST."""

from collections import OrderedDict
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader


'''
class Net(nn.Module):
    ## !!!!!!!! num_classes è temporaneo !!!!!!!!!!!!!!!
    def __init__(self, num_classes: int) -> None:
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
'''
'''
class LogisticRegression(nn.Module):
    """A network for logistic regression using a single fully connected layer.

    As described in the Li et al., 2020 paper :

    [Federated Optimization in Heterogeneous Networks]
    (https://arxiv.org/pdf/1812.06127.pdf)
    """

    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.linear = nn.Linear(28 * 28, num_classes)

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input Tensor that will pass through the network

        Returns
        -------
        torch.Tensor
            The resulting Tensor after it has passed through the network
        """
        output_tensor = self.linear(torch.flatten(input_tensor, 1))
        return output_tensor

'''

def train(  # pylint: disable=too-many-arguments
    net: nn.Module,
    trainloader: DataLoader,
    device: torch.device,
    epochs: int,
    learning_rate: float,
    proximal_mu: float,
) -> None:
    """Train the network on the training set."""
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters())
    net.train()
    for epoch in range(epochs):
        correct, total, epoch_loss = 0, 0, 0.0
        for images, labels in trainloader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = net(images)
            loss = criterion(net(images), labels)
            loss.backward()
            optimizer.step()
            # Metrics
            epoch_loss += loss
            total += labels.size(0)
            correct += (torch.max(outputs.data, 1)[1] == labels).sum().item()
        epoch_loss /= len(trainloader.dataset)
        epoch_acc = correct / total
        print(f"Epoch {epoch+1}: train loss {epoch_loss}, accuracy {epoch_acc}")


def test(
    net: nn.Module, testloader: DataLoader, device: torch.device
) -> Tuple[float, float]:
    """Evaluate the network on the entire test set."""
    print('[Model] Starting Test')
    
    criterion = nn.CrossEntropyLoss()
    correct, total, loss = 0, 0, 0.0
    net.eval()
    print('[Model] Evaluation done')

    with torch.no_grad():
        print('[Model] testloader dim: ', len(testloader))
        for images, labels in testloader:
            print('[Model] Prediction process')
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)
            loss += criterion(outputs, labels).item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            break # !!!!!!!!!!!!!!!!!!!!!!! remove print('[])
    if len(testloader.dataset) == 0:
        raise ValueError("Testloader can't be 0, exiting...")
    loss /= len(testloader.dataset)
    accuracy = correct / total

    print('[Model] Ending Test')
    return loss, accuracy


def get_parameters(net) -> List[np.ndarray]:
    return [val.cpu().numpy() for _, val in net.state_dict().items()]


def set_parameters(net, parameters: List[np.ndarray]):
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
    net.load_state_dict(state_dict, strict=True)


######################################################################################

#BATCH_NORM_DECAY = 0.997
#BATCH_NORM_EPSILON = 1e-5
LAYER_NORM_EPSILON = 1e-5
GROUP_NORM_EPSILON = 1e-5

def get_norm_layer(norm, num_channels):
    if norm == "batch":
        return nn.BatchNorm2d(num_channels)
    elif norm == "layer":
        return nn.LayerNorm(num_channels, eps=LAYER_NORM_EPSILON)
    else:  # Assuming "group" normalization
        # Assuming GroupNorm from PyTorch 1.6+
        # MODIFICATO NUM_GROUP=3, prima era 2
        num_groups = 2  # You can adjust this number based on your requirements
        if num_channels % num_groups != 0:
            num_groups = num_channels  # Use num_channels groups if num_channels is not divisible
        return nn.GroupNorm(num_groups=num_groups, num_channels=num_channels, eps=GROUP_NORM_EPSILON)
        #return nn.GroupNorm(num_groups=2, num_channels=num_channels, eps=GROUP_NORM_EPSILON)


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, downsample=False, norm="group", l2_weight_decay=1e-3, stride=1, seed=None):
        super(ResBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.gn1 = get_norm_layer(norm, out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.gn2 = get_norm_layer(norm, out_channels)

        if downsample:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                get_norm_layer(norm, out_channels)
            )
        else:
            self.shortcut = nn.Sequential()

    def forward(self, x):
        shortcut = self.shortcut(x)
        out = self.conv1(x)
        out = self.gn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.gn2(out)
        out += shortcut
        out = self.relu(out)
        return out


class FFALayer(nn.Module):
    def __init__(self, prob=0.5, eps=1e-6, momentum1=0.99, momentum2=0.99, nfeat=None):
        super(FFALayer, self).__init__()
        self.prob = prob
        self.eps = eps
        self.momentum1 = momentum1
        self.momentum2 = momentum2
        self.nfeat = nfeat

        self.register_buffer('running_var_mean_bmic', torch.ones(self.nfeat))
        self.register_buffer('running_var_std_bmic', torch.ones(self.nfeat))
        self.register_buffer('running_mean_bmic', torch.zeros(self.nfeat))
        self.register_buffer('running_std_bmic', torch.ones(self.nfeat))

    def forward(self, x):
        if not self.training: return x
        if np.random.random() > self.prob: return x

        mean = x.mean(dim=[2, 3], keepdim=False)
        std = (x.var(dim=[2, 3], keepdim=False) + self.eps)
        std = std.sqrt()

        self.momentum_updating_running_mean_and_std(mean, std)

        var_mu = self.var(mean)
        var_std = self.var(std)

        running_var_mean_bmic = 1 / (1 + 1 / (self.running_var_mean_bmic + self.eps))
        gamma_mu = x.shape[1] * running_var_mean_bmic / sum(running_var_mean_bmic)

        running_var_std_bmic = 1 / (1 + 1 / (self.running_var_std_bmic + self.eps))
        gamma_std = x.shape[1] * running_var_std_bmic / sum(running_var_std_bmic)

        var_mu = (gamma_mu + 1) * var_mu
        var_std = (gamma_std + 1) * var_std

        var_mu = var_mu.sqrt().repeat(x.shape[0], 1)
        var_std = var_std.sqrt().repeat(x.shape[0], 1)

        beta = self.gaussian_sampling(mean, var_mu)
        gamma = self.gaussian_sampling(std, var_std)

        x = (x - mean.reshape(x.shape[0], x.shape[1], 1, 1)) / std.reshape(x.shape[0], x.shape[1], 1, 1)
        x = x * gamma.reshape(x.shape[0], x.shape[1], 1, 1) + beta.reshape(x.shape[0], x.shape[1], 1, 1)

        return x

    def gaussian_sampling(self, mu, std):
        e = torch.randn_like(std)
        z = e.mul(std).add_(mu)
        return z

    def sqrtvar(self, x):
        t = (x.var(dim=0, keepdim=True) + self.eps).sqrt()
        t = t.repeat(x.shape[0], 1)
        return t

    def var(self, x):
        t = x.var(dim=0, keepdim=False) + self.eps
        return t

    def momentum_updating_running_mean_and_std(self, mean, std):
        with torch.no_grad():
            self.running_mean_bmic = self.running_mean_bmic * self.momentum1 + \
                                     mean.mean(dim=0, keepdim=False) * (1 - self.momentum1)
            self.running_std_bmic = self.running_std_bmic * self.momentum1 + \
                                    std.mean(dim=0, keepdim=False) * (1 - self.momentum1)

    def momentum_updating_running_var(self, var_mean, var_std):
        with torch.no_grad():
            self.running_var_mean_bmic = self.running_var_mean_bmic * self.momentum2 + var_mean * (1 - self.momentum2)
            self.running_var_std_bmic = self.running_var_std_bmic * self.momentum2 + var_std * (1 - self.momentum2)


class ResNet18FA(nn.Module):
    def __init__(self, num_classes=10, norm=""):
        super(ResNet18FA, self).__init__()
        self.in_channels = 3  # Set input channels to 3
        self.layer0 = self.make_layer(3, 32, norm=norm)  # Modify the first layer to accept 3 channels
        self.layer1 = self.make_layer(32, 64, norm=norm)
        self.layer2 = self.make_layer(64, 128, stride=2, norm=norm)
        self.layer3 = self.make_layer(128, 256, stride=2, norm=norm)
        self.layer4 = self.make_layer(256, 512, stride=2, norm=norm)
        self.gap = nn.AdaptiveAvgPool2d(1)
        #self.fc = nn.Linear(512, num_classes)
        self.fc = nn.Linear(256, num_classes)
        self.ffa = [FFALayer(nfeat=nfeat) for nfeat in [32, 64, 128, 256, 512]]


    def make_layer(self, out_channels, blocks, stride=1, norm=""):
        layers = []
        layers.append(ResBlock(self.in_channels, out_channels, stride, norm=norm))
        self.in_channels = out_channels
        for _ in range(1, blocks):
            layers.append(ResBlock(self.in_channels, out_channels, norm=norm))
        return nn.Sequential(*layers)

    def set_running_var(self, var_mean, var_std):
        for ffa, mean, std in zip(self.ffa, var_mean, var_std):
            ffa.running_var_mean_bmic.data = mean
            ffa.running_var_std_bmic.data = std

    def get_running_var(self):
        mean = [ffa.running_var_mean_bmic.data for ffa in self.ffa]
        std = [ffa.running_var_std_bmic.data for ffa in self.ffa]
        return mean, std

    '''def forward(self, x):
        print("Input shape:", x.shape)  # Print input shape
        x = self.layer0(x)
        print("After layer0 shape:", x.shape)  # Print shape after layer0
        x = self.ffa[0](x)
        print("After FFA1 shape:", x.shape)  # Print shape after FFA1

        x = self.layer1(x)
        print("After layer1 shape:", x.shape)  # Print shape after layer1
        x = self.ffa[1](x)
        print("After FFA2 shape:", x.shape)  # Print shape after FFA2

        x = self.layer2(x)
        print("After layer2 shape:", x.shape)  # Print shape after layer2
        x = self.ffa[2](x)
        print("After FFA3 shape:", x.shape)  # Print shape after FFA3

        x = self.layer3(x)
        print("After layer3 shape:", x.shape)  # Print shape after layer3
        x = self.ffa[3](x)
        print("After FFA4 shape:", x.shape)  # Print shape after FFA4

        x = self.layer4(x)
        print("After layer4 shape:", x.shape)  # Print shape after layer4
        x = self.ffa[4](x)
        print("After FFA5 shape:", x.shape)  # Print shape after FFA5

        x = self.gap(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        print("Final output shape:", x.shape)  # Print final output shape
        return x
'''
    def forward(self, x):
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.gap(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
