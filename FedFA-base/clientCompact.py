from collections import OrderedDict
from typing import Callable, Dict, List, Tuple

import flwr as fl
import numpy as np
import torch
from flwr.common.typing import NDArrays, Scalar
from hydra.utils import instantiate
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from fedprox.models import test, train

class FlowerClient(fl.client.NumPyClient):
    def __init__(self, net, trainloader, valloader, device, num_epochs, learning_rate, straggler_schedule):
        self.net = net
        self.trainloader = trainloader
        self.valloader = valloader
        self.device = device
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.straggler_schedule = straggler_schedule

    def get_parameters(self, config) -> fl.common.typing.NDArrays:
        return [val.cpu().numpy() for _, val in self.net.state_dict().items()]

    def set_parameters(self, parameters: fl.common.typing.NDArrays) -> None:
        params_dict = zip(self.net.state_dict().keys(), parameters)
        state_dict = {k: torch.Tensor(v) for k, v in params_dict}
        self.net.load_state_dict(state_dict, strict=True)

    def fit(self, parameters: fl.common.typing.NDArrays, config) -> Tuple[fl.common.typing.NDArrays, int, Dict]:
        self.set_parameters(parameters)
        num_epochs = np.random.randint(1, self.num_epochs) if self.straggler_schedule[int(config["curr_round"]) - 1] and self.num_epochs > 1 else self.num_epochs
        if config["drop_client"]:
            return self.get_parameters({}), len(self.trainloader), {"is_straggler": True}
        train(self.net, self.trainloader, self.device, epochs=num_epochs, learning_rate=self.learning_rate, proximal_mu=float(config["proximal_mu"]))
        return self.get_parameters({}), len(self.trainloader), {"is_straggler": False}

    def evaluate(self, parameters: fl.common.typing.NDArrays, config) -> Tuple[float, int, Dict]:
        self.set_parameters(parameters)
        loss, accuracy = test(self.net, self.valloader, self.device)
        return float(loss), len(self.valloader), {"accuracy": float(accuracy)}

def gen_client_fn(num_clients: int, num_rounds: int, num_epochs: int, trainloaders: List[DataLoader], valloaders: List[DataLoader], learning_rate: float, stragglers: float, model: DictConfig) -> Callable[[str], FlowerClient]:
    stragglers_mat = np.transpose(np.random.choice([0, 1], size=(num_rounds, num_clients), p=[1 - stragglers, stragglers]))

    def client_fn(cid: str) -> FlowerClient:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        net = instantiate(model).to(device)
        trainloader, valloader = trainloaders[int(cid)], valloaders[int(cid)]
        return FlowerClient(net, trainloader, valloader, device, num_epochs, learning_rate, stragglers_mat[int(cid)])

    return client_fn
