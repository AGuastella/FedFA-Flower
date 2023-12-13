"""Defines the MNIST Flower Client and a function to instantiate it."""
from collections import OrderedDict
from typing import Callable, Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader
from hydra.utils import instantiate
from omegaconf import DictConfig

import flwr as fl
from flwr.common.typing import NDArrays, Scalar

from FedFA.modelsFedFAlight import ResNet18FA, test, train, get_parameters, set_parameters
#from FedFA.models import Net, test, train, get_parameters, set_parameters

class FlowerClient(fl.client.NumPyClient):
    def __init__(
        self,
        net: torch.nn.Module,
        trainloader: DataLoader,
        valloader: DataLoader,
        device: torch.device,
        num_epochs: int,
        learning_rate: float,
        straggler_schedule: np.ndarray,
    ):  # pylint: disable=too-many-arguments
        self.net = net
        self.trainloader = trainloader
        self.valloader = valloader
        self.device = device
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.straggler_schedule = straggler_schedule

        print('[Client] Created Client')


    def get_parameters(self, config: Dict[str, Scalar]) -> NDArrays:
        # Return the parameters of the current net.
        print('[Client] Get Parameter')
        return [val.cpu().numpy() for _, val in self.net.state_dict().items()]

    def set_parameters(self, parameters: NDArrays) -> None:
        """Change the parameters of the model using the given ones."""
        params_dict = zip(self.net.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
        self.net.load_state_dict(state_dict, strict=True)

    def fit(
        self, parameters: NDArrays, config: Dict[str, Scalar]
    ) -> Tuple[NDArrays, int, Dict]:
        """Implement distributed fit function for a given client."""
        self.set_parameters(parameters)

        # At each round check if the client is a straggler,
        # if so, train less epochs (to simulate partial work)
        # if the client is told to be dropped (e.g. because not using
        # FedProx in the server), the fit method returns without doing
        # training.
        # This method always returns via the metrics (last argument being
        # returned) whether the client is a straggler or not. This info
        # is used by strategies other than FedProx to discard the update.
        '''if (
            self.straggler_schedule[int(config["curr_round"]) - 1]
            and self.num_epochs > 1
        ):
            num_epochs = np.random.randint(1, self.num_epochs)

            if config["drop_client"]:
                # return without doing any training.
                # The flag in the metric will be used to tell the strategy
                # to discard the model upon aggregation
                return (
                    self.get_parameters({}),
                    len(self.trainloader),
                    {"is_straggler": True},
                )

        else:'''
        
        num_epochs = self.num_epochs

        #print('trainloader: ', self.trainloader)
        print('[Client] Start training')
        train(
            self.net,
            self.trainloader,
            self.device,
            epochs=num_epochs,
            learning_rate=self.learning_rate,
            # proximal_mu=float(config["proximal_mu"]),
        )

        return self.get_parameters({}), len(self.trainloader), {"is_straggler": False}

    def evaluate(
        self, parameters: NDArrays, config: Dict[str, Scalar]
    ) -> Tuple[float, int, Dict]:
        """Implement distributed evaluation for a given client."""
        self.set_parameters(parameters)
        loss, accuracy = test(self.net, self.valloader, self.device)
        return float(loss), len(self.valloader), {"accuracy": float(accuracy)}

def gen_client_fn(
    num_clients: int,
    num_rounds: int,
    num_epochs: int,
    trainloaders: List[DataLoader],
    valloaders: List[DataLoader],
    learning_rate: float,
    stragglers: float,
    # model: DictConfig,
) -> Callable[[str], FlowerClient]:  # pylint: disable=too-many-arguments
    # Defines a straggling schedule for each clients, i.e at which round will they
    # be a straggler. This is done so at each round the proportion of straggling
    # clients is respected
    stragglers_mat = np.transpose(
        np.random.choice(
            [0, 1], size=(num_rounds, num_clients), p=[1 - stragglers, stragglers]
        )
    )

    def client_fn(cid: str) -> FlowerClient:
        """Create a Flower client representing a single organization."""
        # Load model
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # net = instantiate(model).to(device)
        net = ResNet18FA().to(device) # !?!?!?!?!?!?!?!?!?!?!?!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        #net = Net().to(device) # !?!?!?!?!?!?!?!?!?!?!?!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

        # Note: each client gets a different trainloader/valloader, so each client
        # will train and evaluate on their own unique data
        trainloader = trainloaders[int(cid)]
        valloader = valloaders[int(cid)]

        return FlowerClient(
            net,
            trainloader,
            valloader,
            device,
            num_epochs,
            learning_rate,
            stragglers_mat[int(cid)],
        )

    return client_fn
