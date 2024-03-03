from collections import OrderedDict
from logging import INFO
from typing import Dict, Tuple, List

import flwr as fl
import torch
from datasets import Dataset
from datasets.utils.logging import disable_progress_bar
from flwr.common import Metrics
from flwr.common.logger import log
from flwr.common.typing import Scalar
from flwr_datasets import FederatedDataset
from torch.utils.data import DataLoader

from net import resnet18
from utils import train, test, apply_transforms

NETWORK = resnet18(pretrained=False, in_channels=3, num_classes=53)
NUM_LAYERS = 27


# Flower client, adapted from Pytorch quickstart example
class FedClient(fl.client.NumPyClient):
    def __init__(self, trainset, valset, cid):
        self.trainset = trainset
        self.valset = valset
        self.cid = cid

        # Instantiate model
        self.model = NETWORK

        # Determine device
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)  # send model to device

    def get_parameters(self, config):
        return [val.cpu().numpy() for name, val in self.model.state_dict().items() if 'bn' not in name]

    def fit(self, parameters, config):
        set_params(self.model, parameters, self.cid)

        # Read from config
        batch, epochs, patience = config['batch_size'], config['epochs'], config['patience']

        # Construct dataloader
        trainloader = DataLoader(self.trainset, batch_size=batch, shuffle=True, drop_last=True)

        # Define optimizer
        optimizer = torch.optim.SGD(self.model.parameters(), lr=0.01, momentum=0.9)
        # Train
        train(self.model, trainloader, optimizer, epochs=epochs, patience=patience, device=self.device)

        # cifar batch 64
        valloader = DataLoader(self.valset, batch_size=64, drop_last=True)
        loss, accuracy = test(self.model, valloader, device=self.device)

        # Return local model and statistics
        return self.get_parameters({}), len(trainloader.dataset), {'loss': float(loss), 'accuracy': float(accuracy)}

    def evaluate(self, parameters, config):
        set_params(self.model, parameters, self.cid)

        # cifar batch 64
        valloader = DataLoader(self.valset, batch_size=64, drop_last=True)

        # Evaluate
        loss, accuracy = test(self.model, valloader, device=self.device)

        # Return statistics
        return float(loss), len(valloader.dataset), {'accuracy': float(accuracy)}


def get_client_fn(dataset: FederatedDataset):
    """Return a function to construct a client.

    The VirtualClientEngine will execute this function whenever a client is sampled by
    the strategy to participate.
    """

    def client_fn(cid: str) -> fl.client.Client:
        """Construct a FlowerClient with its own dataset partition."""

        # Let's get the partition corresponding to the i-th client
        client_dataset = dataset.load_partition(int(cid), 'train')

        client_dataset_splits = client_dataset.train_test_split(test_size=0.2)

        trainset = client_dataset_splits['train']
        valset = client_dataset_splits['test']

        # Now we apply the transform to each batch.
        trainset = trainset.with_transform(apply_transforms)
        valset = valset.with_transform(apply_transforms)

        # Create and return client
        return FedClient(trainset, valset, int(cid)).to_client()

    return client_fn


def fit_config(server_round: int) -> Dict[str, Scalar]:
    """Return a configuration with static batch size and (local) epochs."""
    config = {
        'epochs': 10,  # Number of local epochs done by clients
        'batch_size': 32,  # Batch size to use by clients during fit()
        'patience': 5,  # early stopping
    }
    return config


def set_params(model: torch.nn.ModuleList, params: List[fl.common.NDArrays], cid: int):
    """Set model weights from a list of NumPy ndarrays."""
    keys = [k for k in model.state_dict().keys() if 'bn' not in k]
    if len(params) == NUM_LAYERS:
        client_params = params
    else:
        cluster_label = params[-1][cid]
        client_params = params[cluster_label * NUM_LAYERS: (cluster_label + 1) * NUM_LAYERS]
    params_dict = zip(keys, client_params)
    state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
    model.load_state_dict(state_dict, strict=False)


def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    """Aggregation function for (federated) evaluation metrics, i.e. those returned by
    the client's evaluate() method."""
    # Multiply accuracy of each client by number of examples used
    accuracies = [num_examples * m['accuracy'] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]

    acc = sum(accuracies) / sum(examples)
    log(INFO, f"Decentralized acc: {acc}")
    # Aggregate and return custom metric (weighted average)
    return {'accuracy': acc}


def get_evaluate_fn(
        centralized_testset: Dataset,
):
    """Return an evaluation function for centralized evaluation."""

    def evaluate(
            server_round: int, parameters: fl.common.NDArrays, config: Dict[str, Scalar]
    ):
        """Use the entire CIFAR-10 test set for evaluation."""

        # Determine device
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        model = NETWORK
        set_params(model, parameters, 0)
        model.to(device)

        # Apply transform to dataset
        testset = centralized_testset.with_transform(apply_transforms)

        # Disable tqdm for dataset preprocessing
        disable_progress_bar()

        testloader = DataLoader(testset, batch_size=50)
        loss, accuracy = test(model, testloader, device=device)

        return loss, {'accuracy': accuracy}

    return evaluate
