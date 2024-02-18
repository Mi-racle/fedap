from collections import OrderedDict
from typing import Dict, Tuple, List

import flwr as fl
import torch
from datasets import Dataset
from datasets.utils.logging import disable_progress_bar
from flwr.common import Metrics
from flwr.common.typing import Scalar
from flwr_datasets import FederatedDataset
from torch.utils.data import DataLoader

from net import Net, VGG16
from utils import train, test, apply_transforms

NUM_CLIENTS = 100
NETWORK = VGG16()


# Flower client, adapted from Pytorch quickstart example
class FedClient(fl.client.NumPyClient):
    def __init__(self, trainset, valset):
        self.trainset = trainset
        self.valset = valset

        # Instantiate model
        self.model = NETWORK

        # Determine device
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)  # send model to device

    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def fit(self, parameters, config):
        set_params(self.model, parameters)

        # Read from config
        batch, epochs = config["batch_size"], config["epochs"]

        # Construct dataloader
        trainloader = DataLoader(self.trainset, batch_size=batch, shuffle=True)

        # Define optimizer
        optimizer = torch.optim.SGD(self.model.parameters(), lr=0.01, momentum=0.9)
        # Train
        train(self.model, trainloader, optimizer, epochs=epochs, device=self.device)

        # Return local model and statistics
        return self.get_parameters({}), len(trainloader.dataset), {}

    def evaluate(self, parameters, config):
        set_params(self.model, parameters)

        # Construct dataloader
        valloader = DataLoader(self.valset, batch_size=64)

        # Evaluate
        loss, accuracy = test(self.model, valloader, device=self.device)

        # Return statistics
        return float(loss), len(valloader.dataset), {"accuracy": float(accuracy)}


def get_client_fn(dataset: FederatedDataset):
    """Return a function to construct a client.

    The VirtualClientEngine will execute this function whenever a client is sampled by
    the strategy to participate.
    """

    def client_fn(cid: str) -> fl.client.Client:
        """Construct a FlowerClient with its own dataset partition."""

        # Let's get the partition corresponding to the i-th client
        client_dataset = dataset.load_partition(int(cid), "train")

        # Now let's split it into train (90%) and validation (10%)
        client_dataset_splits = client_dataset.train_test_split(test_size=0.1)

        trainset = client_dataset_splits["train"]
        valset = client_dataset_splits["test"]

        # Now we apply the transform to each batch.
        trainset = trainset.with_transform(apply_transforms)
        valset = valset.with_transform(apply_transforms)

        # Create and return client
        return FedClient(trainset, valset).to_client()

    return client_fn


def fit_config(server_round: int) -> Dict[str, Scalar]:
    """Return a configuration with static batch size and (local) epochs."""
    config = {
        "epochs": 1,  # Number of local epochs done by clients
        "batch_size": 32,  # Batch size to use by clients during fit()
    }
    return config


def set_params(model: torch.nn.ModuleList, params: List[fl.common.NDArrays]):
    """Set model weights from a list of NumPy ndarrays."""
    params_dict = zip(model.state_dict().keys(), params)
    state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
    model.load_state_dict(state_dict, strict=True)


def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    """Aggregation function for (federated) evaluation metrics, i.e. those returned by
    the client's evaluate() method."""
    # Multiply accuracy of each client by number of examples used
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]

    # Aggregate and return custom metric (weighted average)
    return {"accuracy": sum(accuracies) / sum(examples)}


def get_evaluate_fn(
        centralized_testset: Dataset,
):
    """Return an evaluation function for centralized evaluation."""

    def evaluate(
            server_round: int, parameters: fl.common.NDArrays, config: Dict[str, Scalar]
    ):
        """Use the entire CIFAR-10 test set for evaluation."""

        # Determine device
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        model = NETWORK
        set_params(model, parameters)
        model.to(device)

        # Apply transform to dataset
        testset = centralized_testset.with_transform(apply_transforms)

        # Disable tqdm for dataset preprocessing
        disable_progress_bar()

        testloader = DataLoader(testset, batch_size=50)
        loss, accuracy = test(model, testloader, device=device)

        return loss, {"accuracy": accuracy}

    return evaluate
