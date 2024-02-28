import argparse
import multiprocessing

import torch
import flwr as fl
from datasets import disable_progress_bar
from flwr_datasets import FederatedDataset

from client import fit_config, weighted_average, get_evaluate_fn, get_client_fn
from dirichlet import DirichletPartitioner
from strategy import FedAP


def main():
    # Parse input arguments
    parser = argparse.ArgumentParser(description='Flower Simulation with PyTorch')

    parser.add_argument(
        '--num_cpus',
        type=int,
        default=2,
        help='Number of CPUs to assign to a virtual client',
    )
    parser.add_argument(
        '--num_gpus',
        type=float,
        default=0.0,
        help='Ratio of GPU memory to assign to a virtual client',
    )
    parser.add_argument(
        '--num_clients',
        type=int,
        default=10,
        help='Number of clients'
    )
    parser.add_argument('--num_rounds', type=int, default=50, help='Number of FL rounds.')
    parser.add_argument('--affinity', type=str, default='euclidean', help='Distance function in affinity propagation.')

    args = parser.parse_args()

    num_cpus = args.num_cpus
    num_gpus = args.num_gpus
    num_clients = args.num_clients
    num_rounds = args.num_rounds
    affinity = args.affinity

    # Download MNIST dataset and partition it
    mnist_fds = FederatedDataset(
        dataset='./cifar10',
        # partitioners={'train': NUM_CLIENTS},
        partitioners={'train': DirichletPartitioner(num_clients)},
    )
    centralized_testset = mnist_fds.load_full('test')

    # Configure the strategy
    # strategy = fl.server.strategy.FedAvg(
    strategy = FedAP(
        # fraction_fit=0.1,  # Sample 10% of available clients for training
        # fraction_evaluate=0.05,  # Sample 5% of available clients for evaluation
        min_fit_clients=num_clients,  # Never sample less than min_fit_clients clients for training
        min_evaluate_clients=num_clients,  # Never sample less than min_evaluate_clients clients for evaluation
        min_available_clients=int(
            num_clients * 1
        ),  # Wait until at least min_available_clients clients are available
        on_fit_config_fn=fit_config,
        evaluate_metrics_aggregation_fn=weighted_average,  # Aggregate federated metrics
        evaluate_fn=get_evaluate_fn(centralized_testset),  # Global evaluation function
        affinity=affinity
    )

    # Resources to be assigned to each virtual client
    client_resources = {
        'num_cpus': max(multiprocessing.cpu_count() // num_clients, 4),
        'num_gpus': num_gpus if not torch.cuda.is_available() else max(1. / num_clients, .1)
    }

    # Start Logger
    fl.common.logger.configure(identifier="Experiment", filename="log.txt")

    # Start simulation
    fl.simulation.start_simulation(
        client_fn=get_client_fn(mnist_fds),
        num_clients=num_clients,
        client_resources=client_resources,
        config=fl.server.ServerConfig(num_rounds=num_rounds),
        strategy=strategy,
        actor_kwargs={
            'on_actor_init_fn': disable_progress_bar  # disable tqdm on each actor/process spawning virtual clients
        },
    )


if __name__ == '__main__':
    main()
