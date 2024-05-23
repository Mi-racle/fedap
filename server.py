import argparse
import multiprocessing
import os
from logging import INFO

import torch
import flwr as fl
from datasets import disable_progress_bar
from flwr.common import log
from flwr_datasets import FederatedDataset

from clients.fedclient import fit_config, weighted_average, get_evaluate_fn, get_client_fn

from partitioner import DirichletPartitioner, LabelPartitioner, MixPartitioner
from strategies.myfedavgm import MyFedAvgM
from strategies.myfedprox import MyFedProx
from strategies.myfedavg import MyFedAvg
from strategies.fedacc import FedAcc
from strategies.fedap import FedAP
from utils import increment_path


def main():
    # Parse input arguments
    parser = argparse.ArgumentParser(description='Flower Simulation with PyTorch')

    parser.add_argument('--num_cpus', type=int, default=2, help='Number of CPUs')
    parser.add_argument('--num_gpus', type=float, default=0.0, help='Ratio of GPU memory')
    parser.add_argument('--num_clients', type=int, default=4, help='Number of clients')
    parser.add_argument('--num_rounds', type=int, default=30, help='Number of FL rounds')
    parser.add_argument('--affinity', type=str, default='fedavg', help='Distance function in affinity propagation')
    parser.add_argument('--dataset', type=str, default='cifar10', help='Dataset')
    parser.add_argument('--partitioner', type=str, default='iid', help='Distribution')

    args = parser.parse_args()

    num_cpus = args.num_cpus
    num_gpus = args.num_gpus
    num_clients = args.num_clients
    num_rounds = args.num_rounds
    affinity = args.affinity
    dataset = args.dataset
    partitioner = args.partitioner

    # Download MNIST dataset and partition it
    mnist_fds = FederatedDataset(
        dataset=f'{dataset}',
        partitioners={
            'train':
                num_clients if partitioner == 'iid' else (
                    DirichletPartitioner(num_clients, alpha=0.1) if partitioner == 'dirichlet' else (
                        LabelPartitioner(num_clients, labels_per_client=10) if partitioner == 'label' else
                        MixPartitioner(num_clients, alpha=0.1)
                    )
                )
        },
    )
    centralized_testset = mnist_fds.load_full('test')

    # Configure the strategy

    if affinity == 'avg' or affinity == 'fedavg':
        strategy = MyFedAvg(
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
        )
        log(INFO, 'FedAvg')
    elif affinity == 'prox' or affinity == 'fedprox':
        strategy = MyFedProx(
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
            proximal_mu=0.3
        )
        log(INFO, 'FedProx')
    elif affinity == 'acc' or affinity == 'fedacc':
        strategy = FedAcc(
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
        )
        log(INFO, 'FedAcc')
    elif affinity == 'avgm' or affinity == 'fedavgm':
        strategy = MyFedAvgM(
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
            # server_learning_rate=0.01,
            # server_momentum=0.9
        )
        log(INFO, 'FedAdam')
    else:
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
        log(INFO, f'FedAP with {affinity}')

    # Resources to be assigned to each virtual client
    client_resources = {
        'num_cpus': max(multiprocessing.cpu_count() // num_clients, 4),
        'num_gpus': num_gpus if not torch.cuda.is_available() else max(1. / num_clients, .1)
    }

    if not os.path.exists('runs'):
        os.mkdir('runs')
    with open(f'runs/loss_{affinity}.txt', 'a+') as fout:
        fout.write('---------------------------------------------\n')
    with open(f'runs/accuracy_{affinity}.txt', 'a+') as fout:
        fout.write('---------------------------------------------\n')
    with open(f'runs/cluster_{affinity}.txt', 'a+') as fout:
        fout.write('---------------------------------------------\n')

    # Start Logger
    fl.common.logger.configure(identifier='Experiment', filename=increment_path(f'runs/log_{affinity}.txt'))

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
