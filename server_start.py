import argparse

import flwr as fl

from clients.fedclient import fit_config, weighted_average, get_evaluate_fn
from strategies.fedacc import FedAcc
from strategies.fedap import FedAP
from strategies.myfedavg import MyFedAvg
from strategies.myfedprox import MyFedProx


def main():
    parser = argparse.ArgumentParser(description='Flower Simulation with PyTorch')

    parser.add_argument('--num_cpus', type=int, default=2, help='Number of CPUs')
    parser.add_argument('--num_gpus', type=float, default=0.0, help='Ratio of GPU memory')
    parser.add_argument('--num_clients', type=int, default=4, help='Number of clients')
    parser.add_argument('--num_rounds', type=int, default=30, help='Number of FL rounds')
    parser.add_argument('--affinity', type=str, default='fedavg', help='Distance function in affinity propagation')

    args = parser.parse_args()

    num_cpus = args.num_cpus
    num_gpus = args.num_gpus
    num_clients = args.num_clients
    num_rounds = args.num_rounds
    affinity = args.affinity

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
        )
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
            proximal_mu=0.3
        )
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
        )
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
            affinity=affinity
        )

    fl.server.start_server(
        server_address='localhost:8080',
        config=fl.server.ServerConfig(num_rounds=3),
        strategy=strategy,
    )


if __name__ == '__main__':
    main()

