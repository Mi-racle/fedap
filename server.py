import argparse

import flwr as fl
from datasets import disable_progress_bar
from flwr_datasets import FederatedDataset

from client import NUM_CLIENTS, fit_config, weighted_average, get_evaluate_fn, get_client_fn


def main():

    # Parse input arguments
    parser = argparse.ArgumentParser(description="Flower Simulation with PyTorch")

    parser.add_argument(
        "--num_cpus",
        type=int,
        default=1,
        help="Number of CPUs to assign to a virtual client",
    )
    parser.add_argument(
        "--num_gpus",
        type=float,
        default=0.0,
        help="Ratio of GPU memory to assign to a virtual client",
    )
    parser.add_argument("--num_rounds", type=int, default=10, help="Number of FL rounds.")

    args = parser.parse_args()

    # Download MNIST dataset and partition it
    mnist_fds = FederatedDataset(dataset="./mnist", partitioners={"train": NUM_CLIENTS})
    centralized_testset = mnist_fds.load_full("test")

    # Configure the strategy
    strategy = fl.server.strategy.FedAvg(
        fraction_fit=0.1,  # Sample 10% of available clients for training
        fraction_evaluate=0.05,  # Sample 5% of available clients for evaluation
        min_fit_clients=10,  # Never sample less than 10 clients for training
        min_evaluate_clients=5,  # Never sample less than 5 clients for evaluation
        min_available_clients=int(
            NUM_CLIENTS * 0.75
        ),  # Wait until at least 75 clients are available
        on_fit_config_fn=fit_config,
        evaluate_metrics_aggregation_fn=weighted_average,  # Aggregate federated metrics
        evaluate_fn=get_evaluate_fn(centralized_testset),  # Global evaluation function
    )

    # Resources to be assigned to each virtual client
    client_resources = {
        "num_cpus": args.num_cpus,
        "num_gpus": args.num_gpus,
    }

    # Start simulation
    fl.simulation.start_simulation(
        client_fn=get_client_fn(mnist_fds),
        num_clients=NUM_CLIENTS,
        client_resources=client_resources,
        config=fl.server.ServerConfig(num_rounds=args.num_rounds),
        strategy=strategy,
        actor_kwargs={
            "on_actor_init_fn": disable_progress_bar  # disable tqdm on each actor/process spawning virtual clients
        },
    )


if __name__ == "__main__":

    main()
