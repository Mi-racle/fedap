from functools import reduce
from logging import WARNING
from typing import Optional, Callable, Dict, Tuple, List, Union

import numpy as np
from flwr.common import NDArrays, Scalar, Parameters, MetricsAggregationFn, log, FitRes, parameters_to_ndarrays, \
    ndarrays_to_parameters
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg
from sklearn.cluster import AffinityPropagation

WARNING_MIN_AVAILABLE_CLIENTS_TOO_LOW = """
Setting `min_available_clients` lower than `min_fit_clients` or
`min_evaluate_clients` can cause the server to fail when there are too few clients
connected to the server. `min_available_clients` must be set to a value larger
than or equal to the values of `min_fit_clients` and `min_evaluate_clients`.
"""


class FedAP(FedAvg):

    def __init__(
            self,
            *,
            fraction_fit: float = 1.0,
            fraction_evaluate: float = 1.0,
            min_fit_clients: int = 2,
            min_evaluate_clients: int = 2,
            min_available_clients: int = 2,
            evaluate_fn: Optional[
                Callable[
                    [int, NDArrays, Dict[str, Scalar]],
                    Optional[Tuple[float, Dict[str, Scalar]]],
                ]
            ] = None,
            on_fit_config_fn: Optional[Callable[[int], Dict[str, Scalar]]] = None,
            on_evaluate_config_fn: Optional[Callable[[int], Dict[str, Scalar]]] = None,
            accept_failures: bool = True,
            initial_parameters: Optional[Parameters] = None,
            fit_metrics_aggregation_fn: Optional[MetricsAggregationFn] = None,
            evaluate_metrics_aggregation_fn: Optional[MetricsAggregationFn] = None,
    ) -> None:
        super(FedAP, self).__init__(
            fraction_fit=fraction_fit,
            fraction_evaluate=fraction_evaluate,
            min_fit_clients=min_fit_clients,
            min_evaluate_clients=min_evaluate_clients,
            min_available_clients=min_available_clients,
            evaluate_fn=evaluate_fn,
            on_fit_config_fn=on_fit_config_fn,
            on_evaluate_config_fn=on_evaluate_config_fn,
            accept_failures=accept_failures,
            initial_parameters=initial_parameters,
            fit_metrics_aggregation_fn=fit_metrics_aggregation_fn,
            evaluate_metrics_aggregation_fn=evaluate_metrics_aggregation_fn,
        )

    def aggregate_fit(
            self,
            server_round: int,
            results: List[Tuple[ClientProxy, FitRes]],
            failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Aggregate fit results using weighted average."""
        if not results:
            return None, {}
        # Do not aggregate if there are failures and failures are not accepted
        if not self.accept_failures and failures:
            return None, {}

        weights_results = [
            (parameters_to_ndarrays(fit_res.parameters), fit_res.num_examples)
            for _, fit_res in results
        ]
        aggregated_ndarrays = aggregate(weights_results)

        parameters_aggregated = ndarrays_to_parameters(aggregated_ndarrays)

        # Aggregate custom metrics if aggregation fn was provided
        metrics_aggregated = {}
        if self.fit_metrics_aggregation_fn:
            fit_metrics = [(res.num_examples, res.metrics) for _, res in results]
            metrics_aggregated = self.fit_metrics_aggregation_fn(fit_metrics)
        elif server_round == 1:  # Only log this warning once
            log(WARNING, "No fit_metrics_aggregation_fn provided")

        return parameters_aggregated, metrics_aggregated


def aggregate(results: List[Tuple[NDArrays, int]]) -> NDArrays:
    """Compute weighted average."""
    # Calculate the total number of examples used during training
    num_examples_total = sum(num_examples for (_, num_examples) in results)

    # Create a list of weights, each multiplied by the related number of examples
    # list[list[ndarray]] / [num_clients][num_layers][...]
    # weighted_weights = [
    #     [layer * num_examples for layer in weights] for weights, num_examples in results
    # ]

    # Compute average weights of each layer
    # weights_prime: NDArrays = [
    #     reduce(np.add, layer_updates) / num_examples_total
    #     for layer_updates in zip(*weighted_weights)
    # ]

    flattened_weights = np.stack(
        [
            np.concatenate(
                [
                    layer.flatten() for layer in weights
                ]
            ) for weights, num_examples in results
        ]
    )

    clustering = AffinityPropagation(damping=0.5).fit(flattened_weights)
    cluster_labels = clustering.labels_
    max_label = max(cluster_labels)

    weights_prime: NDArrays = []
    for cluster in range(0, max_label + 1):
        temp_weights = []
        cluster_size = 0
        for i, label in enumerate(cluster_labels):
            if label == cluster:
                cluster_size += 1
                temp_weights.append(results[i][0])
        aggregated_weights: NDArrays = [
            reduce(np.add, layer_updates) / cluster_size
            for layer_updates in zip(*temp_weights)
        ]
        for layer in aggregated_weights:
            weights_prime.append(layer)

    weights_prime.append(np.array(cluster_labels))

    # weighted_weights = [
    #     [layer * cluster_labels[i] for layer in weights] for i, (weights, num_examples) in enumerate(results)
    # ]
    #
    # weights_prime: NDArrays = [
    #     reduce(np.add, layer_updates) / num_examples_total
    #     for layer_updates in zip(*weighted_weights)
    # ]

    return weights_prime
