from functools import reduce
from logging import INFO, WARNING
from typing import Optional, Callable, Dict, Tuple, List, Union

import numpy as np
from flwr.common import NDArrays, Scalar, Parameters, MetricsAggregationFn, log, FitRes, parameters_to_ndarrays, \
    ndarrays_to_parameters, EvaluateRes
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg
from flwr.server.strategy.aggregate import weighted_loss_avg

from cluster import MyAffinityPropagation


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
            affinity: str = 'euclidean',
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
        self.affinity = affinity
        log(INFO, f"Affinity: {affinity}")

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
            (parameters_to_ndarrays(fit_res.parameters), fit_res.metrics['accuracy'])
            for _, fit_res in results
        ]
        aggregated_ndarrays = self._aggregate(weights_results, self.affinity)

        parameters_aggregated = ndarrays_to_parameters(aggregated_ndarrays)

        # Aggregate custom metrics if aggregation fn was provided
        metrics_aggregated = {}
        if self.fit_metrics_aggregation_fn:
            fit_metrics = [(res.num_examples, res.metrics) for _, res in results]
            metrics_aggregated = self.fit_metrics_aggregation_fn(fit_metrics)
        elif server_round == 1:  # Only log this warning once
            log(WARNING, "No fit_metrics_aggregation_fn provided")

        return parameters_aggregated, metrics_aggregated

    def aggregate_evaluate(
            self,
            server_round: int,
            results: List[Tuple[ClientProxy, EvaluateRes]],
            failures: List[Union[Tuple[ClientProxy, EvaluateRes], BaseException]],
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        """Aggregate evaluation losses using weighted average."""
        if not results:
            return None, {}
        # Do not aggregate if there are failures and failures are not accepted
        if not self.accept_failures and failures:
            return None, {}

        # Aggregate loss
        loss_aggregated = weighted_loss_avg(
            [
                (evaluate_res.num_examples, evaluate_res.loss)
                for _, evaluate_res in results
            ]
        )
        confusion_matrix = [[0 for __ in range(53)] for _ in range(53)]
        for _, evaluate_res in results:
            confusion_matrix += evaluate_res.metrics['confusion_matrix']
        confusion_matrix = np.array(confusion_matrix)
        np.savetxt(f'runs/confusion_matrix_{self.affinity}.csv', confusion_matrix, delimiter=',')

        # Aggregate custom metrics if aggregation fn was provided
        metrics_aggregated = {}
        if self.evaluate_metrics_aggregation_fn:
            eval_metrics = [(res.num_examples, res.metrics) for _, res in results]
            metrics_aggregated = self.evaluate_metrics_aggregation_fn(eval_metrics)
        elif server_round == 1:  # Only log this warning once
            log(WARNING, "No evaluate_metrics_aggregation_fn provided")

        with open(f'runs/loss_{self.affinity}.txt', 'a+') as fout:
            fout.write(str(loss_aggregated) + '\n')
        with open(f'runs/accuracy_{self.affinity}.txt', 'a+') as fout:
            fout.write(str(metrics_aggregated['accuracy']) + '\n')

        return loss_aggregated, metrics_aggregated

    def _aggregate(self, results: List[Tuple[NDArrays, int]], affinity: str) -> NDArrays:
        """Compute weighted average."""

        accuracy_total = sum(accuracy for _, accuracy in results)

        flattened_weights = np.stack(
            [
                np.concatenate(
                    [
                        layer.flatten() for layer in weights
                    ]
                ) for weights, accuracy in results
            ]
        )

        clustering = MyAffinityPropagation(damping=0.5, affinity=affinity).fit(flattened_weights)
        # clustering = MyAffinityPropagation(damping=0.5, affinity=affinity, preference=-87).fit(flattened_weights)
        cluster_labels = clustering.labels_
        with open(f'runs/cluster_{self.affinity}.txt', 'a+') as fout:
            for cluster_label in cluster_labels:
                fout.write(str(cluster_label) + ',')
            fout.write('\n')
        max_label = max(cluster_labels)
        log(INFO, f'number of cluster: {max_label + 1}')
        weights_prime: NDArrays = []
        cluster_accuracy = 0.
        for cluster in range(0, max_label + 1):
            temp_weights = []
            cluster_size = 0
            for i, label in enumerate(cluster_labels):
                if label == cluster:
                    cluster_size += 1
                    for l in range(len(results[i][0])):
                        results[i][0][l] *= results[i][1]
                    temp_weights.append(results[i][0])
                    cluster_accuracy += results[i][1]

            aggregated_weights: NDArrays = [
                reduce(np.add, layer_updates) / cluster_accuracy
                for layer_updates in zip(*temp_weights)
            ]
            for layer in aggregated_weights:
                weights_prime.append(layer)

        weights_prime.append(np.array(cluster_labels))

        return weights_prime
