"""Flower FedAvg server that persists the aggregated global model."""

from pathlib import Path

import flwr as fl
import torch

from model import build_model, trainable_parameters


ROOT_DIR = Path(__file__).resolve().parent
GLOBAL_CHECKPOINT = ROOT_DIR / "checkpoints" / "global_fedchest_model.pt"


class SavingFedAvg(fl.server.strategy.FedAvg):
    """FedAvg strategy which saves the current global trainable state."""

    def aggregate_fit(self, server_round, results, failures):
        aggregated = super().aggregate_fit(server_round, results, failures)
        if aggregated[0] is not None:
            parameters, metrics = aggregated
            model = build_model()
            trainable = list(trainable_parameters(model))
            ndarrays = fl.common.parameters_to_ndarrays(parameters)
            if len(ndarrays) != len(trainable):
                raise RuntimeError("Aggregated parameter count does not match the model")
            for parameter, value in zip(trainable, ndarrays):
                parameter.data.copy_(torch.from_numpy(value).to(parameter.dtype))
            GLOBAL_CHECKPOINT.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"model_state_dict": model.state_dict(), "round": server_round, "metrics": metrics}, GLOBAL_CHECKPOINT)
        return aggregated


if __name__ == "__main__":
    strategy = SavingFedAvg(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=3,
        min_evaluate_clients=3,
        min_available_clients=3,
    )
    fl.server.start_server(
        server_address="0.0.0.0:8080",
        config=fl.server.ServerConfig(num_rounds=3),
        strategy=strategy,
    )
