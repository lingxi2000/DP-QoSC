from __future__ import annotations

from typing import Optional

import torch


dp_para = {
    "use_dp": True,
    "epsilon": [1.0, 2.0, 3.0, 4.0, 5.0],
    "noise_types": ["EM"],
    "clip_bounds": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
    "optimal_candidate_count": 50,
    "optimal_count_ablation_values": [10, 20, 30, 40, 50],
    "dataset_settings": {
        "QWS": {
            "proxy_pool_size": 400,
            "near_optimal_probability_fraction": 0.05,
        },
        "Normal": {
            "proxy_pool_size": 100,
            "near_optimal_probability_fraction": 0.10,
        },
    },
    "representative_rho": 0.8,
    "max_utility_degradation": 0.05,
    "delta_ablation_values": [0.10,0.15,0.20,0.25,0.30,0.35,0.40],

    "pool_construction_mechanism": "PNF",
    "pool_construction_epsilon": 1.0,
    "pool_construction_clip_bound": 1.0,
    "representative_numerical_epsilon": 1e-12,
    "pipeline_runs": 20,
}


def dataset_dp_settings(dataset: str, para=None) -> dict:
    para = dp_para if para is None else para
    settings = para.get("dataset_settings", {})
    dataset_name = str(dataset).strip()

    for configured_name, configured_values in settings.items():
        if str(configured_name).lower() == dataset_name.lower():
            return dict(configured_values)

    raise ValueError(
        f"No dataset-specific DP settings are configured for dataset={dataset!r}."
    )


def validate_dp_para(para=None) -> None:
    para = dp_para if para is None else para

    epsilon_values = para.get("epsilon", [])
    noise_types = para.get("noise_types", [])
    clip_bounds = para.get("clip_bounds", [])

    if not isinstance(epsilon_values, (list, tuple)) or not epsilon_values:
        raise ValueError("dp_para['epsilon'] must be a non-empty list or tuple.")
    if any(float(value) <= 0.0 for value in epsilon_values):
        raise ValueError("Every privacy budget in dp_para['epsilon'] must be positive.")

    if not isinstance(noise_types, (list, tuple)) or not noise_types:
        raise ValueError("dp_para['noise_types'] must be a non-empty list or tuple.")
    allowed_mechanisms = {"PNF", "EM"}
    unknown_mechanisms = {str(value) for value in noise_types} - allowed_mechanisms
    if unknown_mechanisms:
        raise ValueError(
            f"Unsupported sequence-level mechanisms: {sorted(unknown_mechanisms)}"
        )

    if not isinstance(clip_bounds, (list, tuple)) or not clip_bounds:
        raise ValueError("dp_para['clip_bounds'] must be a non-empty list or tuple.")
    if any(not 0.0 < float(value) <= 1.0 for value in clip_bounds):
        raise ValueError("Every clipping threshold must satisfy 0 < tau <= 1.")

    optimal_candidate_count = int(para.get("optimal_candidate_count", 0))
    if optimal_candidate_count <= 0:
        raise ValueError("dp_para['optimal_candidate_count'] must be positive.")

    optimal_count_values = para.get("optimal_count_ablation_values", [])
    if not isinstance(optimal_count_values, (list, tuple)) or not optimal_count_values:
        raise ValueError(
            "optimal_count_ablation_values must be a non-empty list or tuple."
        )
    if any(float(value) != int(float(value)) for value in optimal_count_values):
        raise ValueError("Every OptCount ablation value must be an integer.")
    if any(int(float(value)) <= 0 for value in optimal_count_values):
        raise ValueError("Every OptCount ablation value must be positive.")
    if len({int(float(value)) for value in optimal_count_values}) != len(optimal_count_values):
        raise ValueError("optimal_count_ablation_values must not contain duplicates.")

    dataset_settings = para.get("dataset_settings", {})
    if not isinstance(dataset_settings, dict) or not dataset_settings:
        raise ValueError("dp_para['dataset_settings'] must be a non-empty dict.")
    for dataset_name, values in dataset_settings.items():
        if not isinstance(values, dict):
            raise ValueError(f"dataset_settings[{dataset_name!r}] must be a dict.")
        proxy_pool_size = int(values.get("proxy_pool_size", 0))
        if proxy_pool_size <= 0:
            raise ValueError(
                f"dataset_settings[{dataset_name!r}]['proxy_pool_size'] must be positive."
            )
        nop_fraction = float(values.get("near_optimal_probability_fraction", 0.0))
        if not 0.0 < nop_fraction < 1.0:
            raise ValueError(
                f"dataset_settings[{dataset_name!r}]['near_optimal_probability_fraction'] "
                "must satisfy 0 < alpha < 1."
            )

    rho = float(para.get("representative_rho", 0.0))
    if not 0.0 < rho <= 1.0:
        raise ValueError("representative_rho must satisfy 0 < rho <= 1.")

    delta = float(para.get("max_utility_degradation", -1.0))
    if not 0.0 <= delta <= 1.0:
        raise ValueError("max_utility_degradation must satisfy 0 <= delta <= 1.")

    delta_values = para.get("delta_ablation_values", [])
    if not isinstance(delta_values, (list, tuple)) or not delta_values:
        raise ValueError("delta_ablation_values must be a non-empty list or tuple.")
    if any(not 0.0 <= float(value) <= 1.0 for value in delta_values):
        raise ValueError("Every delta ablation value must satisfy 0 <= delta <= 1.")
    if len({float(value) for value in delta_values}) != len(delta_values):
        raise ValueError("delta_ablation_values must not contain duplicates.")

    construction_mechanism = str(para.get("pool_construction_mechanism", "PNF"))
    if construction_mechanism not in {"PNF", "EM"}:
        raise ValueError("pool_construction_mechanism must be 'PNF' or 'EM'.")

    construction_epsilon = float(para.get("pool_construction_epsilon", 0.0))
    if construction_epsilon <= 0.0:
        raise ValueError("pool_construction_epsilon must be positive.")

    construction_clip_bound = float(para.get("pool_construction_clip_bound", 0.0))
    if not 0.0 < construction_clip_bound <= 1.0:
        raise ValueError(
            "pool_construction_clip_bound must satisfy 0 < tau <= 1."
        )

    numerical_epsilon = float(para.get("representative_numerical_epsilon", 0.0))
    if numerical_epsilon <= 0.0:
        raise ValueError("representative_numerical_epsilon must be positive.")

    pipeline_runs = int(para.get("pipeline_runs", 0))
    if pipeline_runs <= 0:
        raise ValueError("dp_para['pipeline_runs'] must be positive.")


class Mechanisms:
    @staticmethod
    def _validate_scores(scores: torch.Tensor) -> None:
        if scores.ndim != 2:
            raise ValueError(
                "scores must have shape [batch_size, candidate_count]."
            )

        if scores.shape[1] == 0:
            raise ValueError("scores must contain at least one candidate.")

        if not torch.isfinite(scores).all():
            raise ValueError("scores contain NaN or infinity.")

    @staticmethod
    def exponential_mechanism(
        scores: torch.Tensor,
        epsilon: float,
        sensitivity: float,
    ) -> torch.Tensor:
        epsilon = float(epsilon)
        sensitivity = float(sensitivity)
        Mechanisms._validate_scores(scores)

        if epsilon <= 0.0:
            raise ValueError("epsilon must be positive.")

        if sensitivity <= 0.0:
            raise ValueError("sensitivity must be positive.")

        logits = epsilon * scores / (2.0 * sensitivity)
        logits = logits - torch.max(logits, dim=1, keepdim=True).values
        probabilities = torch.softmax(logits, dim=1)

        selected_indices = torch.multinomial(
            probabilities,
            num_samples=1,
            replacement=True,
        )
        return selected_indices.squeeze(1)

    @staticmethod
    def pnf_mechanism(
        scores: torch.Tensor,
        epsilon: float,
        sensitivity: float,
    ) -> torch.Tensor:
        epsilon = float(epsilon)
        sensitivity = float(sensitivity)
        Mechanisms._validate_scores(scores)

        if epsilon <= 0.0:
            raise ValueError("epsilon must be positive.")

        if sensitivity <= 0.0:
            raise ValueError("sensitivity must be positive.")

        empirical_max = torch.max(scores, dim=1, keepdim=True).values
        probabilities = torch.exp(
            epsilon * (scores - empirical_max) / (2.0 * sensitivity)
        )
        probabilities = torch.clamp(probabilities, min=0.0, max=1.0)

        batch_size, candidate_count = scores.shape
        selected = torch.empty(
            batch_size,
            dtype=torch.long,
            device=scores.device,
        )

        for batch_index in range(batch_size):
            permutation = torch.randperm(candidate_count, device=scores.device)
            ordered_probabilities = probabilities[batch_index, permutation]
            draws = torch.rand(
                candidate_count,
                dtype=scores.dtype,
                device=scores.device,
            )
            successes = draws < ordered_probabilities
            success_positions = torch.nonzero(successes, as_tuple=False).flatten()

            if success_positions.numel() == 0:
                raise RuntimeError(
                    "PNF failed to terminate although an empirical-max "
                    "candidate should have acceptance probability 1."
                )

            first_success = int(success_positions[0].item())
            selected[batch_index] = permutation[first_success]

        return selected

    @staticmethod
    def Non_DP(
        scores: torch.Tensor,
        epsilon: Optional[float] = None,
        sensitivity: Optional[float] = None,
    ) -> torch.Tensor:
        del epsilon, sensitivity
        Mechanisms._validate_scores(scores)
        return torch.argmax(scores, dim=1)

    @staticmethod
    def add_noise(
        scores: torch.Tensor,
        dp_para_local,
        sensitivity: float,
    ) -> torch.Tensor:
        if not dp_para_local or not dp_para_local.get("use_dp", False):
            return Mechanisms.Non_DP(scores)

        if "noise_type" not in dp_para_local:
            raise ValueError(
                "dp_para must contain scalar key 'noise_type' when calling add_noise."
            )

        noise_type = str(dp_para_local["noise_type"])

        if noise_type == "Non_DP":
            return Mechanisms.Non_DP(scores)

        epsilon = dp_para_local["epsilon"]
        if isinstance(epsilon, (list, tuple)):
            raise ValueError(
                "dp_para['epsilon'] must be scalar when calling add_noise."
            )

        epsilon = float(epsilon)
        sensitivity = float(sensitivity)

        if epsilon <= 0.0:
            raise ValueError("epsilon must be positive.")

        if sensitivity <= 0.0:
            raise ValueError("sensitivity must be positive.")

        if noise_type == "PNF":
            return Mechanisms.pnf_mechanism(
                scores=scores,
                epsilon=epsilon,
                sensitivity=sensitivity,
            )

        if noise_type == "EM":
            return Mechanisms.exponential_mechanism(
                scores=scores,
                epsilon=epsilon,
                sensitivity=sensitivity,
            )

        raise ValueError(f"Unknown sequence-level noise_type: {noise_type}")


def clip_scores(scores: torch.Tensor, clip_bound: float) -> torch.Tensor:
    clip_bound = float(clip_bound)

    if not 0.0 < clip_bound <= 1.0:
        raise ValueError("clip_bound must satisfy 0 < tau <= 1.")

    if not torch.isfinite(scores).all():
        raise ValueError("scores contain NaN or infinity.")

    return torch.clamp(scores, min=0.0, max=clip_bound)
