from __future__ import annotations

import random
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

CONTEXT_FEATURES = [
    "query_hour",
    "elapsed_minutes",
    "remaining_task_count",
    "remaining_workload",
    "observed_progress",
    "route_phase",
    "recent_pace",
    "task_density",
    "remaining_spread",
    "aoi_transition_burden",
    "weather_severity",
    "congestion_proxy",
    "rcot_minutes",
    "progress_gap",
    "pace_ratio",
    "reference_support",
    "reference_dispersion",
    "reference_ood_probability",
    "rcot_trust",
]

TASK_FEATURES = [
    "distance_to_current",
    "delta_x",
    "delta_y",
    "service_burden",
    "package_count",
    "same_aoi",
    "time_window_slack",
    "task_density",
]


@dataclass
class HSGConfig:
    hidden_dim: int = 64
    max_tasks: int = 32
    dropout: float = 0.10
    cpu_threads: int = 1
    deterministic: bool = True

    def __post_init__(self) -> None:
        if int(self.hidden_dim) < 4 or int(self.max_tasks) < 2 or int(self.cpu_threads) < 1:
            raise ValueError("HSG dimensions and cpu_threads are invalid")
        if not 0.0 <= float(self.dropout) < 1.0:
            raise ValueError("HSG dropout must be within [0, 1)")
        if not isinstance(self.deterministic, bool):
            raise ValueError("HSG deterministic must be a boolean")


@dataclass
class FeatureScaler:
    mean: np.ndarray
    std: np.ndarray

    def __post_init__(self) -> None:
        self.mean = np.asarray(self.mean, dtype=float).reshape(-1)
        self.std = np.asarray(self.std, dtype=float).reshape(-1)
        if len(self.mean) == 0 or len(self.mean) != len(self.std):
            raise ValueError("Scaler mean/std must be nonempty vectors with equal length")
        if not np.isfinite(self.mean).all() or not np.isfinite(self.std).all():
            raise ValueError("Scaler parameters must be finite")
        if (self.std <= 0.0).any():
            raise ValueError("Scaler standard deviations must be positive")

    @classmethod
    def fit(cls, values: np.ndarray) -> FeatureScaler:
        values = np.asarray(values, dtype=float)
        if values.ndim != 2 or len(values) == 0 or not np.isfinite(values).all():
            raise ValueError("Scaler values must be a nonempty finite 2D array")
        mean = values.mean(axis=0)
        std = values.std(axis=0)
        std[std < 1e-6] = 1.0
        return cls(mean=mean, std=std)

    def transform(self, values: np.ndarray) -> np.ndarray:
        values = np.asarray(values, dtype=float)
        if values.ndim != 2 or values.shape[1] != len(self.mean):
            raise ValueError("Scaler input has an incompatible shape")
        if not np.isfinite(values).all():
            raise ValueError("Scaler input contains non-finite values")
        return (values - self.mean) / self.std


class SnapshotTaskDataset(Dataset[dict[str, torch.Tensor]]):
    def __init__(
        self,
        snapshots: pd.DataFrame,
        tasks: pd.DataFrame,
        *,
        max_tasks: int,
        context_scaler: FeatureScaler | None = None,
        task_scaler: FeatureScaler | None = None,
    ) -> None:
        if int(max_tasks) < 2:
            raise ValueError("max_tasks must be at least 2")
        required_snapshot = set(
            CONTEXT_FEATURES + ["snapshot_id", "target_route_remaining_minutes"]
        )
        required_task = set(TASK_FEATURES + ["snapshot_id", "task_id", "target_next"])
        missing_snapshot = required_snapshot.difference(snapshots.columns)
        missing_task = required_task.difference(tasks.columns)
        if missing_snapshot or missing_task:
            raise ValueError(
                f"Missing HSG columns: snapshots={sorted(missing_snapshot)}, tasks={sorted(missing_task)}"
            )
        if snapshots.empty or tasks.empty:
            raise ValueError("HSG datasets must be nonempty")
        if snapshots["snapshot_id"].isna().any() or snapshots["snapshot_id"].duplicated().any():
            raise ValueError("snapshot_id must be nonmissing and unique in snapshots")
        if tasks[["snapshot_id", "task_id"]].isna().any().any():
            raise ValueError("Task snapshot_id and task_id cannot be missing")
        if tasks.duplicated(["snapshot_id", "task_id"]).any():
            raise ValueError("task_id must be unique within each snapshot")
        snapshot_values = (
            snapshots[CONTEXT_FEATURES + ["target_route_remaining_minutes"]]
            .astype(float)
            .to_numpy()
        )
        task_values_for_validation = tasks[TASK_FEATURES].astype(float).to_numpy()
        if (
            not np.isfinite(snapshot_values).all()
            or not np.isfinite(task_values_for_validation).all()
        ):
            raise ValueError("HSG context, task, and target values must be finite")
        if (snapshots["target_route_remaining_minutes"].astype(float) < 0.0).any():
            raise ValueError("HSG ETA targets cannot be negative")
        target_next = pd.to_numeric(tasks["target_next"], errors="coerce")
        if target_next.isna().any() or not target_next.isin([0, 1]).all():
            raise ValueError("target_next must contain only 0/1 values")
        next_counts = (
            tasks.assign(_target_next=target_next).groupby("snapshot_id")["_target_next"].sum()
        )
        if not (next_counts == 1).all():
            raise ValueError("Every task set must contain exactly one target_next task")

        self.snapshots = snapshots.reset_index(drop=True).copy()
        self.tasks_by_snapshot = {
            str(key): value.copy() for key, value in tasks.groupby("snapshot_id")
        }
        missing_task_sets = set(self.snapshots["snapshot_id"].astype(str)).difference(
            self.tasks_by_snapshot
        )
        if missing_task_sets:
            raise ValueError(f"Snapshots without pending tasks: {sorted(missing_task_sets)[:5]}")
        self.max_tasks = int(max_tasks)
        context_values = self.snapshots[CONTEXT_FEATURES].astype(float).to_numpy()
        self.context_scaler = context_scaler or FeatureScaler.fit(context_values)

        all_task_values = tasks[TASK_FEATURES].astype(float).to_numpy()
        self.task_scaler = task_scaler or FeatureScaler.fit(all_task_values)

    def __len__(self) -> int:
        return len(self.snapshots)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        row = self.snapshots.iloc[index]
        # Order and truncate only with observable fields. actual_rank is a future label and
        # must never determine the model input order or which tasks survive truncation.
        task_frame = self.tasks_by_snapshot[str(row["snapshot_id"])].copy()
        task_frame["_task_id_sort"] = task_frame["task_id"].astype(str)
        task_frame = task_frame.sort_values(
            ["distance_to_current", "_task_id_sort"], kind="stable"
        ).head(self.max_tasks)
        task_values = task_frame[TASK_FEATURES].astype(float).to_numpy()
        task_values = self.task_scaler.transform(task_values)
        n_tasks = len(task_values)
        padded = np.zeros((self.max_tasks, len(TASK_FEATURES)), dtype=np.float32)
        padded[:n_tasks] = task_values.astype(np.float32)
        mask = np.zeros(self.max_tasks, dtype=bool)
        mask[:n_tasks] = True
        target_candidates = np.flatnonzero(task_frame["target_next"].to_numpy() == 1)
        # -100 is CrossEntropyLoss.ignore_index when the visible target was excluded by
        # observable-field truncation. ETA loss still uses the full route context features.
        route_target = int(target_candidates[0]) if len(target_candidates) else -100
        context = self.context_scaler.transform(
            row[CONTEXT_FEATURES].astype(float).to_numpy()[None, :]
        )[0]
        return {
            "context": torch.tensor(context, dtype=torch.float32),
            "tasks": torch.tensor(padded, dtype=torch.float32),
            "mask": torch.tensor(mask, dtype=torch.bool),
            "route_target": torch.tensor(route_target, dtype=torch.long),
            "eta_target": torch.tensor(
                float(row["target_route_remaining_minutes"]), dtype=torch.float32
            ),
        }


class HSGETA(nn.Module):
    """Compact hierarchical set-graph ETA model for GPU/CPU reproducible training."""

    def __init__(self, config: HSGConfig) -> None:
        super().__init__()
        h = config.hidden_dim
        self.config = config
        self.context_encoder = nn.Sequential(
            nn.Linear(len(CONTEXT_FEATURES), h),
            nn.GELU(),
            nn.LayerNorm(h),
            nn.Dropout(config.dropout),
            nn.Linear(h, h),
        )
        self.task_encoder = nn.Sequential(
            nn.Linear(len(TASK_FEATURES), h),
            nn.GELU(),
            nn.LayerNorm(h),
            nn.Linear(h, h),
        )
        self.graph_projection = nn.Linear(h, h)
        self.graph_norm = nn.LayerNorm(h)
        self.route_scorer = nn.Sequential(
            nn.Linear(h * 2, h),
            nn.GELU(),
            nn.Linear(h, 1),
        )
        self.eta_head = nn.Sequential(
            nn.Linear(h * 3, h),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(h, 3),
        )

    def forward(
        self,
        context: torch.Tensor,
        tasks: torch.Tensor,
        mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        context_embedding = self.context_encoder(context)
        task_embedding = self.task_encoder(tasks)
        # Spatially weighted message passing over the pending-task graph. Delta x/y are
        # task coordinates relative to the current courier position after scaling.
        coordinates = tasks[..., 1:3]
        pairwise_distance = torch.cdist(coordinates, coordinates)
        edge_mask = mask[:, :, None] & mask[:, None, :]
        adjacency = torch.exp(-pairwise_distance).masked_fill(~edge_mask, 0.0)
        adjacency = adjacency / adjacency.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        graph_message = torch.bmm(adjacency, task_embedding)
        task_embedding = self.graph_norm(task_embedding + self.graph_projection(graph_message))
        context_expanded = context_embedding[:, None, :].expand_as(task_embedding)
        route_logits = self.route_scorer(
            torch.cat([task_embedding, context_expanded], dim=-1)
        ).squeeze(-1)
        route_logits = route_logits.masked_fill(~mask, torch.finfo(route_logits.dtype).min)
        route_weights = torch.softmax(route_logits, dim=-1)
        route_pool = torch.sum(task_embedding * route_weights[..., None], dim=1)
        valid = mask.float().unsqueeze(-1)
        mean_pool = torch.sum(task_embedding * valid, dim=1) / valid.sum(dim=1).clamp_min(1.0)
        raw = self.eta_head(torch.cat([context_embedding, route_pool, mean_pool], dim=-1))
        q10 = torch.nn.functional.softplus(raw[:, 0])
        q50 = q10 + torch.nn.functional.softplus(raw[:, 1])
        q90 = q50 + torch.nn.functional.softplus(raw[:, 2])
        quantiles = torch.stack([q10, q50, q90], dim=1)
        return {"quantiles": quantiles, "route_logits": route_logits}


def quantile_loss(predictions: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    quantiles = torch.tensor([0.10, 0.50, 0.90], device=predictions.device)
    residual = target[:, None] - predictions
    return torch.maximum(quantiles * residual, (quantiles - 1.0) * residual).mean()


def train_hsg_eta(
    train_snapshots: pd.DataFrame,
    train_tasks: pd.DataFrame,
    validation_snapshots: pd.DataFrame,
    validation_tasks: pd.DataFrame,
    *,
    config: HSGConfig,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    amp: bool,
    seed: int,
    output_path: str | Path,
    patience: int = 3,
) -> dict[str, Any]:
    if int(epochs) < 1 or int(batch_size) < 1 or int(patience) < 1:
        raise ValueError("epochs, batch_size, and patience must be positive")
    if not np.isfinite(float(learning_rate)) or float(learning_rate) <= 0.0:
        raise ValueError("learning_rate must be finite and positive")

    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(bool(config.deterministic), warn_only=True)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = bool(config.deterministic)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu":
        torch.set_num_threads(max(1, int(config.cpu_threads)))
    train_dataset = SnapshotTaskDataset(train_snapshots, train_tasks, max_tasks=config.max_tasks)
    validation_dataset = SnapshotTaskDataset(
        validation_snapshots,
        validation_tasks,
        max_tasks=config.max_tasks,
        context_scaler=train_dataset.context_scaler,
        task_scaler=train_dataset.task_scaler,
    )
    pin_memory = device.type == "cuda"
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=pin_memory,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=pin_memory,
    )
    model = HSGETA(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    use_amp = bool(amp and device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    history: list[dict[str, float]] = []
    best_loss = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    epochs_without_improvement = 0
    total_start = perf_counter()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    for epoch in range(1, epochs + 1):
        epoch_start = perf_counter()
        model.train()
        train_losses: list[float] = []
        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            context = batch["context"].to(device, non_blocking=pin_memory)
            tasks = batch["tasks"].to(device, non_blocking=pin_memory)
            mask = batch["mask"].to(device, non_blocking=pin_memory)
            route_target = batch["route_target"].to(device, non_blocking=pin_memory)
            eta_target = batch["eta_target"].to(device, non_blocking=pin_memory)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                output = model(context, tasks, mask)
                eta_loss = quantile_loss(output["quantiles"], eta_target)
                route_loss = nn.functional.cross_entropy(
                    output["route_logits"], route_target, ignore_index=-100
                )
                if torch.isnan(route_loss):
                    route_loss = torch.zeros((), device=device, dtype=eta_loss.dtype)
                loss = eta_loss + 0.10 * route_loss
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            scaler.step(optimizer)
            scaler.update()
            train_losses.append(float(loss.detach().cpu()))

        model.eval()
        validation_losses: list[float] = []
        route_correct = 0
        route_total = 0
        with torch.no_grad():
            for batch in validation_loader:
                context = batch["context"].to(device, non_blocking=pin_memory)
                tasks = batch["tasks"].to(device, non_blocking=pin_memory)
                mask = batch["mask"].to(device, non_blocking=pin_memory)
                route_target = batch["route_target"].to(device, non_blocking=pin_memory)
                eta_target = batch["eta_target"].to(device, non_blocking=pin_memory)
                output = model(context, tasks, mask)
                loss = quantile_loss(output["quantiles"], eta_target)
                validation_losses.append(float(loss.detach().cpu()))
                valid_route = route_target != -100
                if valid_route.any():
                    route_correct += int(
                        (
                            output["route_logits"][valid_route].argmax(dim=1)
                            == route_target[valid_route]
                        )
                        .sum()
                        .item()
                    )
                    route_total += int(valid_route.sum().item())
        validation_loss = float(np.mean(validation_losses))
        row = {
            "epoch": float(epoch),
            "train_loss": float(np.mean(train_losses)),
            "validation_quantile_loss": validation_loss,
            "validation_route_top1": route_correct / max(route_total, 1),
            "epoch_seconds": perf_counter() - epoch_start,
        }
        history.append(row)
        if validation_loss < best_loss - 1e-8:
            best_loss = validation_loss
            best_state = {
                key: value.detach().cpu().clone() for key, value in model.state_dict().items()
            }
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= max(int(patience), 1):
                break

    if best_state is None:
        raise RuntimeError("No HSG-ETA checkpoint was produced")
    payload = {
        "model_state": best_state,
        "model_config": asdict(config),
        "context_mean": train_dataset.context_scaler.mean,
        "context_std": train_dataset.context_scaler.std,
        "task_mean": train_dataset.task_scaler.mean,
        "task_std": train_dataset.task_scaler.std,
        "history": history,
        "device": str(device),
        "amp_enabled": use_amp,
        "deterministic_algorithms": bool(config.deterministic),
        "parameter_count": int(sum(parameter.numel() for parameter in model.parameters())),
        "training_seconds": perf_counter() - total_start,
        "peak_cuda_memory_bytes": (
            int(torch.cuda.max_memory_allocated()) if device.type == "cuda" else 0
        ),
    }
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output_path)
    return {
        "history": history,
        "device": str(device),
        "amp_enabled": use_amp,
        "deterministic_algorithms": bool(config.deterministic),
        "best_loss": best_loss,
        "parameter_count": payload["parameter_count"],
        "training_seconds": payload["training_seconds"],
        "peak_cuda_memory_bytes": payload["peak_cuda_memory_bytes"],
    }


def load_hsg_checkpoint(
    path: str | Path,
) -> tuple[HSGETA, FeatureScaler, FeatureScaler, dict[str, Any]]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise TypeError("HSG checkpoint payload must be a mapping")
    required = {
        "model_config",
        "model_state",
        "context_mean",
        "context_std",
        "task_mean",
        "task_std",
    }
    missing = required.difference(payload)
    if missing:
        raise ValueError(f"HSG checkpoint is missing fields: {sorted(missing)}")
    config = HSGConfig(**payload["model_config"])
    model = HSGETA(config)
    model.load_state_dict(payload["model_state"], strict=True)
    model.eval()
    context_scaler = FeatureScaler(
        np.asarray(payload["context_mean"]), np.asarray(payload["context_std"])
    )
    task_scaler = FeatureScaler(np.asarray(payload["task_mean"]), np.asarray(payload["task_std"]))
    return model, context_scaler, task_scaler, payload


def predict_hsg_eta(
    model: HSGETA,
    snapshots: pd.DataFrame,
    tasks: pd.DataFrame,
    *,
    context_scaler: FeatureScaler,
    task_scaler: FeatureScaler,
    batch_size: int = 64,
) -> pd.DataFrame:
    if int(batch_size) < 1:
        raise ValueError("batch_size must be positive")
    dataset = SnapshotTaskDataset(
        snapshots,
        tasks,
        max_tasks=model.config.max_tasks,
        context_scaler=context_scaler,
        task_scaler=task_scaler,
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    try:
        device = next(model.parameters()).device
    except StopIteration as error:
        raise RuntimeError("HSG model has no parameters") from error
    outputs: list[np.ndarray] = []
    route_accuracy_parts: list[np.ndarray] = []
    route_available_parts: list[np.ndarray] = []
    route_entropy_parts: list[np.ndarray] = []
    route_normalized_entropy_parts: list[np.ndarray] = []
    route_top1_probability_parts: list[np.ndarray] = []
    with torch.no_grad():
        for batch in loader:
            context = batch["context"].to(device)
            task_tensor = batch["tasks"].to(device)
            mask = batch["mask"].to(device)
            route_target = batch["route_target"].to(device)
            result = model(context, task_tensor, mask)
            outputs.append(result["quantiles"].cpu().numpy())
            route_probabilities = torch.softmax(result["route_logits"], dim=1)
            route_correct = result["route_logits"].argmax(dim=1) == route_target
            route_correct = torch.where(
                route_target == -100,
                torch.zeros_like(route_correct),
                route_correct,
            )
            route_available = route_target != -100
            route_accuracy_parts.append(route_correct.cpu().numpy())
            route_available_parts.append(route_available.cpu().numpy())
            entropy = -(route_probabilities * torch.log(route_probabilities.clamp_min(1e-9))).sum(
                dim=1
            )
            valid_counts = mask.sum(dim=1).clamp_min(2).float()
            normalized_entropy = entropy / torch.log(valid_counts)
            route_entropy_parts.append(entropy.cpu().numpy())
            route_normalized_entropy_parts.append(normalized_entropy.cpu().numpy())
            route_top1_probability_parts.append(route_probabilities.max(dim=1).values.cpu().numpy())
    values = np.concatenate(outputs, axis=0)
    route_correct = np.concatenate(route_accuracy_parts, axis=0)
    route_available = np.concatenate(route_available_parts, axis=0)
    route_entropy = np.concatenate(route_entropy_parts, axis=0)
    route_normalized_entropy = np.concatenate(route_normalized_entropy_parts, axis=0)
    route_top1_probability = np.concatenate(route_top1_probability_parts, axis=0)
    return pd.DataFrame(
        {
            "q10": values[:, 0],
            "q50": values[:, 1],
            "q90": values[:, 2],
            "route_top1_correct": route_correct.astype(int),
            "route_target_available": route_available.astype(int),
            "route_entropy": route_entropy,
            "route_normalized_entropy": route_normalized_entropy,
            "route_top1_probability": route_top1_probability,
        },
        index=snapshots.index,
    )
