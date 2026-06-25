from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class SnapshotRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False, strict=True)

    city: str = Field(min_length=1, max_length=128)
    query_hour: float = Field(ge=0.0, lt=24.0)
    elapsed_minutes: float = Field(ge=0.0)
    completed_task_count: int = Field(ge=1)
    remaining_task_count: int = Field(ge=1)
    initial_task_count: int = Field(ge=2)
    completed_workload: float = Field(ge=0.0)
    remaining_workload: float = Field(ge=0.0)
    initial_workload: float = Field(gt=0.0)
    observed_progress: float = Field(ge=0.0, le=1.0)
    route_phase: float = Field(ge=0.0, le=1.0)
    recent_pace: float = Field(ge=0.0)
    task_density: float = Field(ge=0.0)
    remaining_spread: float = Field(ge=0.0)
    aoi_transition_burden: float = Field(ge=0.0, le=1.0)
    weather_severity: float = Field(ge=0.0, le=1.0)
    congestion_proxy: float = Field(gt=0.0)
    trajectory_missingness: float = Field(ge=0.0, le=1.0)

    @model_validator(mode="after")
    def validate_internal_consistency(self) -> SnapshotRequest:
        if not self.city.strip():
            raise ValueError("city cannot be blank")
        if self.completed_task_count + self.remaining_task_count != self.initial_task_count:
            raise ValueError(
                "completed_task_count + remaining_task_count must equal initial_task_count"
            )
        workload_sum = self.completed_workload + self.remaining_workload
        workload_tolerance = max(1.0, 0.02 * self.initial_workload)
        if abs(workload_sum - self.initial_workload) > workload_tolerance:
            raise ValueError(
                "completed_workload + remaining_workload is inconsistent with initial_workload"
            )
        expected_progress = self.completed_workload / self.initial_workload
        if abs(expected_progress - self.observed_progress) > 0.05:
            raise ValueError("observed_progress is inconsistent with workload progress")
        expected_phase = self.completed_task_count / self.initial_task_count
        if abs(expected_phase - self.route_phase) > 0.05:
            raise ValueError("route_phase is inconsistent with task counts")
        return self


class ETAPrediction(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False, strict=True)

    q10: float = Field(ge=0.0)
    q50: float = Field(ge=0.0)
    q90: float = Field(ge=0.0)
    tail_threshold_minutes: float = Field(ge=0.0)
    tail_probability: float = Field(ge=0.0, le=1.0)
    expected_excess_minutes: float = Field(ge=0.0)
    tail_risk: float = Field(ge=0.0)
    reliability_adjusted_priority: float = Field(ge=0.0)
    rcot_minutes: float = Field(ge=0.0)
    reference_support: float = Field(ge=0.0, le=1.0)
    rcot_trust: float = Field(ge=0.0, le=1.0)
    reliability_status: Literal["AUTO", "REVIEW"]
    quantile_champion: str = Field(min_length=1)
    rcot_promotion_gate: Literal["PROMOTE", "HOLD"]
    model_version: str = Field(min_length=1)

    @model_validator(mode="after")
    def validate_prediction_consistency(self) -> ETAPrediction:
        if not self.q10 <= self.q50 <= self.q90:
            raise ValueError("Prediction quantiles must satisfy q10 <= q50 <= q90")
        if self.reliability_adjusted_priority > self.tail_risk + 1e-9:
            raise ValueError("Reliability-adjusted priority cannot exceed unadjusted tail risk")
        return self


class BatchPredictionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    snapshots: list[SnapshotRequest] = Field(min_length=1, max_length=1000)


class BatchPredictionResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    predictions: list[ETAPrediction]


class TriageRequest(BatchPredictionRequest):
    capacity: float = Field(gt=0.0, le=1.0)


class TriageSelection(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    request_index: int
    priority_rank: int
    prediction: ETAPrediction


class TriageResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    capacity: float
    selected_count: int
    decision_policy: str
    selected: list[TriageSelection]
