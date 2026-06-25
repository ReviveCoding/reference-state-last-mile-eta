from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import threading
import time
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Literal, cast

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Request

from reference_eta import __version__
from reference_eta.decisions.triage import derive_tail_scores, threshold_for_rows
from reference_eta.features.rcot import ReferenceOperationalTimeTransformer
from reference_eta.models.baselines import QuantileLightGBMModel
from reference_eta.models.calibration import ConformalQuantileCalibrator
from reference_eta.serving.body_limit import RequestBodyLimitMiddleware
from reference_eta.serving.schemas import (
    BatchPredictionRequest,
    BatchPredictionResponse,
    ETAPrediction,
    SnapshotRequest,
    TriageRequest,
    TriageResponse,
    TriageSelection,
)

ROOT = Path(__file__).resolve().parents[3]
ARTIFACT_DIR = Path(os.getenv("REFERENCE_ETA_ARTIFACT_DIR", ROOT / "artifacts"))
LOGGER = logging.getLogger("reference_eta.serving")
_BUNDLE_ID_PATTERN = re.compile(r"[0-9a-f]{20}")
_REQUEST_ID_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,127}")
_ARTIFACT_CACHE_LOCK = threading.RLock()
_ARTIFACT_CACHE_KEY: tuple[str, str] | None = None
_ARTIFACT_CACHE_VALUE: ArtifactBundle | None = None  # assigned after class construction


def _safe_request_id(raw: str | None) -> str:
    if raw is not None and _REQUEST_ID_PATTERN.fullmatch(raw):
        return raw
    return uuid.uuid4().hex


def _batch_limit() -> int:
    raw = os.getenv("REFERENCE_ETA_MAX_BATCH_SIZE", "256")
    try:
        value = int(raw)
    except ValueError as error:
        raise ValueError("REFERENCE_ETA_MAX_BATCH_SIZE must be an integer") from error
    if not 1 <= value <= 1000:
        raise ValueError("REFERENCE_ETA_MAX_BATCH_SIZE must be between 1 and 1000")
    return value


def _max_request_bytes() -> int:
    raw = os.getenv("REFERENCE_ETA_MAX_REQUEST_BYTES", "1048576")
    try:
        value = int(raw)
    except ValueError as error:
        raise ValueError("REFERENCE_ETA_MAX_REQUEST_BYTES must be an integer") from error
    if not 1024 <= value <= 100 * 1024 * 1024:
        raise ValueError("REFERENCE_ETA_MAX_REQUEST_BYTES must be between 1024 and 104857600")
    return value


def _enforce_batch_limit(size: int) -> None:
    try:
        limit = _batch_limit()
    except ValueError as exc:
        raise HTTPException(
            status_code=503, detail="Serving batch configuration is invalid"
        ) from exc
    if size > limit:
        raise HTTPException(
            status_code=413,
            detail=f"Batch contains {size} snapshots; configured maximum is {limit}",
        )


def _validate_deployment_pins(bundle_id: str, manifest_sha256: str) -> None:
    expected_bundle = os.getenv("REFERENCE_ETA_EXPECTED_BUNDLE_ID")
    if expected_bundle is not None:
        if not _BUNDLE_ID_PATTERN.fullmatch(expected_bundle):
            raise ValueError("REFERENCE_ETA_EXPECTED_BUNDLE_ID is invalid")
        if bundle_id != expected_bundle:
            raise ValueError("Serving bundle does not match the deployment-pinned bundle ID")
    expected_manifest = os.getenv("REFERENCE_ETA_EXPECTED_MANIFEST_SHA256")
    if expected_manifest is not None:
        if not re.fullmatch(r"[0-9a-f]{64}", expected_manifest):
            raise ValueError("REFERENCE_ETA_EXPECTED_MANIFEST_SHA256 is invalid")
        if manifest_sha256 != expected_manifest:
            raise ValueError("Serving manifest does not match the deployment-pinned digest")


@dataclass(frozen=True)
class ArtifactBundle:
    rcot: ReferenceOperationalTimeTransformer
    model: QuantileLightGBMModel
    calibrator: ConformalQuantileCalibrator
    thresholds: dict[tuple[str, int], float]
    release_decision: dict[str, Any]
    provenance: dict[str, Any] | None = None
    integrity_verified: bool = False
    bundle_id: str = "legacy"
    manifest_sha256: str = "unknown"
    model_version: str = __version__


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _verify_artifact_integrity(paths: dict[str, Path], manifest_path: Path) -> None:
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing artifact manifest: {manifest_path}")
    raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise TypeError("Artifact manifest must contain a list of file records")
    records: dict[str, list[dict[str, Any]]] = {}
    for item in raw:
        if not isinstance(item, dict) or not {"path", "size_bytes", "sha256"}.issubset(item):
            raise ValueError("Artifact manifest contains an invalid record")
        records.setdefault(Path(str(item["path"])).name, []).append(item)

    for name, path in paths.items():
        candidates = records.get(path.name, [])
        if len(candidates) != 1:
            raise ValueError(f"Manifest must contain exactly one record for {name}: {path.name}")
        record = candidates[0]
        if int(record["size_bytes"]) != path.stat().st_size:
            raise ValueError(f"Artifact size mismatch for {path.name}")
        if str(record["sha256"]) != _sha256(path):
            raise ValueError(f"Artifact checksum mismatch for {path.name}")


def _validate_release_decision(release_decision: dict[str, Any]) -> None:
    if int(release_decision.get("artifact_schema_version", -1)) != 1:
        raise ValueError("Unsupported or missing artifact_schema_version")
    release_gate = str(release_decision.get("release_gate", ""))
    allow_unreleased = os.getenv("REFERENCE_ETA_ALLOW_UNRELEASED", "0") == "1"
    if not release_gate.startswith("PASS_") and not allow_unreleased:
        raise ValueError(
            f"Refusing to serve unreleased artifacts: {release_gate or 'missing gate'}"
        )
    if release_decision.get("quantile_champion") not in {"with_rcot", "without_rcot"}:
        raise ValueError("Unsupported quantile_champion")
    if release_decision.get("rcot_promotion_gate") not in {"PROMOTE", "HOLD"}:
        raise ValueError("Unsupported rcot_promotion_gate")
    if release_decision.get("decision_champion") not in {
        "q50_eta",
        "q90_eta",
        "tail_risk",
        "reliability_adjusted_priority",
    }:
        raise ValueError("Unsupported decision_champion")
    if (
        release_decision["quantile_champion"] == "with_rcot"
        and release_decision["rcot_promotion_gate"] != "PROMOTE"
    ):
        raise ValueError("with_rcot champion requires a PROMOTE gate")


def _validate_runtime_compatibility(provenance: dict[str, Any]) -> str:
    if int(provenance.get("provenance_schema_version", -1)) != 2:
        raise ValueError("Unsupported or missing provenance_schema_version")
    project = provenance.get("project", {})
    if not isinstance(project, dict):
        raise TypeError("Run provenance project must be a mapping")
    if str(project.get("name", "")) != "reference-state-last-mile-eta":
        raise ValueError("Run provenance project name is invalid")
    model_version = str(project.get("version", ""))
    if not model_version:
        raise ValueError("Run provenance is missing the trained project version")
    allow_artifact_mismatch = os.getenv("REFERENCE_ETA_ALLOW_ARTIFACT_VERSION_MISMATCH", "0") == "1"
    if model_version != __version__ and not allow_artifact_mismatch:
        raise ValueError(
            f"Artifact/code version mismatch: trained={model_version}, serving={__version__}"
        )
    packages = provenance.get("runtime", {}).get("packages", {})
    if not isinstance(packages, dict):
        raise TypeError("Run provenance runtime.packages must be a mapping")
    allow_mismatch = os.getenv("REFERENCE_ETA_ALLOW_VERSION_MISMATCH", "0") == "1"
    for package_name in ("scikit-learn", "lightgbm", "joblib"):
        expected = str(packages.get(package_name, ""))
        if not expected or expected == "not-installed":
            raise ValueError(f"Run provenance is missing the trained {package_name} version")
        try:
            actual = version(package_name)
        except PackageNotFoundError as error:
            raise ValueError(
                f"Required runtime package is not installed: {package_name}"
            ) from error
        if actual != expected and not allow_mismatch:
            raise ValueError(
                f"Runtime package mismatch for {package_name}: trained={expected}, serving={actual}"
            )
    return model_version


def _serving_policy(bundle: ArtifactBundle) -> dict[str, float]:
    raw = bundle.release_decision.get("serving_policy", {})
    if not isinstance(raw, dict):
        raise TypeError("release_decision.serving_policy must be a mapping")
    defaults = {
        "trust_review_threshold": 0.35,
        "support_review_threshold": 0.40,
        "tail_probability_review_threshold": 0.50,
    }
    result = {key: float(raw.get(key, value)) for key, value in defaults.items()}
    for key in (
        "trust_review_threshold",
        "support_review_threshold",
        "tail_probability_review_threshold",
    ):
        if not 0.0 <= result[key] <= 1.0:
            raise ValueError(f"{key} must be between 0 and 1")
    return result


def _load_artifacts_uncached(bundle_dir_text: str, manifest_sha256: str) -> ArtifactBundle:
    # A content digest, not only metadata timestamps, is part of the cache key. Immutable
    # versioned bundles allow a running process and a concurrently starting process to keep
    # serving the previous verified bundle until the atomic pointer switches.
    bundle_dir = Path(bundle_dir_text)
    manifest_path = bundle_dir / "artifact_manifest.json"
    if _sha256(manifest_path) != manifest_sha256:
        raise ValueError("Serving-bundle manifest checksum mismatch")
    paths = {
        "rcot": bundle_dir / "rcot.joblib",
        "model": bundle_dir / "quantile_champion.joblib",
        "calibrator": bundle_dir / "cqr_calibrator.joblib",
        "thresholds": bundle_dir / "tail_thresholds.joblib",
        "release": bundle_dir / "release_decision.json",
        "provenance": bundle_dir / "run_provenance.json",
    }
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Run `make smoke` first. Missing artifacts: {missing}")
    _verify_artifact_integrity(paths, manifest_path)

    thresholds = joblib.load(paths["thresholds"])
    if not isinstance(thresholds, dict):
        raise TypeError("tail_thresholds.joblib does not contain a threshold mapping")
    release_decision = json.loads(paths["release"].read_text(encoding="utf-8"))
    if not isinstance(release_decision, dict):
        raise TypeError("release_decision.json must contain an object")
    provenance = json.loads(paths["provenance"].read_text(encoding="utf-8"))
    if not isinstance(provenance, dict):
        raise TypeError("run_provenance.json must contain an object")
    _validate_release_decision(release_decision)
    model_version = _validate_runtime_compatibility(provenance)
    bundle = ArtifactBundle(
        rcot=ReferenceOperationalTimeTransformer.load(str(paths["rcot"])),
        model=QuantileLightGBMModel.load(str(paths["model"])),
        calibrator=ConformalQuantileCalibrator.load(str(paths["calibrator"])),
        thresholds=thresholds,
        release_decision=release_decision,
        provenance=provenance,
        integrity_verified=True,
        bundle_id=(bundle_dir.name if _BUNDLE_ID_PATTERN.fullmatch(bundle_dir.name) else "legacy"),
        manifest_sha256=manifest_sha256,
        model_version=model_version,
    )
    _serving_policy(bundle)
    return bundle


def _clear_artifact_cache() -> None:
    global _ARTIFACT_CACHE_KEY, _ARTIFACT_CACHE_VALUE
    with _ARTIFACT_CACHE_LOCK:
        _ARTIFACT_CACHE_KEY = None
        _ARTIFACT_CACHE_VALUE = None


def _load_artifacts_for_manifest(bundle_dir_text: str, manifest_sha256: str) -> ArtifactBundle:
    """Load exactly once per immutable manifest, even under concurrent first requests."""

    global _ARTIFACT_CACHE_KEY, _ARTIFACT_CACHE_VALUE
    key = (str(bundle_dir_text), str(manifest_sha256))
    with _ARTIFACT_CACHE_LOCK:
        if _ARTIFACT_CACHE_KEY == key and _ARTIFACT_CACHE_VALUE is not None:
            return _ARTIFACT_CACHE_VALUE
        bundle = _load_artifacts_uncached(*key)
        _ARTIFACT_CACHE_KEY = key
        _ARTIFACT_CACHE_VALUE = bundle
        return bundle


_load_artifacts_for_manifest.cache_clear = _clear_artifact_cache  # type: ignore[attr-defined]


def _load_artifacts() -> ArtifactBundle:
    pointer_path = ARTIFACT_DIR / "current_bundle.json"
    if pointer_path.exists():
        pointer = json.loads(pointer_path.read_text(encoding="utf-8"))
        if not isinstance(pointer, dict):
            raise TypeError("current_bundle.json must contain an object")
        if int(pointer.get("artifact_schema_version", -1)) != 1:
            raise ValueError("Unsupported current-bundle schema")
        bundle_id = str(pointer.get("bundle_id", ""))
        if not _BUNDLE_ID_PATTERN.fullmatch(bundle_id):
            raise ValueError("Invalid serving bundle identifier")
        expected_manifest_sha = str(pointer.get("manifest_sha256", ""))
        if not re.fullmatch(r"[0-9a-f]{64}", expected_manifest_sha):
            raise ValueError("Invalid serving-bundle manifest digest")
        _validate_deployment_pins(bundle_id, expected_manifest_sha)
        bundles_root = (ARTIFACT_DIR / "serving_bundles").resolve()
        bundle_dir = ARTIFACT_DIR / "serving_bundles" / bundle_id
        if not bundle_dir.is_dir():
            raise FileNotFoundError(f"Serving bundle does not exist: {bundle_id}")
        resolved_bundle = bundle_dir.resolve()
        if not resolved_bundle.is_relative_to(bundles_root) or bundle_dir.is_symlink():
            raise ValueError("Serving bundle path escapes the configured artifact directory")
        return _load_artifacts_for_manifest(str(resolved_bundle), expected_manifest_sha)

    # Legacy/root fallback keeps existing bundles usable while migrating older artifacts.
    manifest = ARTIFACT_DIR / "artifact_manifest.json"
    if not manifest.exists():
        raise FileNotFoundError(f"Missing artifact manifest: {manifest}")
    manifest_sha = _sha256(manifest)
    _validate_deployment_pins("", manifest_sha)
    return _load_artifacts_for_manifest(str(ARTIFACT_DIR.resolve()), manifest_sha)


def _predict_many(
    requests: list[SnapshotRequest], bundle: ArtifactBundle | None = None
) -> list[ETAPrediction]:
    try:
        bundle = _load_artifacts() if bundle is None else bundle
        policy = _serving_policy(bundle)
    except Exception as exc:
        raise HTTPException(
            status_code=503, detail="Model artifacts are unavailable or invalid"
        ) from exc

    frame = pd.DataFrame([request.model_dump() for request in requests])
    enriched = bundle.rcot.transform(frame)
    quantiles = bundle.calibrator.transform(bundle.model.predict(enriched)).reset_index(drop=True)
    thresholds = threshold_for_rows(enriched, bundle.thresholds)
    trust = enriched["rcot_trust"].to_numpy(dtype=float)
    scores = derive_tail_scores(
        quantiles,
        thresholds,
        trust,
    ).reset_index(drop=True)
    quantile_champion = str(bundle.release_decision.get("quantile_champion", "unknown"))
    rcot_gate = str(bundle.release_decision.get("rcot_promotion_gate", "unknown"))

    predictions: list[ETAPrediction] = []
    for index in range(len(enriched)):
        support = float(enriched.iloc[index]["reference_support"])
        row_trust = float(enriched.iloc[index]["rcot_trust"])
        tail_probability = float(scores.iloc[index]["tail_probability"])
        status: Literal["AUTO", "REVIEW"] = (
            "REVIEW"
            if row_trust < policy["trust_review_threshold"]
            or support < policy["support_review_threshold"]
            or tail_probability >= policy["tail_probability_review_threshold"]
            else "AUTO"
        )
        predictions.append(
            ETAPrediction(
                q10=float(quantiles.iloc[index]["q10"]),
                q50=float(quantiles.iloc[index]["q50"]),
                q90=float(quantiles.iloc[index]["q90"]),
                tail_threshold_minutes=float(thresholds[index]),
                tail_probability=tail_probability,
                expected_excess_minutes=float(scores.iloc[index]["expected_excess_minutes"]),
                tail_risk=float(scores.iloc[index]["tail_risk"]),
                reliability_adjusted_priority=float(
                    scores.iloc[index]["reliability_adjusted_priority"]
                ),
                rcot_minutes=float(enriched.iloc[index]["rcot_minutes"]),
                reference_support=support,
                rcot_trust=row_trust,
                reliability_status=status,
                quantile_champion=quantile_champion,
                rcot_promotion_gate=cast(Literal["PROMOTE", "HOLD"], rcot_gate),
                model_version=bundle.model_version,
            )
        )
    return predictions


def _decision_priority(prediction: ETAPrediction, champion: str) -> float:
    mapping = {
        "q50_eta": prediction.q50,
        "q90_eta": prediction.q90,
        "tail_risk": prediction.tail_risk,
        "reliability_adjusted_priority": prediction.reliability_adjusted_priority,
    }
    if champion not in mapping:
        raise ValueError(f"Unsupported decision champion: {champion}")
    return float(mapping[champion])


@asynccontextmanager
async def lifespan(application: FastAPI):
    application.state.startup_error = None
    application.state.bundle_id = "unavailable"
    application.state.manifest_sha256 = "unavailable"
    application.state.model_version = "unavailable"
    try:
        bundle = _load_artifacts()
        application.state.bundle_id = bundle.bundle_id
        application.state.manifest_sha256 = bundle.manifest_sha256
        application.state.model_version = bundle.model_version
    except Exception as exc:
        application.state.startup_error = f"{type(exc).__name__}: {exc}"
        if os.getenv("REFERENCE_ETA_FAIL_STARTUP_IF_NOT_READY", "0") == "1":
            raise
    yield
    _clear_artifact_cache()


app = FastAPI(
    title="Reference-State Last-Mile ETA",
    version=__version__,
    lifespan=lifespan,
)
app.add_middleware(RequestBodyLimitMiddleware, limit_provider=_max_request_bytes)


def _set_request_metadata(
    request: Request, bundle: ArtifactBundle, *, batch_size: int, decision_policy: str | None = None
) -> None:
    request.state.bundle_id = bundle.bundle_id
    request.state.manifest_sha256 = bundle.manifest_sha256
    request.state.batch_size = int(batch_size)
    request.state.decision_policy = decision_policy
    request.state.model_version = bundle.model_version
    request.state.worker_pid = os.getpid()
    request.app.state.bundle_id = bundle.bundle_id
    request.app.state.manifest_sha256 = bundle.manifest_sha256
    request.app.state.model_version = bundle.model_version


@app.middleware("http")
async def structured_request_telemetry(request: Request, call_next):  # noqa: ANN001
    request_id = _safe_request_id(request.headers.get("x-request-id"))
    started = time.perf_counter()
    status_code = 500
    response = None
    try:
        response = await call_next(request)
        status_code = response.status_code
        return response
    finally:
        latency_ms = (time.perf_counter() - started) * 1000.0
        bundle_id = str(
            getattr(request.state, "bundle_id", getattr(app.state, "bundle_id", "unavailable"))
        )
        manifest_sha = str(
            getattr(
                request.state,
                "manifest_sha256",
                getattr(app.state, "manifest_sha256", "unavailable"),
            )
        )
        if response is not None:
            response.headers["X-Request-ID"] = request_id
            response.headers["X-Model-Version"] = str(
                getattr(
                    request.state, "model_version", getattr(app.state, "model_version", __version__)
                )
            )
            response.headers["X-Bundle-ID"] = bundle_id
            response.headers["X-Worker-PID"] = str(
                getattr(request.state, "worker_pid", os.getpid())
            )
        if os.getenv("REFERENCE_ETA_LOG_REQUESTS", "1") == "1":
            record = {
                "event": "http_request",
                "request_id": request_id,
                "method": request.method,
                "path": request.url.path,
                "status_code": status_code,
                "latency_ms": round(latency_ms, 3),
                "bundle_id": bundle_id,
                "manifest_sha256_prefix": manifest_sha[:12],
                "model_version": str(
                    getattr(
                        request.state,
                        "model_version",
                        getattr(app.state, "model_version", __version__),
                    )
                ),
                "worker_pid": int(getattr(request.state, "worker_pid", os.getpid())),
                "batch_size": getattr(request.state, "batch_size", None),
                "decision_policy": getattr(request.state, "decision_policy", None),
            }
            LOGGER.info(json.dumps(record, sort_keys=True, separators=(",", ":")))


@app.get("/health")
def health(request: Request) -> dict[str, str]:
    try:
        bundle = _load_artifacts()
        _set_request_metadata(request, bundle, batch_size=0)
        return {
            "status": "ready",
            "version": __version__,
            "model_version": bundle.model_version,
            "release_gate": str(bundle.release_decision.get("release_gate", "unknown")),
            "quantile_champion": str(bundle.release_decision.get("quantile_champion", "unknown")),
            "rcot_promotion_gate": str(
                bundle.release_decision.get("rcot_promotion_gate", "unknown")
            ),
            "decision_champion": str(bundle.release_decision.get("decision_champion", "unknown")),
            "artifact_integrity": "verified" if bundle.integrity_verified else "unverified",
        }
    except Exception:
        return {
            "status": "not_ready",
            "version": __version__,
            "model_version": "unavailable",
            "release_gate": "unavailable",
            "quantile_champion": "unavailable",
            "rcot_promotion_gate": "unavailable",
            "decision_champion": "unavailable",
            "artifact_integrity": "unavailable",
        }


@app.get("/live")
def live(request: Request) -> dict[str, str]:
    request.state.worker_pid = os.getpid()
    return {"status": "alive", "version": __version__}


@app.get("/ready")
def ready(request: Request) -> dict[str, str]:
    try:
        bundle = _load_artifacts()
        _set_request_metadata(request, bundle, batch_size=0)
    except Exception as exc:
        raise HTTPException(status_code=503, detail="Model artifacts are not ready") from exc
    return {
        "status": "ready",
        "version": __version__,
        "model_version": bundle.model_version,
        "release_gate": str(bundle.release_decision.get("release_gate", "unknown")),
        "artifact_integrity": "verified" if bundle.integrity_verified else "unverified",
    }


@app.post("/v1/eta/predict", response_model=ETAPrediction)
def predict_eta(payload: SnapshotRequest, request: Request) -> ETAPrediction:
    bundle = _load_artifacts()
    _set_request_metadata(request, bundle, batch_size=1)
    return _predict_many([payload], bundle=bundle)[0]


@app.post("/v1/eta/batch", response_model=BatchPredictionResponse)
def predict_eta_batch(payload: BatchPredictionRequest, request: Request) -> BatchPredictionResponse:
    _enforce_batch_limit(len(payload.snapshots))
    bundle = _load_artifacts()
    _set_request_metadata(request, bundle, batch_size=len(payload.snapshots))
    return BatchPredictionResponse(predictions=_predict_many(payload.snapshots, bundle=bundle))


@app.post("/v1/triage/rank", response_model=TriageResponse)
def rank_triage(payload: TriageRequest, request: Request) -> TriageResponse:
    _enforce_batch_limit(len(payload.snapshots))
    try:
        bundle = _load_artifacts()
        champion = str(bundle.release_decision.get("decision_champion", ""))
        _set_request_metadata(
            request, bundle, batch_size=len(payload.snapshots), decision_policy=champion
        )
        predictions = _predict_many(payload.snapshots, bundle=bundle)
        priorities = np.asarray(
            [_decision_priority(prediction, champion) for prediction in predictions], dtype=float
        )
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=503, detail="Decision policy is unavailable or invalid"
        ) from exc
    selected_count = max(1, int(np.ceil(len(predictions) * payload.capacity)))
    order = np.argsort(-priorities, kind="stable")[:selected_count]
    selected = [
        TriageSelection(
            request_index=int(index),
            priority_rank=rank,
            prediction=predictions[int(index)],
        )
        for rank, index in enumerate(order, start=1)
    ]
    return TriageResponse(
        capacity=payload.capacity,
        selected_count=selected_count,
        decision_policy=champion,
        selected=selected,
    )
