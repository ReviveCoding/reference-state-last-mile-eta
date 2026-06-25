from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor

import reference_eta.serving.api as api


def test_concurrent_cache_miss_loads_immutable_bundle_once(monkeypatch) -> None:
    api._clear_artifact_cache()
    calls = 0
    calls_lock = threading.Lock()
    sentinel = object()

    def fake_loader(bundle_dir: str, manifest_sha: str):  # noqa: ANN202
        nonlocal calls
        with calls_lock:
            calls += 1
        time.sleep(0.05)
        assert bundle_dir == "/bundle"
        assert manifest_sha == "a" * 64
        return sentinel

    monkeypatch.setattr(api, "_load_artifacts_uncached", fake_loader)
    with ThreadPoolExecutor(max_workers=8) as executor:
        results = list(
            executor.map(
                lambda _: api._load_artifacts_for_manifest("/bundle", "a" * 64),
                range(16),
            )
        )
    assert calls == 1
    assert all(result is sentinel for result in results)


def test_new_manifest_and_cache_clear_trigger_exactly_one_new_load(monkeypatch) -> None:
    api._clear_artifact_cache()
    calls: list[tuple[str, str]] = []

    def fake_loader(bundle_dir: str, manifest_sha: str):  # noqa: ANN202
        calls.append((bundle_dir, manifest_sha))
        return object()

    monkeypatch.setattr(api, "_load_artifacts_uncached", fake_loader)
    first = api._load_artifacts_for_manifest("/bundle", "a" * 64)
    assert api._load_artifacts_for_manifest("/bundle", "a" * 64) is first
    api._load_artifacts_for_manifest("/bundle", "b" * 64)
    api._clear_artifact_cache()
    api._load_artifacts_for_manifest("/bundle", "b" * 64)
    assert calls == [
        ("/bundle", "a" * 64),
        ("/bundle", "b" * 64),
        ("/bundle", "b" * 64),
    ]
