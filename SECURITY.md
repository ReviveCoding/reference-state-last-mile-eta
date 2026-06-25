# Security notes

## Trusted artifacts only

Serving verifies immutable-bundle size and SHA-256 records before loading model files. This detects
corruption and bundle mismatch, but a checksum alone is not an identity signature. Python `joblib`
artifacts can execute code during deserialization, so only bundles produced by a trusted release
workflow should be served.

The API also checks the recorded training versions of scikit-learn, LightGBM, and joblib against the
serving runtime. A mismatch is rejected unless the explicit diagnostic override
`REFERENCE_ETA_ALLOW_VERSION_MISMATCH=1` is set.

For deployment pinning, set both:

```text
REFERENCE_ETA_EXPECTED_BUNDLE_ID
REFERENCE_ETA_EXPECTED_MANIFEST_SHA256
```

This prevents a valid but unintended local bundle from being selected.

## Input and resource controls

Request models use strict types, reject extra fields and non-finite values, and enforce internal
snapshot consistency. Batch size is capped by both the schema and `REFERENCE_ETA_MAX_BATCH_SIZE`. Raw request bytes are bounded during ASGI streaming, so chunked transfer without `Content-Length` cannot bypass `REFERENCE_ETA_MAX_REQUEST_BYTES`.
`/live` reports process liveness, while `/ready` requires a valid released bundle. Client request IDs must match a bounded safe-character contract; invalid values are replaced rather than logged or echoed.

## Supply-chain evidence

Local releases produce reproducible distributions, a CycloneDX runtime SBOM, run provenance, and
content manifests. `.github/workflows/release-attest.yml` is configured to request GitHub artifact
attestations on version tags. The bundled local report does not claim that a remote attestation was
executed.

## External data and secrets

Do not commit delivery, customer, courier, precise location, credential, cloud-token, or proprietary
operational data. The generated samples contain no real customer records. Keep cloud credentials in
the platform secret store, never in configs or reports.

## Reporting

Report suspected vulnerabilities privately to the repository owner before public disclosure. Do not
attach sensitive data or executable model bundles to public issues.
