# SageMaker execution path

This directory is an execution scaffold, not evidence of an executed cloud job. Configure AWS
credentials and an existing S3 bucket, upload the frozen data manifest, then run `run_job.py`.
The entry point records hyperparameters, input/output URIs, image/framework version, and artifact
hashes. Do not claim an AWS run until the returned job metadata and S3 artifacts are preserved.
