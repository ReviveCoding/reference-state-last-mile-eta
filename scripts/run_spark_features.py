from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        from pyspark.sql import SparkSession
        from pyspark.sql import functions as F
    except ImportError as exc:
        raise SystemExit("Install the optional dependency with: pip install -e '.[spark]'") from exc

    spark = SparkSession.builder.appName("reference-eta-features").getOrCreate()
    frame = spark.read.option("header", True).option("inferSchema", True).csv(str(args.input))
    enriched = (
        frame.withColumn("route_phase", F.col("completed_task_count") / F.col("initial_task_count"))
        .withColumn(
            "remaining_workload_ratio", F.col("remaining_workload") / F.col("initial_workload")
        )
        .repartition("city", "work_date")
    )
    enriched.write.mode("overwrite").partitionBy("city", "work_date").parquet(str(args.output))
    spark.stop()


if __name__ == "__main__":
    main()
