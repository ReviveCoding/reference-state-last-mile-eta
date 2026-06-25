# LaDe adapter contract

Normalize official LaDe delivery records to:

| Column | Meaning |
|---|---|
| courier_id | Stable courier identifier within the released data |
| city | City/scenario identifier |
| accept_time | Task-accept event time |
| delivery_time | Task-finish event time |
| latitude | Released task/event latitude |
| longitude | Released task/event longitude |

Optional static columns can include AOI, package attributes, and time requirements. Snapshot
construction must never use task-finish events after the query time as input features. Future finish
times may only be used to build labels.
