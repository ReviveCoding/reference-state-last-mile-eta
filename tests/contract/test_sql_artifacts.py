from pathlib import Path


def test_substantive_sql_artifacts_exist() -> None:
    root = Path(__file__).resolve().parents[2]
    sql_files = sorted((root / "sql").glob("*.sql"))
    assert len(sql_files) >= 5
    assert all("CREATE TABLE" in path.read_text(encoding="utf-8").upper() for path in sql_files)
