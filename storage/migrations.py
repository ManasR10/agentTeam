from __future__ import annotations

import sqlite3
from pathlib import Path

# The schema lives next to this module as plain SQL so it is easy to read and
# diff. There is only one version today; when the shape changes, add migration
# steps here keyed off a stored version rather than editing schema.sql in place.

SCHEMA_PATH = Path(__file__).resolve().parent / "schema.sql"


def apply_schema(connection: sqlite3.Connection) -> None:
    """Create the workflow tables if they do not already exist."""
    schema_sql = SCHEMA_PATH.read_text(encoding="utf-8")
    with connection:
        connection.executescript(schema_sql)
