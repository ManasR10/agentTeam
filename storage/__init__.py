"""Durable storage for Phase 4 workflow runs and audit events.

A `RunStore` (see `base.py`) is the interface the workflow service depends on;
`sqlite_store.py` is the only implementation for now. SQLite is enough for a
single CLI process and gives transactions, querying, and ordering without an
external server.
"""
