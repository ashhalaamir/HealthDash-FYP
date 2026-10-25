"""Shared test setup.

The `engine` fixture runs every test that uses it on SQLite and, when TEST_DATABASE_URL is set, on PostgreSQL too.
Each PostgreSQL test gets its own throwaway schema that is dropped afterwards, so the tests can never touch the
tables in the database you actually use.
"""
import os
import uuid

import pytest
from sqlalchemy import create_engine, text

from src.db import get_engine

PG_URL = os.environ.get("TEST_DATABASE_URL")


@pytest.fixture(params=["sqlite"] + (["postgres"] if PG_URL else []))
def engine(request, tmp_path):
    if request.param == "sqlite":
        eng = get_engine(f"sqlite:///{(tmp_path / 't.db').as_posix()}")
        yield eng
        eng.dispose()
        return
    admin, schema = create_engine(PG_URL), f"pytest_{uuid.uuid4().hex[:10]}"
    with admin.begin() as c:
        c.execute(text(f"CREATE SCHEMA {schema}"))
    eng = get_engine(PG_URL, connect_args={"options": f"-csearch_path={schema}"})
    yield eng
    eng.dispose()
    with admin.begin() as c:
        c.execute(text(f"DROP SCHEMA {schema} CASCADE"))
    admin.dispose()