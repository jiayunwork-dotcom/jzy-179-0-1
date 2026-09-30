"""pytest 公共夹具：临时数据库 + 关闭线程池的 FastAPI TestClient。"""
from __future__ import annotations

import os
import sys

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.main import create_app  # noqa: E402


@pytest.fixture()
def client(tmp_path):
    db = tmp_path / "test.db"
    app = create_app(db_path=str(db))
    # with 块触发 lifespan：进入时复位残留作业，退出时关停线程池与数据库
    with TestClient(app) as c:
        yield c
