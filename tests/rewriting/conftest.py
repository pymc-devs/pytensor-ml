import logging

import pytest


@pytest.fixture(autouse=True)
def fail_on_swallowed_rewrite_errors(caplog):
    """Fail if a node rewriter raised while a test was rewriting a graph.

    Pytensor catches exceptions from a node rewriter, reports them through ``logger.error``, and leaves
    the graph untouched. Nothing else in the suite notices: ``filterwarnings = ["error"]`` only sees
    warnings, and a test asserting that a rewrite did *not* fire passes either way -- whether the
    rewrite correctly declined to match or crashed on the first node it touched.
    """
    yield

    failures = [
        record.getMessage()
        for record in caplog.get_records("call")
        if record.name.startswith("pytensor.graph.rewriting") and record.levelno >= logging.ERROR
    ]

    assert not failures, "a node rewriter raised and pytensor swallowed it:\n" + "\n".join(failures)
