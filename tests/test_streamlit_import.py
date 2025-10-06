import pytest

@pytest.mark.xfail(reason=(
    "streamlit_app.py avvia UI e carica il modello a import-time. "
    "Per testarlo in CI senza avviare Streamlit, metti un guard (es. STREAMLIT_TEST=1) "
    "che salti UI/load; poi toglieremo xfail e testeremo le funzioni."
))
def test_streamlit_import_safe():
    assert True