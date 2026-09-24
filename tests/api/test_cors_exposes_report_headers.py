"""The screens run on another origin than this API, so a browser hides every response header
the CORS policy does not expose. The report editor's preview says what would stop an edit
saving in X-Report-Blocking; unexposed, the editor would read "nothing" and show a clean page."""
from fastapi.middleware.cors import CORSMiddleware


def test_the_report_headers_are_exposed_to_the_screens():
    from api.main import app

    cors = [m for m in app.user_middleware if m.cls is CORSMiddleware]
    assert len(cors) == 1
    exposed = {h.lower() for h in cors[0].kwargs.get("expose_headers", [])}
    assert {"x-report-blocking", "x-report-run-id"} <= exposed
