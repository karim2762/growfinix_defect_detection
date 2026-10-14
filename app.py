def app(environ, start_response):
    """Minimal WSGI entrypoint for Vercel's Python runtime.

    The detector UI is a Streamlit application and should be started with
    `streamlit run streamlit_app.py`; Vercel cannot host Streamlit's server.
    """
    body = """<!doctype html>
<html lang="en">
  <head><meta charset="utf-8"><title>Product Defect Detector</title></head>
  <body>
    <h1>Product Defect Detector</h1>
    <p>This project is a Streamlit application. Run <code>streamlit run streamlit_app.py</code> to use the detector.</p>
  </body>
</html>""".encode("utf-8")
    start_response(
        "200 OK",
        [
            ("Content-Type", "text/html; charset=utf-8"),
            ("Content-Length", str(len(body))),
        ],
    )
    return [body]


application = app
handler = app


if __name__ == "__main__":
    from wsgiref.simple_server import make_server

    with make_server("127.0.0.1", 8000, app) as server:
        server.serve_forever()
