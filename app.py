from http.server import BaseHTTPRequestHandler


class handler(BaseHTTPRequestHandler):
    """HTTP entrypoint recognized by Vercel's Python runtime."""

    def do_GET(self):
        body = (
            "<!doctype html>"
            '<html lang="en">'
            "<head><meta charset=\"utf-8\"><title>Product Defect Detector</title></head>"
            "<body>"
            "<h1>Product Defect Detector</h1>"
            "<p>This project is a Streamlit application. Run "
            "<code>streamlit run streamlit_app.py</code> to use the detector.</p>"
            "</body>"
            "</html>"
        ).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format, *args):
        return


app = handler
application = handler


if __name__ == "__main__":
    from http.server import HTTPServer

    HTTPServer(("127.0.0.1", 8000), handler).serve_forever()
