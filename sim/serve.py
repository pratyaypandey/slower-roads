#!/usr/bin/env python3
"""Static server for the sim demo with caching disabled, so edits to the ES modules
are always picked up on reload (no stale module cache). Run: python3 serve.py [port]"""
import sys
from http.server import HTTPServer, SimpleHTTPRequestHandler


class NoCacheHandler(SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header("Cache-Control", "no-store, no-cache, must-revalidate, max-age=0")
        self.send_header("Pragma", "no-cache")
        self.send_header("Expires", "0")
        super().end_headers()


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8777
    HTTPServer(("", port), NoCacheHandler).serve_forever()
