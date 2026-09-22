#!/usr/bin/env python3
"""Serve the static demo for local development."""

from __future__ import annotations

import argparse
import os
import re
import shutil
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


RANGE_PATTERN = re.compile(r"bytes=(\d*)-(\d*)$")


class RangeRequestHandler(SimpleHTTPRequestHandler):
    range_start: int | None = None
    range_end: int | None = None

    def send_head(self):
        self.range_start = None
        self.range_end = None
        path = Path(self.translate_path(self.path))
        range_header = self.headers.get("Range")

        if not range_header or not path.is_file():
            return super().send_head()

        match = RANGE_PATTERN.fullmatch(range_header.strip())
        if match is None:
            self.send_error(416, "Invalid byte range")
            return None

        file_size = path.stat().st_size
        start_text, end_text = match.groups()
        if start_text:
            start = int(start_text)
            end = int(end_text) if end_text else file_size - 1
        elif end_text:
            suffix_length = int(end_text)
            start = max(0, file_size - suffix_length)
            end = file_size - 1
        else:
            self.send_error(416, "Empty byte range")
            return None

        if start >= file_size or start > end:
            self.send_response(416)
            self.send_header("Content-Range", f"bytes */{file_size}")
            self.end_headers()
            return None

        end = min(end, file_size - 1)
        self.range_start = start
        self.range_end = end
        file_handle = path.open("rb")
        self.send_response(206)
        self.send_header("Content-Type", self.guess_type(str(path)))
        self.send_header("Content-Range", f"bytes {start}-{end}/{file_size}")
        self.send_header("Content-Length", str(end - start + 1))
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Last-Modified", self.date_time_string(path.stat().st_mtime))
        self.end_headers()
        file_handle.seek(start)
        return file_handle

    def end_headers(self) -> None:
        if self.range_start is None:
            self.send_header("Accept-Ranges", "bytes")
        super().end_headers()

    def copyfile(self, source, outputfile) -> None:
        if self.range_start is None or self.range_end is None:
            shutil.copyfileobj(source, outputfile)
            return

        remaining = self.range_end - self.range_start + 1
        while remaining > 0:
            chunk = source.read(min(64 * 1024, remaining))
            if not chunk:
                break
            outputfile.write(chunk)
            remaining -= len(chunk)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8765)
    arguments = parser.parse_args()

    demo_directory = Path(__file__).resolve().parent
    os.chdir(demo_directory)
    server = ThreadingHTTPServer(
        (arguments.host, arguments.port), RangeRequestHandler
    )
    print(
        f"Serving {demo_directory} at "
        f"http://{arguments.host}:{arguments.port}",
        flush=True,
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
