"""Project-local server entry point: constrain the API socket before it binds.

The installed vLLM version pre-binds ("", port) regardless of --host. The
socket subclass below lives only in this child process; no installed library,
driver or host configuration is changed. Other ports are not rewritten.
This wrapper is CPU-tested, not a new CUDA/vLLM hardware validation.
"""
from __future__ import annotations

import argparse
import contextlib
import runpy
import socket
import sys


@contextlib.contextmanager
def loopback_api_socket(port: int):
    if type(port) is not int or not 1 <= port <= 65535:
        raise ValueError("invalid API port")
    original = socket.socket

    class ApiSocket(original):
        def bind(self, address):
            if self.family in (socket.AF_INET, socket.AF_INET6) and address[1] == port:
                host = address[0]
                if host not in ("", "0.0.0.0", "::", "127.0.0.1", "::1"):
                    raise ValueError("non-loopback API bind refused")
                if self.family == socket.AF_INET6:
                    address = ("::1", port, *address[2:])
                else:
                    address = ("127.0.0.1", port)
            return super().bind(address)

    socket.socket = ApiSocket
    try:
        yield
    finally:
        socket.socket = original


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--host", required=True, choices=["127.0.0.1"])
    parser.add_argument("--port", required=True, type=int)
    args, _ = parser.parse_known_args(argv)
    sys.argv = ["vllm.entrypoints.openai.api_server", *argv]
    with loopback_api_socket(args.port):
        runpy.run_module("vllm.entrypoints.openai.api_server", run_name="__main__")


if __name__ == "__main__":
    main()
