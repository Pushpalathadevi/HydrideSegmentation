"""Launch the intranet segmentation web server.

Serves the browser application on Windows and Linux with the same command. The
production server is `waitress`, a pure-Python WSGI server that needs no
compiler and no platform-specific build, which makes it suitable for air-gapped
hosts. When waitress is unavailable the script falls back to the Flask
development server and says so clearly.

Examples
--------
Serve to the intranet on the configured port::

    python scripts/run_web_server.py

Serve on a specific port and bind only to this machine::

    python scripts/run_web_server.py --host 127.0.0.1 --port 8080
"""

from __future__ import annotations

import argparse
import logging
import socket
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hydride_segmentation.web import create_app, load_web_config  # noqa: E402


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Serve the segmentation web app on the intranet")
    parser.add_argument("--config", type=str, default="", help="Web server YAML config path")
    parser.add_argument("--host", type=str, default="", help="Bind address override, for example 0.0.0.0")
    parser.add_argument("--port", type=int, default=0, help="Port override")
    parser.add_argument("--threads", type=int, default=0, help="Waitress worker thread override")
    parser.add_argument(
        "--preload",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Warm trained models at startup (defaults to the configured value)",
    )
    parser.add_argument(
        "--dev",
        action="store_true",
        help="Use the Flask development server with auto-reload instead of waitress",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity",
    )
    parser.add_argument(
        "--print-urls",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Print the URLs colleagues should open",
    )
    return parser


def _local_addresses() -> list[str]:
    """Return intranet addresses this host is likely reachable on."""

    addresses: list[str] = []
    try:
        hostname = socket.gethostname()
        addresses.append(hostname)
        for info in socket.getaddrinfo(hostname, None, socket.AF_INET):
            address = info[4][0]
            if address not in addresses and not address.startswith("127."):
                addresses.append(address)
    except Exception:
        pass
    return addresses


def _announce(host: str, port: int) -> None:
    print("")
    print("  Segmentation web app is starting.")
    if host in {"0.0.0.0", "::"}:
        print(f"    On this machine:      http://localhost:{port}/")
        for address in _local_addresses():
            print(f"    On the intranet:      http://{address}:{port}/")
        print("")
        print("  Share one of the intranet links with colleagues on the same network.")
        print("  If they cannot connect, allow inbound TCP on this port in the host firewall.")
    else:
        print(f"    URL:                  http://{host}:{port}/")
        print("  Bound to a specific address, so only that interface can reach it.")
    print("")
    print("  Press Ctrl+C to stop.")
    print("")


def main() -> int:
    """Run the web server and return a process exit code."""

    args = _build_parser().parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    config = load_web_config(args.config or None)
    host = args.host.strip() or config.host
    port = int(args.port or config.port)
    threads = int(args.threads or config.threads)

    for warning in config.warnings:
        logging.getLogger(__name__).warning("Configuration: %s", warning)

    app = create_app(config=config, preload=args.preload)

    if args.print_urls:
        _announce(host, port)

    if args.dev:
        print("  Running the Flask development server. Do not use this for shared deployments.\n")
        app.run(host=host, port=port, debug=True, use_reloader=True)
        return 0

    try:
        from waitress import serve
    except ImportError:
        print(
            "  waitress is not installed, so the development server is being used instead.\n"
            "  This is fine for a quick trial but not for a shared deployment.\n"
            "  Install it with:  pip install waitress\n",
            file=sys.stderr,
        )
        app.run(host=host, port=port, debug=False, use_reloader=False)
        return 0

    serve(
        app,
        host=host,
        port=port,
        threads=threads,
        channel_timeout=int(config.request_timeout_seconds),
        ident="MicroSeg",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
