"""`temper-engine` entry point, started by the desktop shell or by hand for development."""

import argparse
import os
import sys
import threading
import time

import psutil
import uvicorn


def _exit_with_parent(pid: int) -> None:
    # The shell may be killed without a chance to stop us; never outlive it.
    while psutil.pid_exists(pid):
        time.sleep(2)
    os._exit(0)


# The `temper` subcommands answered by cli.py.
CLI = ("start", "mcp", "models", "load", "ps", "unload", "status", "version", "get", "import", "log", "daemon", "chat", "train", "runs", "export", "serve")


def main() -> None:
    if sys.argv[1:2] in ([c] for c in CLI) or sys.argv[1:3] == ["doctor", "--mcp"]:
        from .cli import main as cli

        sys.exit(cli(sys.argv[1:]))
    if sys.argv[1:2] == ["doctor"]:
        from .doctor import report

        sys.exit(report())

    parser = argparse.ArgumentParser(prog="temper-engine")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--token", default=os.environ.get("TEMPER_TOKEN", ""), help="bearer token; empty disables auth")
    parser.add_argument("--parent-pid", type=int)
    args = parser.parse_args()

    if args.parent_pid:
        threading.Thread(target=_exit_with_parent, args=(args.parent_pid,), daemon=True).start()

    run_engine(args.host, args.port, args.token)


def run_engine(host: str, port: int, token: str) -> None:
    """Serves the engine until stopped: for the desktop shell, and for `temper serve`."""
    # The doctor reports when the shell had to fall back from the usual port.
    os.environ["TEMPER_PORT"] = str(port)
    from .app import create_app
    from .cli import clear_running, write_running

    # Leaves the address and key where `temper start` and `temper mcp add` find them.
    write_running(f"http://127.0.0.1:{port}", token)
    try:
        uvicorn.run(create_app(token=token), host=host, port=port, log_level="warning")
    finally:
        clear_running()


if __name__ == "__main__":
    main()
