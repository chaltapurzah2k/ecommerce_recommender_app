import shutil
import socket
import subprocess
import sys
from pathlib import Path


def find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        sock.listen(1)
        return int(sock.getsockname()[1])


def build_streamlit_command(app_path: Path, port: int, extra_args: list[str]) -> list[str]:
    streamlit_executable = shutil.which("streamlit")
    if streamlit_executable:
        return [
            streamlit_executable,
            "run",
            str(app_path),
            "--server.port",
            str(port),
            *extra_args,
        ]

    return [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(app_path),
        "--server.port",
        str(port),
        *extra_args,
    ]


def main() -> int:
    base_dir = Path(__file__).resolve().parent
    app_name = sys.argv[1] if len(sys.argv) > 1 and not sys.argv[1].startswith("-") else "streamlit_app.py"
    extra_args = sys.argv[2:] if app_name != "streamlit_app.py" else sys.argv[1:]

    app_path = (base_dir / app_name).resolve()
    if not app_path.exists():
        print(f"Streamlit app not found: {app_path}", file=sys.stderr)
        return 1

    port = find_free_port()
    command = build_streamlit_command(app_path, port, extra_args)

    print(f"Starting {app_path.name} on port {port}")
    print(f"URL: http://localhost:{port}")
    return subprocess.call(command, cwd=str(base_dir))


if __name__ == "__main__":
    raise SystemExit(main())