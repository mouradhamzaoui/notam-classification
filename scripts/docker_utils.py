"""
scripts/docker_utils.py
Commandes utilitaires pour gérer les containers Docker.

Usage :
    uv run python scripts/docker_utils.py build
    uv run python scripts/docker_utils.py up
    uv run python scripts/docker_utils.py down
    uv run python scripts/docker_utils.py logs
    uv run python scripts/docker_utils.py status
"""

import sys
import subprocess
import argparse
from pathlib import Path


def run(cmd: str, check: bool = True):
    print(f"\n▶ {cmd}\n")
    result = subprocess.run(cmd, shell=True, check=check)
    return result.returncode


def main():
    parser = argparse.ArgumentParser(description="Docker utilities for NOTAM system")
    parser.add_argument("command", choices=["build","up","down","logs","status","restart","clean"])
    parser.add_argument("--dev", action="store_true", help="Mode développement")
    args = parser.parse_args()

    compose_files = "-f docker-compose.yml"
    if args.dev:
        compose_files += " -f docker-compose.dev.yml"

    commands = {
        "build":   f"docker compose {compose_files} build --no-cache",
        "up":      f"docker compose {compose_files} up -d",
        "down":    f"docker compose {compose_files} down",
        "logs":    f"docker compose {compose_files} logs -f --tail=100",
        "status":  f"docker compose {compose_files} ps",
        "restart": f"docker compose {compose_files} restart",
        "clean":   (
            f"docker compose {compose_files} down -v --remove-orphans && "
            "docker system prune -f"
        ),
    }

    run(commands[args.command], check=False)


if __name__ == "__main__":
    main()