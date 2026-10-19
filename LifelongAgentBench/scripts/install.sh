#!/usr/bin/env bash

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${PROJECT_ROOT}/.venv"
PYTHON_BIN="${PYTHON_BIN:-python3.11}"
EXTRAS=""
CONSTRAINTS="${PROJECT_ROOT}/requirements-lock.txt"

usage() {
    cat <<'EOF'
Usage: bash scripts/install.sh [options]

Options:
  --venv PATH       Virtual environment path (default: .venv)
  --extras LIST     Comma-separated extras: dev
  --constraints PATH  Dependency constraints (default: requirements-lock.txt)
  -h, --help        Show this help message

Environment:
  PYTHON_BIN        Python 3.11 executable (default: python3.11)
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --venv)
            [[ $# -ge 2 ]] || { echo "--venv requires a path" >&2; exit 2; }
            VENV_DIR="$2"
            shift 2
            ;;
        --extras)
            [[ $# -ge 2 ]] || { echo "--extras requires a list" >&2; exit 2; }
            EXTRAS="$2"
            shift 2
            ;;
        --constraints)
            [[ $# -ge 2 ]] || { echo "--constraints requires a path" >&2; exit 2; }
            CONSTRAINTS="$2"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown option: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
done

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
    echo "Python executable not found: ${PYTHON_BIN}" >&2
    echo "Install Python 3.11 or set PYTHON_BIN to its path." >&2
    exit 1
fi

PYTHON_VERSION="$("${PYTHON_BIN}" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
if [[ "${PYTHON_VERSION}" != "3.11" ]]; then
    echo "LifelongAgentBench requires Python 3.11; found ${PYTHON_VERSION}." >&2
    exit 1
fi

cd "${PROJECT_ROOT}"
"${PYTHON_BIN}" -m venv "${VENV_DIR}"
VENV_PYTHON="${VENV_DIR}/bin/python"

"${VENV_PYTHON}" -m pip install --upgrade pip setuptools wheel

if [[ -n "${EXTRAS}" ]]; then
    "${VENV_PYTHON}" -m pip install --constraint "${CONSTRAINTS}" -e "${PROJECT_ROOT}/../tokenworth_core" -e ".[${EXTRAS}]"
else
    "${VENV_PYTHON}" -m pip install --constraint "${CONSTRAINTS}" -e "${PROJECT_ROOT}/../tokenworth_core" -e .
fi

PYTHONDONTWRITEBYTECODE=1 "${VENV_PYTHON}" run.py --help >/dev/null
PYTHONDONTWRITEBYTECODE=1 "${VENV_PYTHON}" scripts/verify_data.py

cat <<EOF

Installation complete.
Activate the environment with:
  source ${VENV_DIR}/bin/activate
EOF
