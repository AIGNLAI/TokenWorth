#!/usr/bin/env bash

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${PROJECT_ROOT}/.venv"
PYTHON_BIN="${PYTHON_BIN:-python3.9}"

usage() {
    cat <<'EOF'
Usage: bash scripts/install.sh [options]

Options:
  --venv PATH       Virtual environment path (default: .venv)
  -h, --help        Show this help message

Environment:
  PYTHON_BIN        Python 3.9 executable (default: python3.9)
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --venv)
            [[ $# -ge 2 ]] || { echo "--venv requires a path" >&2; exit 2; }
            VENV_DIR="$2"
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
    echo "Install Python 3.9 or set PYTHON_BIN to its path." >&2
    exit 1
fi

PYTHON_VERSION="$("${PYTHON_BIN}" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
if [[ "${PYTHON_VERSION}" != "3.9" ]]; then
    echo "ALFWorld requires Python 3.9; found ${PYTHON_VERSION}." >&2
    exit 1
fi

cd "${PROJECT_ROOT}"
if [[ ! -x "${VENV_DIR}/bin/python" ]]; then
    "${PYTHON_BIN}" -m venv "${VENV_DIR}"
fi
VENV_PYTHON="${VENV_DIR}/bin/python"
export PATH="${VENV_DIR}/bin:${PATH}"

"${VENV_PYTHON}" -m pip install --upgrade pip setuptools wheel
"${VENV_PYTHON}" -m pip install --constraint requirements.lock -e "${PROJECT_ROOT}/../tokenworth_core" -e .

PYTHONDONTWRITEBYTECODE=1 "${VENV_PYTHON}" -c \
    'from alfworld.agents.environment import get_environment; from tokenworth_core import TokenWorthCore; assert get_environment("AlfredTWEnv"); TokenWorthCore()'
PYTHONDONTWRITEBYTECODE=1 "${VENV_PYTHON}" scripts/run_suite.py --dry-run >/dev/null

cat <<EOF

Installation complete.
Activate the environment with:
  source ${VENV_DIR}/bin/activate
EOF
