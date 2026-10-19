#!/usr/bin/env bash

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${PROJECT_ROOT}/.venv"
PYTHON_BIN="${PYTHON_BIN:-python3.10}"
EXTRAS=""
UPSTREAM_COMMIT="09dd993f46c3fbf3a799465bb96d524edcb0b199"

usage() {
    cat <<'EOF'
Usage: bash scripts/install.sh [options]

Options:
  --venv PATH       Virtual environment path (default: .venv)
  --extras LIST     Comma-separated project extras, for example: test
  -h, --help        Show this help message

Environment:
  PYTHON_BIN        Python 3.10 executable (default: python3.10)
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
    echo "Install Python 3.10 or set PYTHON_BIN to its path." >&2
    exit 1
fi

PYTHON_VERSION="$(${PYTHON_BIN} -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
if [[ "${PYTHON_VERSION}" != "3.10" ]]; then
    echo "BigCodeBench requires Python 3.10; found ${PYTHON_VERSION}." >&2
    exit 1
fi

cd "${PROJECT_ROOT}"
if [[ ! -x "${VENV_DIR}/bin/python" ]]; then
    "${PYTHON_BIN}" -m venv "${VENV_DIR}"
fi
VENV_PYTHON="${VENV_DIR}/bin/python"
export PATH="${VENV_DIR}/bin:${PATH}"
UPSTREAM_DIR="${VENV_DIR}/src/bigcodebench"

"${VENV_PYTHON}" -m pip install --upgrade pip setuptools wheel

if ! command -v git >/dev/null 2>&1; then
    echo "Git is required to fetch the pinned BigCodeBench source." >&2
    exit 1
fi
if [[ -e "${UPSTREAM_DIR}" && ! -d "${UPSTREAM_DIR}/.git" ]]; then
    echo "Upstream path exists but is not a Git checkout: ${UPSTREAM_DIR}" >&2
    exit 1
fi
if [[ ! -d "${UPSTREAM_DIR}/.git" ]]; then
    mkdir -p "${UPSTREAM_DIR}"
    git -C "${UPSTREAM_DIR}" init
    git -C "${UPSTREAM_DIR}" remote add origin \
        https://github.com/bigcode-project/bigcodebench.git
fi
for attempt in 1 2 3; do
    if git -C "${UPSTREAM_DIR}" fetch --depth 1 origin "${UPSTREAM_COMMIT}"; then
        break
    fi
    if [[ "${attempt}" -eq 3 ]]; then
        echo "Failed to fetch BigCodeBench after ${attempt} attempts." >&2
        exit 1
    fi
    sleep "${attempt}"
done
git -C "${UPSTREAM_DIR}" checkout --detach "${UPSTREAM_COMMIT}"

"${VENV_PYTHON}" -c \
    'import site, sys; from pathlib import Path; root = Path(sys.argv[1]).resolve(); Path(site.getsitepackages()[0], "tokenworth_bigcodebench_upstream.pth").write_text(str(root) + "\n", encoding="utf-8")' \
    "${UPSTREAM_DIR}"

if [[ -n "${EXTRAS}" ]]; then
    "${VENV_PYTHON}" -m pip install --constraint requirements.lock -e "${PROJECT_ROOT}/../tokenworth_core" -e ".[${EXTRAS}]"
else
    "${VENV_PYTHON}" -m pip install --constraint requirements.lock -e "${PROJECT_ROOT}/../tokenworth_core" -e .
fi

PYTHONDONTWRITEBYTECODE=1 "${VENV_PYTHON}" -c \
    'from bigcodebench.data import get_bigcodebench; from bigcodebench.eval import untrusted_check; from bigcodebench.sanitize import sanitize'
PYTHONDONTWRITEBYTECODE=1 "${VENV_PYTHON}" scripts/run_suite.py local --dry-run >/dev/null

cat <<EOF

Installation complete.
Activate the environment with:
  source ${VENV_DIR}/bin/activate
EOF
