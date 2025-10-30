#!/usr/bin/env bash
# Setup script for tab_pfn_benchmarking GPU experiments.
# Recreates the Python environment, downloads Syrupy, and leaves the repo ready
# to run ./run.sh on a CUDA-capable host.

set -euo pipefail
IFS=$'\n\t'

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${REPO_ROOT}"

VENV_DIR="tab_pfn"
SYRUPY_VERSION="1.4"
SYRUPY_ARCHIVE_URL="https://github.com/jeetsukumaran/Syrupy/archive/refs/tags/v${SYRUPY_VERSION}.tar.gz"
SYRUPY_SRC_DIR="${REPO_ROOT}/Syrupy-${SYRUPY_VERSION}"
PYTHON_BIN="${REPO_ROOT}/${VENV_DIR}/bin/python"

log() {
  printf '[setup] %s\n' "$*"
}

require_cmd() {
  if ! command -v "$1" >/dev/null 2>&1; then
    log "Missing required command: $1"
    exit 1
  fi
}

log "Verifying base tooling (python3, wget, tar)"
require_cmd python3
require_cmd wget
require_cmd tar

# Ensure python3 -m venv works (adds ensurepip on some bare Ubuntu installs).
if command -v apt-get >/dev/null 2>&1; then
  log "Ensuring system packages for Python virtualenv support are installed"
  sudo apt-get update
  sudo apt-get install -y python3.10-venv python3-pip-whl python3-setuptools-whl
else
  log "apt-get not found; ensure python3 -m venv is available manually"
fi

if [[ ! -d "${VENV_DIR}" ]]; then
  log "Creating Python virtual environment in ${VENV_DIR}"
  python3 -m venv "${VENV_DIR}"
else
  log "Virtual environment ${VENV_DIR} already exists; reusing it"
fi

log "Upgrading pip inside the virtual environment"
"${PYTHON_BIN}" -m pip install --upgrade pip

log "Installing tabpfn and its CUDA-enabled dependencies"
"${PYTHON_BIN}" -m pip install tabpfn

if [[ ! -d "${SYRUPY_SRC_DIR}" ]]; then
  log "Downloading Syrupy ${SYRUPY_VERSION}"
  wget -qO- "${SYRUPY_ARCHIVE_URL}" | tar xzf -
else
  log "Syrupy sources already present; reusing ${SYRUPY_SRC_DIR}"
fi

log "Copying Syrupy CLI into the virtual environment"
cp "${SYRUPY_SRC_DIR}/scripts/syrupy.py" "${VENV_DIR}/bin/syrupy.py"

log "Patching Syrupy script for Python 3 compatibility"
python3 - <<'PY'
from pathlib import Path

path = Path("tab_pfn/bin/syrupy.py")
text = path.read_text()
lines = text.splitlines()

if not lines[0].startswith("#!"):
    lines.insert(0, "#!/usr/bin/env python3")
elif "python3" not in lines[0]:
    lines[0] = "#!/usr/bin/env python3"

text = "\n".join(lines)

if "except Exception as e:" not in text:
    text = text.replace("except Exception, e:", "except Exception as e:")

if "stdout_bytes = ps.communicate()[0]" not in text:
    original = """    stdout = ps.communicate()[0]

    if debug_level >= 9:
        sys.stderr.write(stdout + "\\n")

    if raw_ps_log is not None:
        raw_ps_log.write(stdout + "\\n")

    records = []
    for row in stdout.split("\\n"):"""
    replacement = """    stdout_bytes = ps.communicate()[0]
    stdout = stdout_bytes.decode(errors="replace")

    if debug_level >= 9:
        sys.stderr.write(stdout + "\\n")

    if raw_ps_log is not None:
        raw_ps_log.write(stdout + "\\n")

    records = []
    for row in stdout.split("\\n"):"""
    if original in text:
        text = text.replace(original, replacement)

path.write_text(text)
PY

log "Making Syrupy CLI executable"
chmod +x "${VENV_DIR}/bin/syrupy.py"

log "Ensuring run.sh is executable"
chmod +x run.sh

log "Verifying TabPFN can see CUDA (requires compatible GPU and drivers)"
set +e
"${PYTHON_BIN}" -c "import torch; print('torch version:', torch.__version__); print('cuda available:', torch.cuda.is_available())"
CUDA_STATUS=$?
set -e
if [[ ${CUDA_STATUS} -ne 0 ]]; then
  log "CUDA check failed; investigate torch installation or GPU drivers"
else
  log "Environment ready. Activate with: source ${VENV_DIR}/bin/activate"
  log "Run benchmarks via: ./run.sh"
fi
