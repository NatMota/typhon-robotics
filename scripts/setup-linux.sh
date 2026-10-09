#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/.."
root="$PWD"
[[ "$(uname -s)" == Linux && "$(uname -m)" == x86_64 ]] || {
 echo 'This frozen setup supports Linux x86_64 (Mint and UCL); macOS needs separate dependency locks.' >&2
 exit 1
}
export UV_CACHE_DIR="$root/.cache/uv" UV_PYTHON_INSTALL_DIR="$root/.python" UV_LINK_MODE=copy UV_HTTP_TIMEOUT=300 CC=gcc
if [[ ! -x tools/uv ]]; then
 mkdir -p tools
 curl -fL https://github.com/astral-sh/uv/releases/download/0.6.17/uv-x86_64-unknown-linux-gnu.tar.gz -o tools/uv-linux.tar.gz
 tar -xzf tools/uv-linux.tar.gz -C tools --strip-components=1
 rm tools/uv-linux.tar.gz
fi
for env in data train sim; do
 [[ -x .venv-$env/bin/python ]] || tools/uv venv --python 3.11.12 ".venv-$env"
 requirements="requirements-$env.lock.txt"
 [[ -f "$requirements" ]] || { echo "Missing frozen dependencies: $requirements" >&2; exit 1; }
 tools/uv pip sync --python ".venv-$env/bin/python" "$requirements" --index-url https://pypi.org/simple
done
