#!/usr/bin/env sh
# Downloads the Linux Python packages for the Docker image into ./wheels.
# Run once from the project folder:  sh scripts/download_wheels.sh
set -e
cd "$(dirname "$0")/.."
python3 -m pip download -r requirements.txt --dest wheels \
    --only-binary=:all: --python-version 3.10 --implementation cp \
    --abi cp310 --abi abi3 --abi none \
    --platform manylinux_2_28_x86_64 --platform manylinux_2_24_x86_64 \
    --platform manylinux_2_17_x86_64 --platform manylinux2014_x86_64 \
    --platform manylinux_2_5_x86_64 --platform manylinux1_x86_64 \
    --platform linux_x86_64 --platform any \
    --retries 10 --timeout 120
echo "Done: $(ls wheels/*.whl | wc -l) packages in ./wheels. Now run: docker compose up --build"
