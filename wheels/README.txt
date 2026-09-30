Optional local package cache for the Docker build.

Two ways to fill it:
  1. scripts\download_wheels_direct.ps1  - downloads every link in links.txt (no Python needed)
  2. scripts\download_wheels.ps1         - uses pip (needs Python 3.10+)
The full list of links, versions and SHA-256 checksums is in LINKS.md.

Run one of them or scripts/download_wheels.sh
(macOS/Linux) once. It fills this folder with the Linux packages the API image
needs, using your normal internet connection. The Docker build then installs
from here instead of downloading everything inside Docker.

If this folder has no .whl files, the build simply downloads them as usual.
