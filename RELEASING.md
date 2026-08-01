# Releasing openptv2

The full release procedure lives in the documentation:

**[docs/developer_guide/packaging_and_releases.md](docs/developer_guide/packaging_and_releases.md)**
(published under *Developer Guide → Packaging & Releases*).

Quick summary:

1. **One-time:** register the PyPI Trusted Publisher (owner `alexlib`, repo
   `openptv2`, workflow `cibuildwheel.yml`, environment `pypi`).
2. Bump `[project].version` in `pyproject.toml`, merge to `main`.
3. Tag and push (`git tag v0.2.2 && git push origin v0.2.2`).

Pushing the tag builds wheels (Linux x86_64, Windows AMD64, macOS arm64;
cp311–313) + sdist, publishes to PyPI via trusted publishing, and attaches the
artifacts to the GitHub Release.
