# Releasing openptv2

The full release procedure lives in the documentation:

**[docs/developer_guide/packaging_and_releases.md](docs/developer_guide/packaging_and_releases.md)**
(published under *Developer Guide → Packaging & Releases*).

Quick summary:

1. **One-time:** register the PyPI Trusted Publisher (owner `alexlib`, repo
   `openptv2`, workflow `cibuildwheel.yml`, environment `pypi`).
2. Versions come from git tags (setuptools-scm) -- nothing to bump. Tag and
   push (`git tag -a v0.2.2 -m "Release 0.2.2" && git push origin v0.2.2`).
3. Development release of `main` when downstream needs unreleased code:
   `gh workflow run cibuildwheel.yml --ref main` publishes `X.Y.(Z+1).devN`
   (skipped by `pip install openptv2`; `pip install --pre` or a requirement
   like `openptv2>=0.5.12.dev3` gets it).

Pushing the tag builds wheels (Linux x86_64, Windows AMD64, macOS arm64;
cp311–313) + sdist, publishes to PyPI via trusted publishing, and attaches the
artifacts to the GitHub Release.
