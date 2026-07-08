# Releasing openptv2 to PyPI

Publishing is automated in [`.github/workflows/cibuildwheel.yml`](.github/workflows/cibuildwheel.yml).
Pushing a version tag builds wheels (Linux x86_64, Windows AMD64, macOS arm64;
cp311/312/313) + the sdist, uploads them to PyPI via **trusted publishing**
(OIDC — no API tokens stored), and attaches them to the GitHub Release.

## One-time setup (maintainer, on PyPI)

Trusted publishing must be configured once. No secrets are added to GitHub.

1. Sign in to <https://pypi.org>.
2. **New project** (openptv2 not yet on PyPI): go to your account →
   *Publishing* → *Add a pending publisher*.
   **Existing project**: project page → *Manage* → *Publishing* → *Add a new publisher*.
3. Fill in exactly:
   - **PyPI Project Name**: `openptv2`
   - **Owner**: `alexlib`
   - **Repository name**: `openptv2`
   - **Workflow name**: `cibuildwheel.yml`
   - **Environment name**: `pypi`   ← must match the workflow's `environment:`
4. Save.

Optional but recommended — gate releases behind approval:
GitHub repo → *Settings* → *Environments* → `pypi` → add yourself as a
**required reviewer**. Each release then waits for a one-click approval before
uploading.

## Cutting a release

1. Bump the version in [`pyproject.toml`](pyproject.toml) (`[project].version`),
   commit, and merge to `main`.
2. Tag and push (tag must match `pyproject.toml`):
   ```bash
   git tag v0.2.1
   git push origin v0.2.1
   ```
   Tag patterns that trigger publishing: `v*` or a leading digit (e.g. `0.2.1`).
3. Watch the **Build Wheels** run. On success it publishes to PyPI and creates
   the release assets. (`skip-existing: true` means re-runs won't fail if a
   version was already uploaded.)

## Test it safely first (optional)

To rehearse without touching real PyPI, add a TestPyPI publisher (same steps,
on <https://test.pypi.org>) and a temporary publish step with
`repository-url: https://test.pypi.org/legacy/`. Remove once confident.

## Notes

- Version is static in `pyproject.toml`; PyPI rejects re-uploading an existing
  version, so always bump before tagging.
- macOS wheels are arm64-only by design; Intel-Mac users install from the sdist.
