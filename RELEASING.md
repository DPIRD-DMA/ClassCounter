# Releasing

ClassCounter is published to PyPI by tag-push. The version comes from the git
tag itself (via `hatch-vcs`), so there is **no version to bump** in
`pyproject.toml`.

## Versioning

Follow [SemVer](https://semver.org/):

- **Major** (`v1.0.0` → `v2.0.0`): incompatible API change. Removed args,
  changed signatures, or behavioural changes that break existing callers.
- **Minor** (`v0.1.0` → `v0.2.0`): backwards-compatible feature added.
- **Patch** (`v0.1.0` → `v0.1.1`): backwards-compatible bug fix.
- **Pre-release** (`v0.2.0b1`, `v0.2.0rc1`, …): a beta of the next version. PyPI
  treats these as pre-releases, so `pip install classcounter` keeps resolving to
  the latest stable and only `pip install --pre classcounter` picks them up.

Commits on `main` between releases show up at install time as a development
version like `0.2.1.dev3+g1234abc`, which is useful but not something you publish.

## Release checklist

1. **Sync `main`** locally and check CI is green on it:
   ```bash
   git checkout main
   git pull
   ```

2. **Update [`CHANGELOG.md`](CHANGELOG.md)**: add a `## [<version>] - <YYYY-MM-DD>`
   section at the top, above the previous release, with the changes grouped under
   `### Added` / `### Changed` / `### Fixed`. Commit on `main`:
   ```bash
   git commit -am "Cut <version>: <short summary>"
   git push
   ```

3. **Tag and push**:
   ```bash
   git tag v<version>          # e.g. v0.2.0
   git push origin v<version>
   ```
   The tag must match `v<major>.<minor>.<patch>`, optionally with a pre-release
   suffix. The publish workflow filters on `v[0-9]+.[0-9]+.[0-9]+*`, and refuses to
   publish if the built version doesn't match the tag.

4. **Approve the deploy**: the [`Publish to PyPI`](.github/workflows/publish.yml)
   workflow runs `uv build` and uploads to PyPI via OIDC trusted publishing.
   The `pypi` GitHub environment is gated by a required reviewer, so go to
   **Actions → Publish to PyPI → Review deployments → Approve** to release.

5. **Verify**:
   - PyPI page: <https://pypi.org/project/classcounter/>
   - GitHub Releases: <https://github.com/DPIRD-DMA/ClassCounter/releases>
     (auto-generated release notes + wheel/sdist attached).

## One-time setup

Both are needed before the first tag-push release:

- **PyPI trusted publisher**: at
  <https://pypi.org/manage/project/classcounter/settings/publishing/> add a GitHub
  publisher with owner `DPIRD-DMA`, repository `ClassCounter`, workflow
  `publish.yml`, environment `pypi`.
- **GitHub `pypi` environment**: in **Settings → Environments**, create `pypi`
  with yourself as a required reviewer, and under deployment branches and tags
  allow only tags matching `v*`.

## Development setup

```bash
uv sync
uv run pre-commit install --hook-type pre-commit --hook-type pre-push
```

Commits run ruff and mypy; pushes also run the test suite (minus the slow
install tests, which CI runs).
