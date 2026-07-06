# Releasing

Listenr versions come from git tags via [setuptools-scm](https://setuptools-scm.readthedocs.io/) —
there is no version number in `pyproject.toml` to bump.

## Cut a release

```bash
git tag v0.2.0
git push origin v0.2.0
```

Pushing the tag triggers [release.yml](../.github/workflows/release.yml), which:

1. Builds the sdist and wheel (version `0.2.0`, derived from the tag).
2. Publishes to [PyPI](https://pypi.org/project/listenr/) via trusted publishing.
3. Creates a GitHub Release with auto-generated notes and the build artifacts attached.

Users then install or upgrade with:

```bash
uv tool install listenr        # or: pipx install listenr
```

## One-time setup (already done, for reference)

- **PyPI trusted publishing**: on pypi.org under the project's *Publishing* settings,
  register this repository with workflow `release.yml` and environment `pypi`.
  No API tokens are stored in the repo.
- **GitHub environment**: create an environment named `pypi` in the repo settings
  (optionally with required reviewers to gate releases).

## Dev builds

Between tags, installs from a checkout get a version like `0.2.1.dev3+g2b6c94b`
(next patch, distance from tag, commit hash). Check with `listenr --version`.
