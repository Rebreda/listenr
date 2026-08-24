# Releasing

Listenr versions come from git tags via [setuptools-scm](https://setuptools-scm.readthedocs.io/): there is no version number in `pyproject.toml` to bump.

## Cut a release

```bash
git tag v0.3.0
git push origin v0.3.0
```

Pushing the tag triggers [release.yml](../.github/workflows/release.yml), which:

1. Runs the test matrix (3.11, 3.12, 3.13). A tag does not publish code that fails.
2. Builds the sdist and wheel, with the version derived from the tag.
3. Runs `twine check --strict`, confirms the built version matches the tag, and
   installs the wheel into a clean venv to check the `listenr` CLI actually runs.
4. Publishes to [PyPI](https://pypi.org/project/listenr/) via trusted publishing.
5. Creates a GitHub Release with auto-generated notes and the artifacts attached.

Users then install or upgrade with:

```bash
uv tool install listenr        # or: pipx install listenr
```

## Rehearsing without touching PyPI

Run the Release workflow manually from the Actions tab. A manual run always
publishes to TestPyPI, never PyPI, so it cannot put a `.dev` version on the
real index. Everything else in the pipeline is identical.

Requires the TestPyPI side of the one-time setup below.

## Publishing setup

Already done, recorded here so it can be re-created or audited.

PyPI uses trusted publishing, so no API token exists anywhere in the repo or
in GitHub secrets. The publisher is registered on pypi.org under the project's
*Publishing* settings with:

| Field | Value |
|---|---|
| Owner | `Rebreda` |
| Repository | `listenr` |
| Workflow | `release.yml` |
| Environment | `pypi` |

All four are case-sensitive and must match the workflow exactly, or PyPI
rejects the OIDC claim at upload time.

The `pypi` and `testpypi` GitHub environments exist with no protection rules.
Add required reviewers to `pypi` in the repo settings if you want a human gate
before an upload.

TestPyPI has no publisher registered. The manual rehearsal above will fail at
the upload step until one is added, using the same form at
<https://test.pypi.org/manage/account/publishing/> with environment
`testpypi`. Tag releases do not need it.

## If the publish fails

- `invalid-publisher` or a rejected OIDC claim: the owner, repo, workflow
  filename or environment name on PyPI does not match the workflow. All four
  are case-sensitive.
- Version check fails with a `.dev` suffix: the tag is not on the commit being
  built. Confirm with `git describe --tags`.
- A version cannot be re-uploaded. If a release is wrong, yank it on PyPI and
  publish a new patch version. Deleting a tag and re-pushing it does not free
  the version number.

## Dev builds

Between tags, installs from a checkout get a version like `0.2.1.dev3+g2b6c94b`:
next patch, distance from the tag, commit hash. Check with `listenr --version`.
