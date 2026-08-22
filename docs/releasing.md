# Releasing

Listenr versions come from git tags via [setuptools-scm](https://setuptools-scm.readthedocs.io/) —
there is no version number in `pyproject.toml` to bump.

## Cut a release

```bash
git tag v0.2.0
git push origin v0.2.0
```

Pushing the tag triggers [release.yml](../.github/workflows/release.yml), which:

1. Runs the test matrix (3.11, 3.12, 3.13). A tag does not publish code that fails.
2. Builds the sdist and wheel (version `0.2.0`, derived from the tag).
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

## One-time setup

Neither index has been configured yet, and `listenr` does not exist on PyPI.
Both steps below are needed before the first release.

### 1. PyPI trusted publisher

Because the project does not exist yet, this has to be a **pending** publisher.
The project-level *Publishing* settings only appear once a project exists, so
there is nothing to configure there on a first release.

Go to <https://pypi.org/manage/account/publishing/> and add a pending publisher:

| Field | Value |
|---|---|
| PyPI project name | `listenr` |
| Owner | `Rebreda` |
| Repository name | `listenr` |
| Workflow name | `release.yml` |
| Environment name | `pypi` |

The environment name must match exactly, or the OIDC claim is rejected at
upload time. The first successful publish converts the pending publisher into
a normal one.

### 2. TestPyPI trusted publisher (optional, for rehearsals)

Same form at <https://test.pypi.org/manage/account/publishing/>, with
environment name `testpypi`. Skip this if you do not want the rehearsal path;
tag releases work without it.

### 3. GitHub environments

Already created: `pypi` and `testpypi`, both with no protection rules. Add
required reviewers on `pypi` in the repo settings if you want a human gate
before an upload.

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

Between tags, installs from a checkout get a version like `0.2.1.dev3+g2b6c94b`
(next patch, distance from tag, commit hash). Check with `listenr --version`.
