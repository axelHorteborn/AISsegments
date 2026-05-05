# Releasing AISsegments

Reference checklist for cutting a new release.  Authoritative configuration
lives in [`pyproject.toml`](pyproject.toml) and
[`.github/workflows/release.yml`](.github/workflows/release.yml).

## One-time PyPI setup

You only need to do this once per registry (TestPyPI and PyPI are
separate accounts and separate publisher configs).

### Account essentials

- PyPI account: https://pypi.org — login `axelhorteborn`
- TestPyPI account: https://test.pypi.org — login `axelhorteborn`
- Enable 2FA on both (TOTP or hardware key).  PyPI requires it for
  anyone who uploads a package.

### Trusted-publisher entries

These are what let the GitHub Action publish without storing API tokens.
On both PyPI and TestPyPI, go to **Account → Publishing → Add a new
pending publisher** and fill in:

| Field | Value |
| --- | --- |
| PyPI Project Name | `aissegments` |
| Owner | `axelande`  *(GitHub username — NOT your PyPI login)* |
| Repository name | `AISsegments` |
| Workflow name | `release.yml` |
| Environment name | `pypi` for the PyPI entry, `testpypi` for the TestPyPI entry |

Important: the **Owner** field is your **GitHub** username (`axelande`),
not your PyPI login (`axelhorteborn`).  PyPI uses the GitHub repo URL,
not your PyPI account name, to verify the OIDC token at publish time.

After saving, the entry shows as "pending" until the first publish lands.

### Optional: GitHub environments

Create matching environments at
`https://github.com/axelande/AISsegments/settings/environments`:

- `pypi` — protected; recommended to require manual approval before deploy.
- `testpypi` — typically unprotected.

The release workflow already references both names; without environments
configured the publish jobs run unprotected, which is fine for a
single-maintainer repo.

## Cutting a release

1. **Update the version** in two places — they must match the tag:
   - `pyproject.toml` → `[project]` → `version`
   - `src/aissegments/__init__.py` → `__version__`

2. **Verify locally**:

   ```bash
   pip install -e ".[dev]"
   ruff check .
   pytest --cov                   # must report 100%
   python -m build                # produces dist/*.whl + *.tar.gz
   twine check dist/*             # validates README rendering, metadata
   ```

3. **Commit + tag + push**.  Tag conventions:

   | Tag pattern        | Routes to            |
   | ------------------ | -------------------- |
   | `v0.2.0`           | PyPI                 |
   | `v0.2.0-rc1`       | TestPyPI (dress rehearsal) |
   | `v0.2.0rc1`        | TestPyPI (PEP 440 form) |

   ```bash
   git add pyproject.toml src/aissegments/__init__.py
   git commit -m "Release v0.2.0"
   git tag v0.2.0
   git push origin main --tags
   ```

4. **Watch the workflow** at
   `https://github.com/axelande/AISsegments/actions`.
   First run takes ~3 minutes (test + build + publish).  If the build job
   reports "Tag does not match pyproject.toml version", you forgot
   step 1.

5. **Verify** the package landed:

   ```bash
   # ~30 s after the workflow turns green
   pip install --upgrade aissegments
   python -c "import aissegments; print(aissegments.__version__)"
   ```

## First-release dress rehearsal

For the very first publish — when the package name still hasn't been
locked on either registry — do this:

```bash
# Tag a release-candidate first.  Goes only to TestPyPI.
git tag v0.2.0-rc1
git push origin v0.2.0-rc1

# Once the workflow is green, install from TestPyPI to confirm.  The
# extra-index-url is necessary because numpy lives on real PyPI, not
# TestPyPI.
pip install \
  --index-url https://test.pypi.org/simple/ \
  --extra-index-url https://pypi.org/simple/ \
  aissegments==0.2.0rc1

# Happy with it?  Tag the real release.
git tag v0.2.0
git push origin v0.2.0
```

## Troubleshooting

**"Trusted publisher rejected the request"** — the publisher entry on
PyPI/TestPyPI didn't match.  Common causes: GitHub username mismatch
(`axelande` vs `axelhorteborn` — see the "Owner" note above), workflow
filename typo, environment name mismatch.

**"403 Forbidden — invalid or non-existent authentication"** on a manual
`twine upload` — your API token expired or was revoked.  Either issue a
new one, or rely on the GitHub Action (recommended).

**"File already exists"** — PyPI doesn't allow overwriting a published
version.  Bump the version (even just the patch), retag, push.

**Tag-vs-version mismatch in the workflow** — the build job's `Verify
tag matches package version` step caught it.  Fix `pyproject.toml` and
`src/aissegments/__init__.py`, delete the bad tag (`git tag -d v0.2.0;
git push origin :refs/tags/v0.2.0`), retag.

## After publication

OMRAT picks up the new version automatically:

- End users: next time they load the QGIS plugin, `qpip` reads
  [OMRAT/requirements.txt](https://github.com/axelande/OMRAT/blob/main/requirements.txt)
  and pulls the new `aissegments` from PyPI.
- You developing OMRAT: `pip install --upgrade aissegments` in your dev
  environment.

If a published release accidentally breaks downstream code, **don't try
to fix it in place** — yank it and publish a patch:

```bash
# On https://pypi.org/manage/project/aissegments/release/0.2.0/
#   click "Yank release", give a reason.  pip stops installing yanked
#   versions by default but lets users force them with
#   `pip install aissegments==0.2.0`.
```

Then bump to `v0.2.1` and re-release.
