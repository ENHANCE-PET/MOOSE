# Security audit — 2026-10-31

Scope: current source, dependency declarations, model/data downloads, archive
handling, subprocess calls, credential patterns, and GitHub workflows. No model
weights or patient data were downloaded. This is not a claim that all possible
vulnerabilities have been excluded.

## Verified findings and repairs

- **Model cache overwrite through ZIP members.** Archives were extracted into the
  shared model base, allowing one archive to overwrite another cached model.
  Extraction now uses a fresh private directory, validates every member before
  extraction, allows only the selected model root and regular files/directories,
  and rejects traversal, Windows paths, and special files. The installed version
  remains intact when download or validation fails. Python's ZIP extraction
  already normalizes `..`; the demonstrated overwrite was through sibling roots
  and existing filesystem links, not a claim of unrestricted ZIP `../` traversal.
- **Local symlink clobber during model and ENHANCE downloads.** Predictable output
  names were opened with `wb`, following pre-existing symlinks. Downloads now use
  private temporary files/directories and rename completed output into place.
  HTTP responses are closed and connect/read timeouts are set.
- **Release workflow shell injection.** Dispatch/release tag text was substituted
  directly into bash before validation. It now enters through environment
  variables, so shell substitutions in tag text remain literal data. A synthetic
  `$(touch ...)` payload executed on the old workflow and is rejected safely now.
- **Unverified model release bytes.** SHA256 digests for the 24 GitHub-hosted
  built-in model assets were obtained from the repository's Release asset API
  metadata and pinned in `moosez/model_checksums.json`. Downloads must match before
  extraction. Legacy URL-only caches refresh once. This trusts the published
  metadata observed on the audit date; it does not retrospectively establish
  model provenance or make arbitrary checkpoints safe to deserialize.
- **Dependency declarations admitted known vulnerable versions.** Both manifests
  now require `torch>=2.13.0`, `pydicom>=3.0.2`, and `pillow>=13.0.0`. Pillow is a
  transitive image-processing dependency whose 12.3.0 release has multiple current
  memory-safety and denial-of-service advisories. Pillow 13 requires Python 3.11;
  package metadata, CI, and support documentation therefore raise the minimum
  from Python 3.10 to 3.11. The Docker image already uses Python 3.13. The
  requirements file also carries the existing pyproject bounds for Requests, Dask,
  and nnUNet. Explicit security floors for already-transitive AnyIO, certifi,
  filelock, fsspec, msgpack, setuptools, and urllib3 ensure an in-place installation
  upgrades known vulnerable versions rather than keeping them satisfied under
  upstream broad ranges. The setuptools build requirement uses the same floor.
  Tornado is absent from this runtime graph and was not added.

Authoritative advisory sources:

- [PyTorch torch.jit.script memory corruption, patched 2.13.0](https://github.com/advisories/GHSA-rrmf-rvhw-rf47)
- [PyTorch checkpoint loading RCE, patched 2.6.0](https://github.com/pytorch/pytorch/security/advisories/GHSA-53q9-r3pm-6pq6)
- [pydicom FileSet traversal, patched 3.0.2 or 2.4.5](https://github.com/advisories/GHSA-v856-2rf8-9f28)
- [Pillow FPX heap overflow, patched 13.0.0](https://github.com/advisories/GHSA-5x8w-8jj5-q52h)
- [Pillow JPEG heap overflow, patched 13.0.0](https://github.com/advisories/GHSA-8g5g-8vmr-9782)

The pydicom FileSet API is not directly called here, but old dependency
installations still admitted the affected package versions. The 2026 MONAI
DS/IS advisory GHSA-6hp3-vr39-rqw8 concerns MONAI, not a separate pydicom floor.

## Validation and limits

- `python3 -m pytest -q tests/test_security.py`: 10 tests and 7 subtests pass
  on Python 3.11.14 and 3.13. Tests use synthetic ZIPs, response bytes, filesystem symlinks, and real bash;
  only the heavy application/UI import and network boundary are substituted.
  Regressions were observed failing before the corresponding repair.
- `python3 -m compileall -q moosez tests/test_security.py`: passes.
- A wheel builds without dependencies and contains the checksum manifest.
- `uv pip compile pyproject.toml --python-version 3.11`: complete metadata
  resolution succeeds at the new minimum supported version. Resolution chooses
  torch 2.15.0, nnunetv2 2.8.1, pydicom 3.0.2, and Pillow 13.0.0. This checks
  resolver compatibility, not runtime inference compatibility.
- `pip-audit -r moose-resolved-py311.txt --no-deps --disable-pip`: 101 resolved
  dependencies audited, none skipped, no known vulnerabilities reported. This is
  a dated advisory-database result, not a guarantee about every installable
  dependency combination or vulnerabilities missing from that database.
- `python3 -m pytest -q`: collection stops at `tests/test_e2e.py` because SimpleITK
  is not installed in the lightweight audit environment. Full inference was not
  run; it needs the imaging stack, weights, and the existing clinical fixture.

## Remaining trust and operational limits

- `clin_ct_dental` uses the external DentalSegmentator host, which lacks a known
  independently published checksum in this audit. That one model retains its
  existing HTTPS trust boundary; its archive still receives the path/type checks.
- Checkpoint loading is delegated to nnUNet. Only trusted model sources and
  private model caches should be used; archive validation is not a pickle sandbox.
- Cached model contents are not rehashed on every inference. A process with write
  access to the cache can still change installed weights or their version marker.
- Download/extraction staging has no total size or decompression quota. A
  compromised model publisher can still consume disk space. Replacement of an
  existing model is not a transaction against crashes or concurrent writers.
- GitHub Actions references and the Docker base image remain mutable upstream
  tags. No literal private key or common GitHub/AWS credential pattern was found
  in the current scanned source. This was not a complete history secret scan.
