# Release preparation

Target: https://github.com/kkoucy/IsoMamba

The local source export is a reviewable preparation artifact. It is not a publication, a verified paper reproduction, or a license grant. No original files were removed.

Before public release:

- Confirm the manuscript title, author spelling/order, affiliations, corresponding author, public paper URL and final citation metadata. The supplied author list is already in both READMEs.
- Select the project license and complete the source provenance review in `THIRD_PARTY_NOTICES.md`.
- Resolve the learning-rate, environment, parameter-count and routing-gradient discrepancies in `CODE_AUDIT.md`; record the chosen code/checkpoint version.
- Select and evaluate the intended public checkpoint, provide its SHA-256 and download URL, and distinguish joint-training and UIEB-only checkpoints.
- Validate installation in a clean CUDA environment and run checkpoint-based inference/training smoke checks. The original THOP CUDA forward succeeded outside the sandbox.
- Define one evaluation protocol (256×256 resize versus original resolution, pairing, RGB/range, PSNR and SSIM implementation); reproduce the paper tables under that protocol.
- Check the LSUI 399-versus-400 discrepancy and the actual dataset split lists.
- Portable dataset CLI arguments and configurable single-GPU selection are implemented; verify their full CUDA runs with the selected public checkpoint.
- Verify figure captions against the final implementation. Real enhancement comparisons require matching baseline implementations/checkpoints and redistributable images.
- Review the exported manifest and repository contents, then initialize Git in the clean export directory and publish when authorized. Do not commit the working directory wholesale.

To prepare a fresh export:

```bash
python tools/prepare_release.py --output /tmp/IsoMamba-release
```

The export includes source, portable configuration, documentation and prepared figure artifacts. It excludes original machine-specific `options/UIE.yml`, training state, weights, wheels, data, analysis outputs, the anonymous PDF and vendored `mamba/`. It retains the BasicSR source tree because removing automatically imported modules prematurely can break training. The script refuses existing destinations and source symlinks and records hashes of every exported file.
