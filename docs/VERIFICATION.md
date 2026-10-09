# Verification record

Date: 2026-10-09. Interpreter: local `mamba` environment, Python 3.10 / PyTorch 2.1.2+cu118. This record reports checks actually run; it does not certify a clean-environment installation or paper reproduction.

| Check | Result |
| --- | --- |
| Existing state-contention unit tests | 7/7 passed |
| State-contention CPU self-test | Passed |
| Hidden-state response-quality CPU self-test | Passed after portability edits |
| Raw hidden-state separation CPU self-test | Passed after portability edits |
| Public alias identity | `IsoMamba is ClusterMamba` |
| Portable configuration | Network, train, validation and logger sections equal the original configuration; paths and experiment name are portable |
| Final joint-training checkpoint | `params` loaded with `strict=True`; no missing/unexpected keys |
| Model parameter count | 24,651,613 unique registered parameters |
| Parameter-count follow-up | Historical training logs agree; inspected original THOP hook implementation; added CPU stage-by-stage counting command |
| Original CUDA THOP procedure | Actual random 1×3×256×256 CUDA forward returned FLOPs: 51.65G and Params: 15.23M outside the sandbox |
| Global-state detach | Exact forward equality with non-identity FiLM; context gradient blocked; feature and all FiLM parameter gradients remain nonzero |
| Routing gradient diagnostic | Real RM_decoder0 gather/scatter with CPU expert stubs: feature gradients exist; routing-logit gradient is None |
| Dataset CLI arguments | Both quality/separation parsers apply paired input/GT overrides and reject incomplete pairs |
| Figure regeneration | Three figures rebuilt from portable aggregate JSON as SVG/PDF/PNG |
| Figure numerical evidence | All six plotted means and sample standard deviations match 489 per-image CSV records; original-summary SHA-256 verified |
| Visual inspection | All three PNGs inspected for layout and legibility |
| Syntax and CLI | Modified/new Python modules compile; inference `--help` runs without GPU access |

Local checkpoint identities (weights are excluded from the release export):

```text
ClusterMamba/models/net_g_300000.pth
SHA-256: 6a0aa56c7a6f6f3a3b64282837144bd9d19626946e00212efc2ebf744c952ea9

ClusterMamba_UIEB/models/net_g_300000.pth
SHA-256: 6d49b42340f1e0f818b64aa5e893ef1afa34a11b361a21247655c63e6884a575
```

Only the first checkpoint was loaded in this session. The raw-state diagnostic source identifies that checkpoint filename; its historical run did not record a checkpoint hash, so the present hash is an identity record, not proof that the file has been unchanged since that run.

The original CUDA THOP forward was subsequently run successfully outside the restricted sandbox, reproducing 15.23 M / 51.65 G. Earlier CUDA visibility failures were specific to the sandbox. Full training, fresh dependency installation, checkpoint-based image evaluation and reproduction of benchmark tables were not run. The source export contains a separate `RELEASE_MANIFEST.json` recording the SHA-256 of every copied file.
