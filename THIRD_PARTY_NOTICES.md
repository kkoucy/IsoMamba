# Third-party provenance review

This file records review tasks; it does not grant a license or replace upstream notices.

| Component | Evidence in this snapshot | Required before publication |
| --- | --- | --- |
| `basicsr/` | BasicSR namespace, local version `1.2.0+10018c6`; modified restoration/data utilities | Identify the actual upstream revision and any intermediate projects; restore applicable license/copyright notices and document modifications |
| `mamba/` | Vendored source reports 2.2.5; no LICENSE file found | Not included in the export; preserve locally while checking origin and modifications |
| Installed `mamba-ssm` / `causal-conv1d` | Environment reports 1.1.1 / 1.1.1 | Depend on packages rather than redistribute local ABI-specific wheels; verify upstream terms |
| Network blocks and restoration utilities | Similar building blocks do not establish exact provenance | Review original files/history and identify all copied source before selecting a project license |
| Dataset imagery and manuscript | Locally available research materials | Not included in the export; obtain redistribution rights before adding examples or the manuscript |

The root project license is pending author selection. Do not describe this snapshot as a licensed open-source release until the license and third-party notices are complete.
