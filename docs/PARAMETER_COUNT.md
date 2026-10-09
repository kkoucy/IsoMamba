# Parameter counting follow-up

The current network has **24,651,613 unique parameters (24.651613 M)** under the default configuration. This agrees with the existing training logs, including `train.log:174` and `experiments/ClusterMamba/train_ClusterMamba_20260531_140036.log:170`. The loaded final checkpoint matches that network strictly.

Reproduce the count without a CUDA forward pass:

```bash
python tools/count_parameters.py
```

`model.named_parameters()` deduplicates shared tensors. The decoder's `encoder_ref` points to existing encoder modules and does not add new unique parameters. Adding `state_dict()` tensor sizes is unsuitable: that includes repeated names for shared modules and non-parameter buffers.

The original complexity entry at the end of `clusterM_arch.py` used the `params` returned by `thop.profile`. Inspecting the installed THOP implementation shows that its parameter hooks are attached only to supported module types. The stacked `x_proj_weight`, `dt_projs_weight`, `dt_projs_bias`, `A_logs`, `Ds` and other directly registered custom parameters are not all counted. Unsupported tensors total **4,314,390** parameters; the unique parameters covered by supported modules total **20,337,223**. Additionally, inactive experts receive no forward hooks, and the aggregation may traverse shared module references more than once. Thus, the THOP return value is not a reliable complete parameter count for this architecture.

The original CUDA THOP procedure was subsequently run outside the restricted sandbox and reproduced **Params: 15.23M** and **FLOPs: 51.65G**. This establishes that the manuscript's 15.23 M value is reproducible under that procedure, while the complete unique parameter count uses a different convention. The earlier inability to access CUDA was specific to the sandbox, not the host GPU.

Reproduction code, run from the repository root in the documented CUDA environment:

```python
from copy import deepcopy
import torch
from thop import profile
from basicsr.models.archs.clusterM_arch import ClusterMamba

B, C, H, W = 1, 3, 256, 256
x = torch.randn(B, C, H, W).cuda()
model = ClusterMamba().cuda()
model.eval()
with torch.no_grad():
    model_for_flops = deepcopy(model)
    flops, params = profile(model_for_flops, inputs=(x,))
    print(f"FLOPs: {flops / 1e9:.2f}G")
    print(f"Params: {params / 1e6:.2f}M")
    del model_for_flops
    torch.cuda.empty_cache()
```

This run profiles a randomly initialized model and random input. Dynamic routing can change operation counts with the input and initialization; 51.65 G is the observed run, not a guaranteed constant for all inputs. The printed FLOPs label is retained above to reproduce the original reporting convention.

The legacy entry now prints the unique count separately, labels THOP's parameter figure as incomplete, and labels its operation count as partial MACs because selective-scan computation is not counted. A complete FLOP estimate needs explicit rules for the fused scan and other functional operations.
