# IsoMamba 开源前代码审计

审计日期：2026-10-09。工作目录约 97 GB；当前 `.git/` 是空的只读挂载，`git status` 返回“not a git repository”。因此没有可核对的提交历史，也没有执行 Git 初始化、提交或推送。

## 代码调用关系

```text
train.py → options parser → dataset factory → *_dataset.py 自动导入
         → model factory → ImageCleanModel → architecture factory
                                            → *_arch.py 自动导入
                                            → ClusterMamba / IsoMamba
         → losses / optimizer / scheduler / validation metrics / logging
test.py → ClusterMamba → CUDA selective scan → paired metrics
tools/infer.py → IsoMamba alias → 原网络 → 原分辨率 PNG
analysis/analyze_state_contention.py
  ↑ analyze_hidden_state_quality.py
      ↑ analyze_raw_hidden_state_separation.py
```

`basicsr/utils/__init__.py` 还会导入 `create_lmdb.py`；数据工厂自动导入所有 `*_dataset.py`。文件未被训练配置直接引用，不代表可安全删除。`archs/funcs.py` 没有发现仓库内调用，但在完成外部调用和历史来源核对前保留。

## 冗余候选与处置

| 内容 | 本次观察 | 建议与当前处置 |
| --- | --- | --- |
| `experiments/ClusterMamba*` | 两组各约 49 GB；包含检查点和训练状态 | 发布包排除，原地保留；应先确定最佳/最终权重及恢复需求，再谈清理 |
| 根目录两个 `.whl` | 约 160 MB，绑定 cp310/cu118/torch2.1/ABI | 发布排除，使用依赖说明；本地保留 |
| `mamba/` | 约 2.8 MB，源码 2.2.5 与安装包 1.1.1 不一致 | 发布排除；核查是否有本地修改后再决定归档 |
| `tb_logger/`、`train.log`、缓存 | 训练日志、TensorBoard、字节码 | 发布排除；不删除 |
| `analysis_outputs/` | 约 3.5 MB，包含绝对路径和诊断结果 | 原地保留；公开图只导出必要聚合数值、协议和来源哈希 |
| 本地匿名论文 PDF | 包含匿名投稿信息；正式公开版本尚未提供 | 发布排除；本地保留 |
| `options/UIE.yml` | 绝对数据/模型路径、100k 断点 | 原地保留；发布使用新增相对路径配置 |
| `funcs.py`、face/flow/submission 工具、未使用数据集 | 未找到部分工具的内部调用；工厂存在自动导入 | 当前保留整个 BasicSR 子树；先建立准确依赖/来源清单，再精简 |
| 重复 import、注释旧实现、演示代码 | 主网络和 loss 存在历史遗留内容 | 暂不修改科学实现；后续独立重构并验证 checkpoint/输出等价性 |

本次“整理”使用文档、公开类名别名、可移植配置和允许清单导出实现，没有删除或移动原文件。

## 论文与实现待核对

| 项目 | 本地论文 | 当前代码/环境 | 影响 |
| --- | --- | --- | --- |
| 初始学习率 | 1e-4 | `UIE.yml` 为 3e-4，前 92k 固定、后 208k 余弦 | 尚不能宣称严格复现论文训练方案 |
| PyTorch | 2.4.1 | 已装 2.1.2+cu118 | 区分历史环境与最终复现环境 |
| 参数量 | 15.23 M | 原始 CUDA THOP 代码已实跑复现 15.23 M / 51.65 G；完整去重参数与训练日志为 24,651,613 | 两种统计口径分别记录，详见 `PARAMETER_COUNT.md` |
| Top-1 STE | 声称软概率承担反向传播 | `expert_assignments_hard` 只用于 `any/nonzero` 索引，选取特征不乘路由权重 | 已用真实 RM_decoder0 gather/scatter 路径与 CPU expert stub 验证：features 有梯度，路由 logits.grad 为 None；仍需核对最终训练版本 |
| 全局状态 stop-gradient | 显式 `sg[H_g]` | 已在 FiLM 输入处加入 `h_global.detach()`；编码器投影路由已有 detach | 明确阻断全局状态分支梯度，同时保留 FiLM 参数与 decoder 特征梯度 |
| LSUI 样本数 | 主表协议 400 | 本地 raw-state 诊断 399 | 不能混用诊断样本数和主表样本数；核查缺失配对 |
| SSIM | 需要统一正式定义 | 历史 `test.py` 为 pytorch-msssim/data_range=1.1，训练验证为其他实现 | 主表重跑前统一 resize、范围、颜色空间和指标实现 |
| 序列打包 | 方法描述为独立序列 | 真模型平方打包并零填充、执行二维卷积和四向 scan | 诊断的 logical packing 排除填充，不能直接当作真实 forward 的完整消融 |

别名与全局状态 detach 均保持 state_dict 键不变。detach 只明确全局上下文梯度边界，不改变前向数值；索引式路由未改动。

## 图表证据边界

状态分离图使用 `raw_hidden_state_separation` 的聚合结果。Shared/Isolated Fixed Expert 共用 expert 0 的 tokenwise operands，仅改变状态组织/组内顺序；Full Isolation 还改变专家参数。所有结果来自同一训练检查点的事后干预；不是独立训练 shared-state 网络的指标。这里的 separation 分数也不是 PSNR、SSIM 或增强质量指标。

CPU 导入、300k 权重严格加载、诊断 self-test 和图表重建已通过验证。随后在沙箱外成功执行原始 CUDA THOP forward；此前 NVIDIA 不可见属于沙箱限制。从零训练和正式评测仍待复现。诊断脚本新增数据路径 CLI、公开配置默认值和可选单卡绑定，未修改递推或统计方法。剩余事项见 `RELEASE.md`，验证记录见 `VERIFICATION.md`。
