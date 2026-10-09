# IsoMamba

论文标题：**State-Isolated Vision Mamba via Update-Tendency Routing for Underwater Image Enhancement**

作者：Yan Luo, Laibin Chang, Zefan Feng, Yi'ang Chen, Kui Jiang, Zhengguo Li

仓库：<https://github.com/kkoucy/IsoMamba>。

![arXiv 审核中](https://img.shields.io/badge/arXiv-pending%20moderation-b31b1b)

**arXiv：论文已提交，目前仍在审核，公开后补充编号和文章链接。** 权重下载、项目许可证及论文与实现的剩余说明见英文 README 和文档。

模型通过来自编码器投影的更新表征进行聚类与路由，将不同组的 token 分配到独立的 Mamba 专家，并通过编码器终态生成 FiLM 调制。主体为四层 U 形网络，包含加法跳连、全局建模的细化模块和图像残差。

![网络结构](assets/figures/architecture.png)

完整安装、训练、推理命令见 [英文 README](README.md)。新增配置 `options/train_IsoMamba.yml` 使用相对路径并取消本机断点；原配置和所有实验文件保持原样。公开名称 `IsoMamba` 是原 `ClusterMamba` 类的别名，不改变参数名称或计算过程。

```bash
# 在依照英文 README 安装环境、准备数据后执行
CUDA_VISIBLE_DEVICES=0 python train.py -opt options/train_IsoMamba.yml
CUDA_VISIBLE_DEVICES=0 python tools/infer.py \
  --weights checkpoints/isomamba.pth \
  --input-dir datasets/example/Input --output-dir results/example
python tools/build_paper_figures.py
python tools/prepare_release.py --output /tmp/IsoMamba-release
```

训练程序会自动恢复同名实验目录下最新的 `.state`，重新训练时应更换实验名称。推理保留原始分辨率，不自动执行论文的 256×256 评测预处理。历史 `test.py` 的 SSIM 设置为 `data_range=1.1`，与训练验证指标不同，尚不能作为统一评测协议。

已准备结构图、聚类路由图和隐藏状态分离诊断图，均包含 SVG/PDF/PNG 版本。诊断使用已有的 UIEB 90 张、LSUI 399 张结果；其对照是同一检查点上的事后分析，不能当作独立训练的 shared-state 基线。图注、误差条含义及数据来源见 [FIGURES.md](docs/FIGURES.md)。

[代码审计](docs/CODE_AUDIT.md) 记录了冗余候选、依赖关系和实现差异。当前没有删除任何原文件；发布导出只复制允许清单中的内容，约 97 GB 的实验产物仍保留本地。正式公开前请完成 [发布事项](docs/RELEASE.md) 中的许可证、权重、评测及论文一致性核对。

原始 THOP 统计代码已在 CUDA 上实际复现：随机 1×3×256×256 输入返回 **Params: 15.23M、FLOPs: 51.65G**。完整去重参数统计为 24.651613 M，两者统计口径不同，详见 [统计说明](docs/PARAMETER_COUNT.md)。
