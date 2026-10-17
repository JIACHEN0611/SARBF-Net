# SARBF-Net 中文说明

模型全称：**Selective Anchored Retrieval & Balanced Feedback Network**。
主要模块为 SAER（选择性锚定检索前端）和 BCIF（平衡跨变量与变量内反馈）。

这是原始最终代码包的仓库整理版。新模型入口是 `model/SARBF_Net.py`，新实验入口是
`python -m tools.sarbf_net.run`。旧接口保留用于已有权重和实验记录兼容，详细见
[命名对照](docs/NAMING.md)。训练超参数、历史块划分和计算公式保持原协议。

## 安装与检查

先激活你服务器原来能跑通实验的环境，再进入本仓库根目录。不要直接重装现有 GPU 版
PyTorch。新机器先根据 [PyTorch 官方页面](https://pytorch.org/get-started/locally/)
安装适配硬件的 PyTorch，再安装依赖：

```bash
python -m pip install -r requirements.txt
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m unittest discover -s tests -v
python -m tools.sarbf_net.smoke
python -m tools.sarbf_net.run --task ETTm1-96 --dry-run
```

CPU 小测试只验证随机输入下的前向和反向传播，不能当作论文结果复现。
本次实际验证环境记录在 `docs/validation-environment.txt`，它不是原服务器训练环境。

## 汇总服务器真实实验材料

要把 Q、最优权重与原始结果全部并入一个压缩包，按 [资产收集说明](docs/ASSETS_zh-CN.md)
运行 `scripts/collect_release.py`。当前准备包不含真实服务器权重；只有导出成功的
`SARBF-Net_With_Assets_v3.zip` 才包含这些真实材料。数据下载地址见
[官方来源与镜像](docs/DATA_SOURCES.md)。已发布权重可用 `tools.sarbf_net.evaluate` 直接回放。

## 准备数据与 Q 矩阵

**当前源码准备包没有包含数据集、真实实验 Q 矩阵、训练权重和实验结果。**
服务器导出的完整包则会包含清单列出的 Q、最优权重和结果，仍不包含原始数据集。
请按 [数据说明](docs/DATA.md) 组织原始数据。Q 矩阵必须从原实验服务器导出，仓库已提供
`tools.sarbf_net.export_q`；不能用单位矩阵或新算法重新生成的矩阵冒充原实验资产。

数据准备完成后先检查单个任务：

```bash
python -m tools.sarbf_net.inputs --tasks ETTm1-96
CUDA_VISIBLE_DEVICES=0 python -u -m tools.sarbf_net.run \
  --task ETTm1-96 --seed 2021 --run-root runs/sarbf_net
```

结果位置为 `runs/sarbf_net/seed2021/ETTm1-96/sarbf_net/result.json`。
训练日志在同目录的 `train.log`，状态为 `COMPLETE` 才表示训练与所选权重的独立回放完成。
`--validation-only` 仅训练并评估验证集，状态为 `VALIDATED`。

## 全部任务与汇总

```bash
bash scripts/run_sarbf_matrix.sh --list-tasks
CUDA_VISIBLE_DEVICES=0 bash scripts/run_sarbf_matrix.sh
python -m tools.sarbf_net.summarize --run-root runs/sarbf_net --output summary.csv
```

保留原包的 13 个数据集、52 项配置，种子 2021/2022/2023，共 156 次训练；
这不等同于当前论文每张表的范围，Exchange 是否入表由论文决定。
批量脚本串行执行，跳过已经 `COMPLETE` 的项。失败目录不会被覆盖，也不会自动从中间
epoch 续训；请换 `--run-root` 或移走对应失败目录后重跑。
汇总仅纳入 COMPLETE 结果，标准差为样本标准差（ddof=1），单种子留空。

**输入协议：总历史长度 768，前 672 点作为七个历史块，后 96 点作为近期窗口；
SAER 修正后的 96 点进入紧凑主干。不能写成模型仅使用 96 点总输入。**

## 放到 GitHub

按 [中文上传教程](docs/GITHUB_zh-CN.md) 操作。推荐 GitHub Desktop，它能保留隐藏的
`.github`、`.gitignore` 文件并正确处理文件夹结构。上传解压后的源码，压缩包本身放到
GitHub 并不会自动成为可浏览的代码仓库。

原始数据、临时运行目录和日志仍被 `.gitignore` 排除。收集后的 `artifacts/q/`、
`artifacts/results/` 可提交；`artifacts/checkpoints/` 下的精选权重已经配置 Git LFS。
完整包上传方法见 `docs/ASSETS_zh-CN.md`。

当前未擅自选择 MIT 或 Apache 等全仓库许可证。原包中有第三方代码及已有版权声明，
具体情况在 `THIRD_PARTY_NOTICES.md` 与 `LICENSE.md` 中如实记录。整理版尚不能宣称
“开箱即复现论文所有结果”；还需要原实验资产、历史环境与最终论文配置核对。

新版收集器支持 PEMSD7、PEMS-BAY，并使用 `--study-root depth=...` 分开保存深度分析，
汇总表按 study/variant 分组。具体服务器命令和报告位置见 `docs/ASSETS_zh-CN.md`。
