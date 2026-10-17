# SARBF-Net

**Selective Anchored Retrieval & Balanced Feedback Network**。[English](README.md)

我们使用 SAER 从历史观测中检索相关信息，修正近期输入，再通过 BCIF 协调变量内更新与跨变量交互。总输入为 **768 点**：前 672 点组成七个历史块，后 96 点为近期窗口；修正后的 96 点进入预测主干。输出端采用折叠后的仿射预测头。

仓库还包含 SAER 在 **DLinear、PatchTST、iTransformer、CycleNet、TSMixer** 上的迁移实验。这部分只接入 SAER，不接 BCIF；原骨干与加入 SAER 的实验组使用相同数据窗口和骨干训练配置。

## 安装与运行检查

先安装与硬件匹配的 PyTorch，再安装其余依赖：

```bash
python -m pip install -r requirements.txt
python -m pip install -r requirements-dev.txt
python -m pytest tests -q
python -m tools.sarbf_net.smoke
python -m tools.saer_transfer.smoke
```

测试和 smoke 使用 CPU 与合成输入，不需要下载数据或训练权重。实际检查范围见 [验证报告](docs/VALIDATION_REPORT.json)。

## 数据与实验资产

原始数据不随仓库分发，下载入口见 [数据来源](docs/DATA_SOURCES.md)，文件组织见 [DATA.md](docs/DATA.md)。训练从本地文件读取数据。

主模型的 Q 矩阵、训练权重和实验记录位于 [v1.0.0 Release](https://github.com/JIACHEN0611/SARBF-Net/releases/tag/v1.0.0)。将 `SARBF-Net_Assets_v1.7z.001` 至 `.004` 四个分卷下载到同一文件夹，用 7-Zip 解压 `.001`，再将其中的 `artifacts/` 放到 `run.py` 同级。具体结构和复评命令见 [资产说明](docs/ASSETS_zh-CN.md)。

源码包不重复包含这些大型文件。五骨干迁移部分提供 **720 项预设配置**；这不表示已收集 720 项结果，v1.0.0 附件也不包含该完整迁移矩阵的训练权重与结果。

## 主模型训练与汇总

按 DATA.md 准备数据和对应的 Q 文件后执行：

```bash
python -m tools.sarbf_net.inputs --tasks ETTm1-96
python -m tools.sarbf_net.run --task ETTm1-96 --seed 2021 --dry-run
python -m tools.sarbf_net.run --task ETTm1-96 --seed 2021 --run-root runs/sarbf_net
python -m tools.sarbf_net.summarize --run-root runs/sarbf_net --output summary.csv
```

`--validation-only` 只进行训练及验证集评估，不评估测试集。完整主矩阵保留 52 项配置、三个种子 2021/2022/2023。Release 另有交通扩展记录，具体范围以资产清单为准；这些记录不等于 52 项启动配置已经扩展。

## 五骨干迁移实验

```bash
python -m tools.saer_transfer.run --backbone DLinear --dataset ETTm1 --horizon 96 --seed 2021 --arm sa_frontend --dry-run
```

数据路径、两组对照、批量运行和结果汇总见 [SAER_TRANSFER.md](docs/SAER_TRANSFER.md)。

## 代码结构与复现范围

主模型入口为 `model/SARBF_Net.py`，有效实现位于 `model/components/`。保留的 `OLinear*.py` 已缩为兼容导入文件，用于早期脚本和旧实验；命名与权重字段见 [NAMING.md](docs/NAMING.md)。五个骨干在 `model/backbones/`，各自的 SAER 入口在 `model/saer_adapters/`。

本次重构在独立进程中对照原始代码，检查初始化、前向、梯度、一次优化及严格权重交叉加载。CPU 数值检查不能替代原始数据上的已发布权重复评，也不能证明与后续服务器 `protocol_v2` 的所有修改一致，详细见 [复现协议](docs/REPRODUCIBILITY.md)。

第三方来源和许可见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)，引用信息见 [CITATION.md](CITATION.md)。仓库更新步骤见 [中文更新说明](docs/GITHUB_UPDATE_zh-CN.md)。
