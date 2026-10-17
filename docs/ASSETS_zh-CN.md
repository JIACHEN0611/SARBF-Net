# 实验资产

主模型实验材料保存在 [v1.0.0 Release](https://github.com/JIACHEN0611/SARBF-Net/releases/tag/v1.0.0)，包含相互匹配的 Q 矩阵、选定训练权重、实验记录及资产清单。原始数据不在附件内，下载入口见 [DATA_SOURCES.md](DATA_SOURCES.md)。

## 下载与解压

1. 将 `SARBF-Net_Assets_v1.7z.001`、`.002`、`.003`、`.004` 全部下载到同一目录，保持文件名和后缀。
2. 用 7-Zip 打开 `.001` 并解压；不用逐个解压其余分卷。
3. 在解压结果中找到 `artifacts/`，将它放在源码根目录，与 `run.py` 同级。不要再套一层同名文件夹。
4. 检查 `artifacts/manifest.json` 和其中列出的 `q/`、`checkpoints/`、`results/` 路径。覆盖范围、相对路径及哈希以该清单为准。

Release 页面自动生成的 Source code ZIP/tar.gz 是标签对应的源码快照，不是上述实验资产分卷。更新主分支源码后，v1.0.0 的四个附件和旧标签继续保留。

## 已发布权重复评

准备与记录哈希匹配的原始数据后，从源码根目录执行：

```bash
python -m tools.sarbf_net.evaluate --task ETTm1-96 --seed 2021 --dataset-root dataset --device cpu
```

默认 study 为 `main`、variant 为 `standard`；其他实验按清单传入 `--study` 和 `--variant`。评估器核对记录、Q、权重和数据哈希，并比较回放指标。Q 不保存在旧 checkpoint 中，必须使用与权重配套的外部文件。

资产目录供已发布结果复评使用。重新训练时，主模型启动器按 [DATA.md](DATA.md) 读取数据旁的 Q 文件；两种路径用途不同，不能只复制 checkpoint 就开始复评。

## 新实验材料

`python scripts/collect_release.py --help` 提供服务器资产收集参数。收集器保留实验字节并生成清单，不生成替代 Q 或训练权重。新增材料应使用新附件名和清单版本，保留已有发布记录。

本次源码整合没有重新下载并复评全部已发布权重；数值兼容检查范围见 [VALIDATION_REPORT.json](VALIDATION_REPORT.json)。五骨干迁移矩阵的权重与结果不包含在当前 v1.0.0 附件中。
