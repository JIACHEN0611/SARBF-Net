# 收集真实 Q、最优权重和结果，生成一个完整压缩包

数据集不进入压缩包，只保留 [DATA_SOURCES.md](DATA_SOURCES.md) 的官方获取入口及
[DATA.md](DATA.md) 的本地放置说明。训练权重、匹配的 Q 数组和原结果应从实际实验服务器
收集，不能用新训练的随机权重、临时单位矩阵或手抄表格代替。

当前的 `SARBF-Net_Assets_Prepare_v3.zip` 是**源码与收集工具包**，还没有服务器真实资产。
下面的命令在你的服务器运行后，才会生成真正包含实验材料的 `SARBF-Net_With_Assets_v3.zip`。
不会启动训练，也不会删除或覆盖原实验。

## 一次导出（推荐）

先把下载的 `SARBF-Net_Assets_Prepare_v3.zip` 上传到服务器的 `/home/ljc/pythonProject/`，
在终端执行：

```bash
cd /home/ljc/pythonProject
unzip -n SARBF-Net_Assets_Prepare_v3.zip -d sarbf_publish_v3
python3 sarbf_publish_v3/SARBF-Net/scripts/collect_release.py \
  --run-root \
    /home/ljc/pythonProject/SA_LCM_FULL_MATRIX_52TASK_FINAL_V1/runs/sa_lcm_final_52task_3seed_v1 \
    /home/ljc/pythonProject/SARBF_NET_PEMSD7_PEMSBAY_V1/runs/pems_fact_v1 \
  --study-root depth=/home/ljc/pythonProject/SARBF_NET_SECTION55_V1/runs/section55_v1/depth \
  --output /home/ljc/pythonProject/SARBF-Net_With_Assets_v3.zip \
  --report /home/ljc/SARBF-Net_Collection_Report_v3.json
```

脚本仅用 Python 标准库，收集阶段不需要激活 GPU 环境。它会根据原包定义的记录格式：

1. 搜索最终主模型 `sa_lcm` / `sarbf_net` / `pair_free_fold` 的 `COMPLETE` 记录；
   校验模型与架构版本、输入 96/总历史 768、任务、原始校验信息和有限 MSE/MAE。
2. 根据每条记录指定的 SHA256 匹配所选 `checkpoint.pth` 和输入/输出 Q 数组。
   记录里的绝对路径失效时，会尝试对应实验目录下的权重及按同名、同哈希匹配的 Q 文件。
3. 保留原 `result.json` 字节、Q 与权重字节，生成逐次结果表及按种子汇总的均值和样本标准差。
4. 把这些材料和本包源码合成一个 zip。Q 按内容去重，不包含任何原始数据集。

输出首先显示实际找到的运行数量和资产大小。导出的任务/种子以 `artifacts/manifest.json`
为准，不能把找到的部分任务称为全部 156 次已完成。

以上目录来自用户已提供的服务器输出：主实验固定使用 FULL_MATRIX 下的正式运行目录；
新增交通数据使用独立 pems_fact 目录；深度分析使用 SECTION55 的 depth 子目录。
不再全盘扫描后按第一个遇到的记录决定主结果，也不把消融包中的拷贝重新加入主实验。
若实际目录有变化，应按真实路径修改。

新版支持收集 PEMSD7、PEMS-BAY 记录；原始 52 项训练配置不变，这不代表新增数据集的
训练配方已经自动补入。新增数据集的回放仍以原记录的参数、数据字节和实际预处理为准，
发布前应核对回放指标。

终端会分别显示 `main/standard` 与 `depth/K3` 等数量。深度分析的权重/结果放在
`artifacts/studies/depth/<K>/`，汇总表也按 study、variant 分组，绝不与主结果求平均。
同一组内的重复且不同记录仍会报错。

只想预览数量、不生成大包，可在上面命令末尾增加 `--dry-run`。
脚本会在写入 zip 前检查输出盘空闲空间；截图中磁盘使用率为 96%，若空间不足需要选择
其他有空间的输出目录，不能为打包而直接删除已有实验。
不要在导出过程中修改原结果或覆盖权重。文件在写入时还会再次核对哈希。

## 重复实验、丢失资产或格式不符

如果同一任务/种子存在不同权重或指标，脚本不会按测试分数挑一份，而是报 `ambiguous`。
此时改用你要发表的那一个实验根目录：

```bash
python3 sarbf_publish_v3/SARBF-Net/scripts/collect_release.py \
  --run-root /absolute/path/to/the/intended/run_root \
  --output /home/ljc/pythonProject/SARBF-Net_With_Assets.zip \
  --report /home/ljc/pythonProject/SARBF-Net_Collection_Report.json
```

`--run-root` 后可列多个无冲突的主实验目录。独立参数分析请用
`--study-root 标签=目录`；脚本用 seed 目录前面的相对路径识别 K3、K4 等分析配置。若数据目录已经移动，可额外添加
`--dataset-root /absolute/path/to/dataset`，帮助按文件名与原哈希找回 Q 数组。

如果没有找到兼容记录、权重已经删除、Q 校验不一致，工具会生成报告并停止，不会输出
一个伪装完整的包。请把 `SARBF-Net_Collection_Report.json` 提供给整理者；如果原实验
结果不是本包规定的 `result.json` 格式，需要先检查实际文件结构再做适配。
已删除的真实权重不能从 MSE/MAE 数值反推出来。

## 报告在哪里

旧版报告的绝对路径是 `/home/ljc/pythonProject/SARBF-Net_Collection_Report.json`。
之前统计命令能够读取它，说明当时报告已经生成。MobaXterm 的 SFTP 列表通常先显示目录，
文件在其下方；刷新后向下滚动即可。可在终端确认：

```bash
ls -lh /home/ljc/pythonProject/SARBF-Net_Collection_Report.json
```

新版命令把报告直接放在 `/home/ljc/SARBF-Net_Collection_Report_v3.json`。

## 完整包中的布局

- `artifacts/q/<sha256>.npy`：实际 Q 数组，重复内容只存一份。
- `artifacts/checkpoints/<task>/seed<seed>/checkpoint.pth`：每次运行选定的最优权重。
- `artifacts/results/<task>/seed<seed>/result.json`：原始完整实验记录，保留旧名用于溯源。
- `artifacts/results/per_run.csv`：单次验证/测试指标。
- `artifacts/results/summary.csv`：按任务汇总的种子均值、标准差和实际数量。
- `artifacts/studies/depth/<K>/checkpoints/` 与 `results/`：独立保存的深度配置。
- `artifacts/manifest.json`：可移植路径、原数据哈希和导出文件哈希。
- `artifacts/collection_environment.txt`：收集时的环境，不冒充原训练环境锁。

原 JSON 里的服务器路径不必手工改写；新增清单负责连接可移植路径。
`RELEASE_MANIFEST.json` 在导出时按实际打包源码刷新，资产文件另外由
`artifacts/manifest.json` 记录。检查点不会因改模型名而重新训练。

## 读者直接回放已训练模型

下载数据并放到 `dataset/` 后，在完整包根目录运行：

```bash
python -m tools.sarbf_net.evaluate --task ETTm1-96 --seed 2021 --device cpu
# 有 GPU 时：
CUDA_VISIBLE_DEVICES=0 python -m tools.sarbf_net.evaluate \
  --task ETTm1-96 --seed 2021 --device cuda
```

深度分析可单独选择，例如：

```bash
python -m tools.sarbf_net.evaluate --task ECL-720 --seed 2021 \
  --study depth --variant K3 --device cpu
```

此命令不训练。它检查所需文件哈希，加载原参数，重跑测试集，并报告与记录指标的差值。
默认绝对容差为 `1e-5`；硬件或依赖差异导致超差时会报告不一致。只有清单包含的任务/种子
才能回放。训练仍使用 `tools.sarbf_net.run` 及 `DATA.md` 中的原目录结构；
内容寻址的 `artifacts/q/` 专供已发布权重回放，不会自动覆盖训练数据目录里的 Q 文件。

## 一起上传 GitHub

完整包可以保存为一个交付压缩包；作为源码仓库上传时，要解压后提交文件。
本版已经取消对 `artifacts/checkpoints/` 下 `.pth` 的忽略，并在 `.gitattributes`
中为这些权重配置 Git LFS。原始训练目录、数据集和其他临时权重仍被忽略。

权重数量多或文件较大时，使用 Git / GitHub Desktop 和 Git LFS，不要逐个网页拖拽。
GitHub 网页单文件上限 25 MiB，普通 Git 单文件超过 100 MiB 会被拒绝；LFS 用于大文件。
在已解压的最终仓库根目录、安装好 Git LFS 后运行：

```bash
git lfs install
git add .
git lfs ls-files
git commit -m "Add SARBF-Net matched Q matrices checkpoints and results"
git push
```

上面的 Git 命令适用于已初始化、已配置远程仓库的目录。第一次建仓库先按
[GITHUB_zh-CN.md](GITHUB_zh-CN.md) 用 Desktop 新建/克隆，再在那个目录启用 LFS 后复制
完整包内容并提交。不要直接把大 zip 当作源码推入普通 Git。

官方说明：
https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-large-files-on-github
https://docs.github.com/en/repositories/working-with-files/managing-large-files/configuring-git-large-file-storage
