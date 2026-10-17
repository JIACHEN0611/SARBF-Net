# 将整合后的源码更新到 GitHub

这次更新使用 `SARBF-Net_Source_v1.1.0.zip`，其中只有源码、文档和测试。已经上传到 v1.0.0 的四个资产分卷继续保留，不用重新上传，也不用删除旧 Release 或移动 v1.0.0 标签。文件名中的 v1.1.0 是本次源码包版本，不会自动建立 GitHub Release。

## 先解压并保留备份

1. 保留原来约 3 GB 的完整包和四个分卷。另建一个目录，例如 `D:\SARBF-Net-Source-v1.1.0`。
2. 将新的源码 ZIP 解压到这个新目录，打开其中的 `SARBF-Net`。这一层直接看到 README.md、run.py、model、tools 等文件。
3. 以后上传的是这一层里面的内容。不要上传整个 ZIP，也不要在仓库根目录增加一层 `SARBF-Net` 文件夹。

这次的五个骨干为 DLinear、PatchTST、iTransformer、CycleNet、TSMixer。旧的 OLinear 文件已缩成兼容导入入口，保留它们即可。

## 网页分批更新

1. 打开 https://github.com/JIACHEN0611/SARBF-Net ，进入 **Code** 页，确认当前分支为 main。
2. 在仓库根目录点击 **Add file → Upload files**。
3. 从解压后的 `SARBF-Net` 中拖入一批完整文件夹或根目录文件，保留层级。可先上传 `model`，下一批上传 `experiments` 和 `tools`，再上传 `tests`、`scripts`，最后上传文档及根目录文件。每批查看待上传列表中的相对路径，例如应出现 `model/backbones/DLinear.py`。
4. 相同路径的文件会更新为新内容。每批等待文件传完，再点 **Commit changes**；提交说明可写 `Integrate SAER transfer experiments`。直接更新 main 时选 **Commit directly to the main branch**；若页面要求分支/PR，就使用新分支并合并全部批次后再检查。
5. 每一批完成后回到仓库根目录继续下一批。不要在网页的 model 文件夹里再拖入整个 model，否则会出现 `model/model`。

GitHub 网页一次最多上传 100 个文件，单文件上限 25 MiB；本包的源码文件均远小于该限制。数量超限时减小当前批次。规则来源：[GitHub 官方上传说明](https://docs.github.com/en/repositories/working-with-files/managing-files/adding-a-file-to-a-repository)。

`.github`、`.gitignore`、`.gitattributes` 也要保留，其中 `.github/workflows/tests.yml` 本次有更新。在 Windows 中开启“查看 → 显示 → 隐藏的项目”有助于检查。若网页拖拽不接受以点开头的文件，可以在已有文件页面点铅笔编辑，将本地文件内容完整替换后提交；`.github` 文件夹仍从根目录拖入。

## 核对更新清单

随包的 `SARBF-Net_Update_Changes.json` 记录相对旧版 v3 源码的 `added`（新增）、`modified`（修改）、`deleted`（删除）三类路径。网页上传新增/修改文件不会自动删除旧文件。若 `deleted` 非空，逐个打开对应旧文件，使用文件菜单 **Delete file** 并提交；若为空则无需删文件。不要根据“名字像旧模型”自行删除兼容文件。

清单基于旧版 v3 源码。GitHub 上另外添加且清单未列出的文件，需要结合实际用途判断；本说明无法核实远程私有仓库的额外改动。资产分卷不属于源码删除清单。

若使用 GitHub Desktop，可以先克隆远程仓库，在本地克隆目录按清单替换源码，再对照 Changes 后提交并 Push origin。不要把任何 `.git` 目录从其他文件夹复制进去。

## 上传后的检查

- 仓库根目录仍直接显示 README.md 和 run.py，中英文 README 能互相跳转。
- 存在 `model/components`、`model/backbones`、`model/saer_adapters` 和 `tools/saer_transfer`。
- **Actions → CPU checks** 在所有批次上传后通过；分批过程中出现缺模块可以等文件全部提交后重跑。Actions 不执行真实数据的完整训练。
- **Releases → v1.0.0** 中四个 `SARBF-Net_Assets_v1.7z.00x` 附件仍在。自动生成的 Source code ZIP/tar.gz 是旧标签源码快照，与当前 main 分支可以不同。

本次没有自动修改远程仓库、可见性、标签或 Release。纯源码 ZIP 较小是正常的：约 3 GB 的 Q、权重和实验材料仍由原有 Release 提供。
