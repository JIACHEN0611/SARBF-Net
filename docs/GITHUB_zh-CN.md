# 把 SARBF-Net 放到 GitHub：Windows 中文步骤

你截图中已经登录 GitHub 仪表盘，可以开始建仓库。下面推荐 GitHub Desktop；
它能同时上传目录和隐藏文件，后续更新也比较方便。只需要选一种方式。

## 方式一：GitHub Desktop（推荐）

1. 下载并解压 `SARBF-Net_GitHub_Ready.zip`，得到 `SARBF-Net` 文件夹。
   进入后应直接看见 `README.md`、`run.py`、`model`、`layers`，不要把压缩包本身当源码上传。
2. 从 https://desktop.github.com/ 安装 GitHub Desktop，启动后选择
   **Sign in to GitHub.com**，在浏览器授权登录你已经注册的账号。
3. 在 Desktop 选择 **File → New repository**。
   - **Name**：`SARBF-Net`
   - **Description**：`Selective Anchored Retrieval & Balanced Feedback Network for multivariate time series forecasting.`
   - **Local path**：选一个新位置，例如 `D:\GitHub`，Desktop 会建立 `D:\GitHub\SARBF-Net`。
   - 不需要生成额外 README、.gitignore 或许可证；本包已经有说明与配置。
4. 把下载解压后的 `SARBF-Net` **里面的所有文件和文件夹**复制到 Desktop 新建的仓库目录。
   在资源管理器开启“查看 → 显示 → 隐藏的项目”，确保 `.github`、`.gitignore`、
   `.gitattributes` 一起复制。不要把整个外层目录再套一层。
5. 回到 Desktop，左侧会显示文件变更。底部 **Summary** 填
   `Prepare SARBF-Net source release`，点击 **Commit to main**。
6. 点击上方 **Publish repository**：名称保持 `SARBF-Net`。第一次整理可保留
   **Keep this code private**；准备对外公开时再更改可见性。若你现在就决定公开，取消该勾选。
   再点击 **Publish repository**。
7. 点击 **View on GitHub** 检查页面。根目录应直接显示 `README.md`、`run.py` 和模型目录，
   页面下方会渲染英文 README，中文说明有入口。

如果你已经在网页建好了同名仓库，使用 **File → Clone repository** 克隆那个仓库，
再把源码内容复制进去、Commit、Push origin；不要重复 Publish 第二个同名仓库。

日后改代码后：Desktop 显示变更 → 写 Summary → Commit to main → Push origin。

## 方式二：网页上传（不装软件）

1. 打开 https://github.com/new ，或在截图右上角 **＋ → New repository**。
2. 仓库名填 `SARBF-Net`，描述用上面的英文描述。先选 Private 或按你的意愿选 Public。
3. **不要勾选 Add a README，不额外生成 .gitignore，不选择全仓库许可证**，避免与包内文件冲突。
4. 点击 **Create repository**。空仓库页面点击 **uploading an existing file**；
   已有文件的仓库使用 **Add file → Upload files**。
5. 打开解压后的 `SARBF-Net` 文件夹，把**里面的内容**拖到上传区域，不是整个外层目录。
   GitHub 网页一次最多上传 100 个文件，目录中的子文件也计数。当前准备包文件已超过
   这一限制，请分批上传：先上传 `model`、`layers`、
   `data_provider`、`experiments`、`utils` 并提交，再上传其余目录与根目录文件。
6. 每批填写提交信息，例如 `Add SARBF-Net model and runtime`，点击页面上的
   **Commit changes** 或 **Propose changes**，按页面引导完成提交。
7. 确保隐藏文件 `.gitignore`、`.gitattributes` 和 `.github/workflows/tests.yml` 已上传。
   网页上传不应用 `.gitattributes` 的换行规则，维护代码优先使用 Desktop/Git。

网页上传没有自动解压功能。只上传 zip 会得到一个压缩包文件，不会展示完整代码树。

## 方式三：Git 命令（已安装 Git 时）

先在网页建一个空仓库，不初始化 README/许可证，然后进入本地解压后的源码目录，
在 Git Bash 执行。下方 `YOUR_GITHUB_USERNAME` 必须替换成你的实际 GitHub 用户名。

```bash
git init -b main
git add .
git commit -m "Prepare SARBF-Net source release"
git remote add origin https://github.com/YOUR_GITHUB_USERNAME/SARBF-Net.git
git push -u origin main
```

如果 Git 提示缺少身份，按提示设置你自己的 `user.name` 与 `user.email` 后重试。
HTTPS 推送通过 Git Credential Manager/浏览器授权登录，不要把账号密码写入源码。
如果已有 origin 或仓库已经初始化，请先查看 `git remote -v` 与 `git status`，
不要反复执行初始化或强制推送；已有网页提交时用克隆方式更容易。

## 上传后需要补的真实复现资产

要按新的要求把 Q、最优权重和真实结果一起提交，请优先完成
[ASSETS_zh-CN.md](ASSETS_zh-CN.md) 的服务器收集及 Git LFS 步骤。
下面网页方式仅适用于小体积的源码准备包；完整资产包请用 Git/Desktop。

本次压缩包包含完整模型源代码和运行工具，但原包不包含下面这些实验材料：

| 项目 | 当前情况 | 下一步 |
| --- | --- | --- |
| 数据集 | 未包含 | 使用原实验数据并记录来源/版本 |
| Q 变换矩阵 | 未包含，原生成程序也未包含 | 在原服务器用 `tools.sarbf_net.export_q` 导出 |
| 训练权重、完整结果 JSON | 未包含 | 从原实验目录整理，需发布时作为单独附件 |
| 原 GPU 环境 | 未包含 | 在原环境导出 pip freeze 和 GPU 信息 |
| 第三方许可与来源锁 | 只有部分源码内声明 | 核对继承代码来源与许可证，保留必要文本 |
| 论文引用信息 | 未提供最终作者/题目/DOI | 发表或预印本确定后更新 CITATION.md |

真实 Q 文件导出命令见 [DATA.md](DATA.md)。矩阵导出 zip 可以在仓库主页的
**Releases → Draft a new release** 中作为附件上传，例如标签 `v1.0.0`。
不要为了上传而把数据、权重、服务器日志都塞进 Git 历史。

仓库已经带 GitHub Actions 的 CPU 检查配置。推送后在 **Actions** 中查看是否通过；
绿色检查表示代码测试通过，不表示论文指标已在 GitHub 上复现。

## 官方操作参考

- 新建仓库：https://docs.github.com/en/repositories/creating-and-managing-repositories/creating-a-new-repository
- Desktop 发布：https://docs.github.com/en/desktop/adding-and-cloning-repositories/adding-an-existing-project-to-github-using-github-desktop
- 网页上传与文件数量限制：https://docs.github.com/en/repositories/working-with-files/managing-files/adding-a-file-to-a-repository
