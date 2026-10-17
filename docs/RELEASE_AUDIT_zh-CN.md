# 源码整合记录（2026-10-17）

本次将 SARBF-Net 的有效实现迁入 `model/components/`，保留旧导入桥及历史权重字段；加入五骨干 SAER 迁移实验、隔离的数据管线与结果汇总入口；更新中英文说明、第三方许可文本和 CPU 检查。

主模型与插件分别对照原始源码，检查三个种子下的初始化、前向、梯度、一次优化和严格权重加载。配置对照覆盖迁移矩阵的全部 720 项预设。实际命令和状态见 [VALIDATION_REPORT.json](VALIDATION_REPORT.json)。这些检查不代表重新完成真实数据训练或全部已发布权重复评。

源码包不包含数据、Q 或训练权重。已有实验材料继续保存在 v1.0.0 Release，原清单与分卷保持不变。源文件哈希更新为当前实现，历史模型哈希保存在 `legacy/ORIGINAL_MODEL_LAYER_SHA256.json`。

此前的准备包操作与检查记录归档于 [旧版说明](legacy/PRE_INTEGRATION_RELEASE_AUDIT_zh-CN.md)，其中的数据缺失状态和旧测试数量仅对应当时版本。
