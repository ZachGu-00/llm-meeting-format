# Voices / Arguments v3

当前论文的代码整理版见 [v1_experiment](v1_experiment/README.md)。该目录提供离线分析入口、冻结源代码和简短运行说明。

## 历史云端接管记录

**实验已按用户要求暂停在 E4 开始阶段。** 本仓库用于接管同一台 GPU 服务器；环境初始化不会恢复实验。

先读 [当前结果与暂停位置](reports/PAUSE_SUMMARY.md)，再读 [Codex 环境配置](docs/CLOUD_SETUP.md)。完整状态见 [STATE.json](handoff/STATE.json)。

- GPU：RTX PRO 5000 / 48 GB；服务器目录 `/workspace/voices_nonadditivity_v3`。
- E0/E1：各 10 个模型完成；E2：9 个适用模型完成；E3：按要求跳过。
- 补充位置平衡检查：10 个模型完成。
- E4：6 模型 × 6 议题；目前 0/36 个完整任务，第一个任务已写入 12,000/49,020 行。
- 原来的五个进程保留在 screen 中，以 SIGSTOP 暂停。服务器需要保持运行。
- 已删除四个非 E4 模型共 20.14 GB 权重；元数据、数据、结果和所选六模型全部保留。

创建环境时选择本仓库。设置 Secret `VOICES_SSH_PRIVATE_KEY`，Setup script 填 `bash cloud/setup.sh`。开启所需网络访问后，先运行：

```bash
python3 cloud/control.py status
```

只有用户明确要求继续后，才执行：

```bash
python3 cloud/control.py resume --confirm-resume
```

获取阶段报告：

```bash
python3 cloud/control.py fetch-reports
```

该命令只下载报告，不等同于完整原始数据备份。接管后的完整任务和备份要求见 `AGENTS.md`。

`experiment/` 为运行代码的精确快照，`reports/` 为已完成阶段结果，`handoff/` 为冻结计划、暂停和清理记录。模型、原始大文件和凭证均未上传。不要用此仓库覆盖正在暂停的服务器工作目录。

