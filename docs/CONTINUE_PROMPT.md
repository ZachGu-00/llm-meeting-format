请接管这个仓库中的 Voices / Arguments v3 实验，继续使用原 GPU 服务器。

先读取 AGENTS.md、handoff/STATE.json、reports/PAUSE_SUMMARY.md 和 docs/CLOUD_SETUP.md，检查 SSH 连通性、原 screen、五个暂停进程、冻结的 E4 score plan 及源文件状态。所有实验都使用服务器现有环境，不要在 Codex 云容器运行 GPU 推理，不要重建第二个队列。

我现在明确授权恢复已暂停的 E4 原进程。核验通过后，可执行 `python3 cloud/control.py resume --confirm-resume`。若进程不存在，先检查断点再提出可执行恢复方案，不要重跑或覆盖已完成 E0/E1/E2/位置检查。

目标是完成冻结的六模型×六议题 E4a/b/c 单轮，E4a/b 六轮蒙特卡洛、H3，以及结果报告和可核验的阶段备份。E3 按要求跳过，E4d 可选。不要让小模型自己报告概率，不要按效应是否漂亮修改阈值或模型名单。保留原始读数与全部不确定、负结果。

继续使用 screen，native BF16/eager/batch16。恢复后观察实际进程与输出增长，定期保存产物；本机 Windows 备份器已经停止，需安排云端备份。只有全部规定任务和报告、原始产物校验完成，才宣告本阶段完成。

