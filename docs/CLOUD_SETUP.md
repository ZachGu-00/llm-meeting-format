# 创建 Codex 云环境并接管

1. 在 Codex 的环境设置中新建环境，选择此私有 GitHub 仓库和 `main` 分支。若列表没有仓库，先在 GitHub 连接中授予该仓库访问权限。
2. 添加 Secret：名称为 `VOICES_SSH_PRIVATE_KEY`，值为本地专用私钥文件的完整内容。私钥在用户电脑的 `Hypergraph/.private/codex_cloud_v3_ed25519`，不在 GitHub。其公钥已在当前服务器配置，禁止端口转发、代理转发和 PTY。
3. Setup script 填入 `bash cloud/setup.sh`。脚本只把 Secret 写到环境的 `~/.ssh/voices_v3_ed25519`，权限 0600；不下载模型、不安装 CUDA、不恢复实验。若环境仅提供其他凭证机制，按该界面的方式安全注入，并用 `VOICES_SSH_KEY_FILE` 指向私钥文件。
4. 允许 agent 阶段所需网络访问，然后启动一个会话，先执行 `python3 cloud/control.py status`。本机已验证专用 SSH 密钥，但新云环境尚未建立，因此该环境到 `157.107.19.62:16675` 的实际连通性尚未验证。
5. 读完状态与结果后，如用户明确要求继续，再执行 `python3 cloud/control.py resume --confirm-resume`。保持 screen 中原有进程和冻结配置；不要新建第二条推理队列。

官方环境文档说明，传统 Codex Cloud 的 Secret 只在 setup 阶段提供，而 agent 阶段网络默认关闭，环境使用 HTTP/HTTPS 代理。因而开启 Internet 不能单独证明非标准端口 SSH 一定可用；需以上一步的实际连接为准。若连接被阻止，先解决环境支持的出站连接方式，不能把连接失败当作服务器实验终止，也不能绕过平台网络限制。[官方环境文档](https://learn.chatgpt.com/docs/environments/cloud-environment)；[网络设置](https://learn.chatgpt.com/docs/cloud/internet-access)。具体 UI 以账号当前提供的环境类型为准。

专用公钥指纹：`SHA256:eSOsxL9eJ36q/k+CPiHCaDbNqWqzCXsh8vtJH2Z1bYU`。`cloud/known_hosts` 保存了通过现有 SSH 连接核对的服务器公钥，脚本使用严格主机校验。

## 给新会话的任务

可复制 `docs/CONTINUE_PROMPT.md`。如果只是检查环境，不要复制其中的恢复授权语句。

## 结果与备份

`fetch-reports` 对下载的报告做大小与 SHA256 校验，保存在被 Git 忽略的 `artifacts/`。它不下载权重，也不声称已备份全部原始数据。原始输入、所有测量结果、已有归档都在 GPU 服务器；E0–E2、位置补充检查和 E4 暂停快照也已备份到用户电脑。

原 Windows 自动备份进程已停止。云端接管后须安排阶段备份：可复用 `experiment/setup/backup_e4_monitor_v3.py` 及其配套 helper，在外部工作目录建立源文件依赖镜像再运行；不得把带 Windows 绝对路径的旧本地 monitor 状态直接当作云端已完成备份。也可在服务器调用其已审阅的 closed-artifact 归档逻辑，下载后逐一校验归档、成员文件及既有依赖。完整验收须包含 E4 报告和原始产物，不能只看报告标题。

## 异常恢复

若五个原进程仍存在且为 T 状态，使用受保护的 resume 命令。若进程缺失、服务器重启或计划哈希变化，不要执行裸 `kill -CONT` 或盲目重建队列；先检查原日志和未完成输出。当前 score plan 尚未闭合，不能直接套用要求完整 score plan 的 `--score-plan` 恢复路径。保留 12,000 行暂停快照和旧 history，由接管代理做明确的续写/恢复审查。

