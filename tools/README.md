# 开发工具

在仓库根目录、安装项目的 Python 环境中运行 `python -m tools.工具名`。

| 工具 | 用途 |
| --- | --- |
| `download_models qwen8b` | 下载固定 revision 的 Qwen8B；`qwen` 为2B |
| `make_synthetic` | 生成不含目标语义的绕行场景与 CPU 配置 |
| `probe_vlm_grounding IMAGE --target TEXT --category CLASS` | 单图真实 VLM 定位与复核；不声称三维/多视图确认 |
| `inspect_vlm_grounding_crop RUN` | 单图诊断结果的上下文裁剪分类 |
| `audit_auto_views RUN --output artifacts/audit.json` | 检查 PREPARED 运行的128相机、安全位置、RGB及深度 |
| `audit_offline RUN` | 检查已完成离线运行的接受、掩码及投影一致性 |
| `audit_vlm_run RUN SOURCE` | 审计复用缓存的纯 VLM 运行及文件指纹 |
| `audit_run RUN [RUN ...]` | 检查移动轨迹、整段碰撞与多视图条件 |
| `compare_coverage --baseline RUN --run RUN` | 同预算下，对历史真实观察做覆盖比较 |
| `validate_gaussian_projection RUN` | 校验完整原高斯字段与导出，并执行真实 gsplat 对比渲染 |
| `validate_viewer_browser URL --output artifacts/viewer.json` | 可选 Chrome/CDP 回放测试，需 Chrome 与 aiohttp |
| `record_versions` | 将当前环境、freeze和已安装包版本写入忽略的 artifacts/ |

各工具的 `--help` 显示参数。审计/诊断可能向指定运行目录追加诊断文件；
对需保留原状的历史结果，先复制到临时目录。浏览器工具默认使用 Linux 的
`/usr/bin/google-chrome`，不属于运行搜索的必要依赖。`record_versions` 不覆盖
仓库的历史实测版本表。
