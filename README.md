# 3DGS 场景目标搜索（纯 VLM）

在已重建的 3D Gaussian Splatting 场景中，通过相机 RGB 搜索文字描述的物体，并把接受的目标标回原始高斯 PLY。支持服务器无显示器运行。

默认流程：**几何自动生成 16×8 相机 → gsplat 渲染 128 张图 → Qwen3-VL 定位与类别/属性复核 → SAM2 分割 → 高斯标记**。另保留移动相机搜索、路径规划和浏览器回放。

## 安装

测试环境为 Linux、Python 3.10、PyTorch 2.8.0+cu128、gsplat 1.4.0、Transformers 4.57.6。真实渲染需要 NVIDIA GPU、兼容的 CUDA 编译工具和 C++ 编译器；首次渲染会构建 gsplat 扩展，后续复用 `.cache/torch_extensions/`。

在仓库根目录执行，已有环境可直接激活：

```bash
conda activate conda_3_10
# 新机器也可先创建：conda create -n gs-search python=3.10 -y
# 先按本机 CUDA/驱动安装匹配的 PyTorch 和 torchvision，然后安装项目。
python -m pip install -e '.[plot,dev]'
python -m tools.download_models qwen8b
```

SAM2 用于语义接受之后的分割与高斯导出，需单独安装：

```bash
git clone https://github.com/facebookresearch/sam2.git third_party/sam2
git -C third_party/sam2 checkout 2b90b9f5ceec907a1c18123530e92e794ad901a4
SAM2_BUILD_CUDA=0 python -m pip install -e third_party/sam2
curl -L --fail --output models/sam2.1_hiera_base_plus.pt \
  https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_base_plus.pt
```

下载是显式联网步骤。运行时默认只读取本地模型；不需要付费服务或令牌。依赖和上游提交记录见 [第三方说明](THIRD_PARTY_NOTICES.md)；`requirements-tested.txt` 是实测环境记录，包含单独安装的 SAM2，不能作为跨平台锁文件直接安装。

`run.sh` 使用当前 Python。机器设置可复制 `.env.example` 为 `.env.local`：设置 Python 路径、可见 GPU、`GS_SEARCH_VLM_DEVICE`、`GS_SEARCH_SAM2_CHECKPOINT`。本地文件不提交。默认模板全部使用 `cuda:0`；双卡可用：

```bash
export CUDA_VISIBLE_DEVICES=0,1
export GS_SEARCH_VLM_DEVICE=cuda:1
./run.sh diagnose
```

显式 shell 设置优先；逻辑 `cuda:0/1` 对应可见卡列表的第一/第二张。

## 更换场景并搜索

支持 Graphdeco 3DGS PLY：XYZ、SH 颜色、log-scale、logit-opacity、wxyz 四元数。普通 XYZRGB 点云不够；原场景不会被覆盖。

```bash
./run.sh offline --scene /path/to/point_cloud.ply \
  --target '放在蓝色圆桌上的电脑' --category computer
```

`--target` 保留完整中文/英文描述；`--category` 是独立复核使用的英文类别，例如 `plate`、`placemat`、`computer`、`trash can`。没有 DINO 阶段，也没有虚构的 VLM 数值置信度：`score=null`。

默认世界上方向为 `[0,-1,0]`（InteriorGS 约定）。Z 朝上的场景需要 `--world-up 0 0 1`。PLY 本身不能可靠确定重力方向和米制尺度。自动采样只使用几何，选择安全自由位置；布局绑定 PLY 校验值及尺度。当前选择一个可行高度层，不保证多楼层、开放场景或小目标完整覆盖。

只生成相机，或只准备图像缓存：

```bash
./run.sh prepare-scene --scene /path/to/point_cloud.ply
./run.sh offline --scene /path/to/point_cloud.ply --prepare-only
```

前者输出 `PREPARED_CONFIG` 和位置报告；后者保存128张 RGB 与期望 Z 深度/alpha，状态 `PREPARED`，不加载 Qwen/SAM2。复用同场景的图像可换目标，无需再次渲染：

```bash
./run.sh offline --scene /path/to/point_cloud.ply \
  --target '绿色盘子' --category plate --reuse-run runs/PREVIOUS_RUN
```

复用只继承图像、相机与匹配的几何参数，重新生成目标判断。`--reuse-grounding` 仅用于目标、类别和模型设置完全相同的中断恢复。不同场景禁止复用。

## 配置与结果

公开模板为 `configs/offline.json` 和 `configs/search.json`。机器专用配置可保存在忽略的 `configs/local/` 中；生成的场景配置保存在各次运行目录。

| 配置段 | 内容 |
| --- | --- |
| `scene/camera/auto_views` | 场景、尺度、相机内参/上方向、自动位置及资源上限 |
| `renderer/vlm/grounding` | 设备、本地模型或服务、候选数、调用/生成预算 |
| `offline` | 时间、磁盘、复核候选数量、批量和报告设置 |
| `segmentation/localization/gaussian_projection` | SAM2、可见深度过滤和高斯导出 |
| `navigation/search/policy/association` | 移动搜索的碰撞、预算、探索与跨视图关联 |

`configs/` 层级内的相对路径从其上级项目目录解析；其他位置的配置从配置目录解析。SAM2 的 `model_config` 是上游 Hydra 配置标识。生成运行保存绝对实际路径便于诊断，结果目录不会提交 Git。

每次独立保存到 `runs/时间_类型_编号/`，包括实际配置/环境、RGB、框图、裁剪、原始模型回答、复核、掩码、相机、决策日志、计数/分项耗时、`report.json`、`index.html` 和图像总览。

有接受且定位成功的候选时，`gaussian_projection/时间_编号/` 包含：

- `scene_target_labels.ply`：完整原场景字段，加目标标签与观察支持数。
- `scene_target_highlighted.ply`：完整场景，将选中高斯的 SH 颜色改为红色。
- `target_gaussians.ply`：原始目标高斯子集；`target_gaussian_indices.npy` 为原 PLY 行索引。
- `candidate_gaussian_indices.npz`、`projection_report.json`：各观察支持及筛选/校验记录。

`FOUND` 是模型接受证据，离线模式允许单图接受；不是人工真值或跨视图实例确认。`INCOMPLETE` 只表示尚未确认，不能断言不存在。没有通过者时不导出目标 PLY。不同观察不因同类而自动合并。

## 渲染、规划、移动搜索和回放

无 GPU/模型的合成渲染和绕行规划：

```bash
python -m tools.make_synthetic
./run.sh render --config configs/synthetic.json
./run.sh plan --config configs/synthetic.json --goal 4.9 1.5 1.1
```

合成场景使用明确命名的 CPU 参考渲染与三维占据图 A*，用于几何验证。真实单视角渲染可使用 `prepare-scene` 输出的实际配置：

```bash
./run.sh render --config runs/PREPARE_RUN/config.json
./run.sh search --config runs/PREPARE_RUN/config.json \
  --target '白色垃圾桶' --category 'trash can' \
  --max-observations 16 --max-path-length 12 --max-seconds 300 --live
```

自动离线站点可能属于不同连通分量；移动搜索会验证整段运动，遇到不可达位置或非法起点报错，不瞬移。移动策略可用 `--exploration-policy coverage`；固定网格则按16×8计划执行。在线确认要求平移基线及 RGB 特征关联。`configs/search.json` 的占位边界需按场景配置后启用 `scene.navigation_enabled`，也可直接使用自动准备的实际配置。

```bash
./run.sh replay runs/RESULT
./run.sh serve runs/RESULT --port 8765
./run.sh project-ply runs/RESULT
```

`index.html` 可离线查看。服务默认监听 `127.0.0.1`；远程使用 SSH 端口转发。移动搜索支持 `--live` / `--record-video`，回放可 `--render-trajectory`；离线128图没有站点之间的连续运动视频。

## 验证与开发

```bash
python -m pytest -q
python -m tools.probe_vlm_grounding IMAGE.png --target '绿色盘子' --category plate
python -m tools.record_versions
```

[验证记录](VALIDATION.md) 区分真实场景/模型验证与模型替身测试。[开发工具](tools/README.md) 包含缓存审计和导出检查。[可选 SplatPlan 接入](docs/splatplan.md) 与默认 A* 后端分别命名。

语义接口仅接收 RGB、文字和 RGB 历史。完整几何用于导航/碰撞/相机采样；离线模式在语义接受后，按用户要求使用 SAM2 掩码与渲染期望深度筛选原高斯。透明混合与遮挡可能影响定位，输出只表示观测支持的可见表面，不保证物体完整几何。

模型、结果、场景、第三方检出、缓存和本地配置均由 `.gitignore` 排除；本地原文件保留。主项目采用 [MIT](LICENSE)，模型与数据遵守各自条款。
