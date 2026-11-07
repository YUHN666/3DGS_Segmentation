# 可选 SplatPlan 后端

默认 `occupancy_astar_3d` 不需要 Splat-Nav。离线128图只检查采集位置，不执行路径规划。

需要移动相机的上游走廊与样条优化时：

```bash
python -m pip install -e '.[navigation]'
git clone https://github.com/chengine/splatnav.git third_party/splatnav
git -C third_party/splatnav checkout e996e229daca4e25703a13243541371290a6f7f8
```

在已经验证边界/起点的配置中设置：

```json
{
  "navigation": {"backend": "splatplan_checked_seed"},
  "splatplan": {
    "repo": "third_party/splatnav",
    "device": "cuda:0",
    "vmax": 0.5,
    "amax": 0.5,
    "fallback_backend": "occupancy_astar_3d"
  }
}
```

将这些字段合入完整配置，保留原有 navigation 的碰撞参数；该片段不是完整配置。
上游真实 `SplatPlan.generate_path` 负责走廊和 Bezier 优化，适配器仅替换为经过
验证的 A* 初始路径，随后独立检查全部线段。显式回退时日志记录原因和实际后端；
A* 结果不会标为 SplatPlan。没有启用 Splat-Loc 或语义查询。

此适配器此前已在本地真实 CUDA 环境运行；开源整理仅回归通用控制与导航测试，
没有重新执行上游 CUDA 优化。上游依赖较多，建议首先使用默认后端。
