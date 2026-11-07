# Third-party components

The MIT license in this repository applies to this project's code. Dependencies,
model weights, datasets and scenes retain their respective licenses. They are
installed/downloaded separately; their source trees and weights are not bundled
in this repository or its wheel.

| Component | Use | Tested version/revision | Upstream license |
| --- | --- | --- | --- |
| [gsplat](https://github.com/nerfstudio-project/gsplat) | Gaussian rendering | 1.4.0 | Apache-2.0 |
| [Transformers](https://github.com/huggingface/transformers) | Local Qwen inference | 4.57.6 | Apache-2.0 |
| [Qwen3-VL](https://github.com/QwenLM/Qwen3-VL) | RGB grounding and review | Qwen3-VL-8B-Instruct snapshot `0c351dd01ed87e9c1b53cbc748cba10e6187ff3b` | Consult the model's upstream model card and license |
| [SAM 2](https://github.com/facebookresearch/sam2) | Box-prompt segmentation | `2b90b9f5ceec907a1c18123530e92e794ad901a4` | Apache-2.0 |
| [Splat-Nav](https://github.com/chengine/splatnav) | Optional SplatPlan geometry backend | `e996e229daca4e25703a13243541371290a6f7f8` | MIT, copyright 2025 chengine |
| [VLFM](https://github.com/rai-opensource/vlfm) | Exploration strategy reference; no runtime import | inspected `584ed56008754fde7997d904983607def8328322` | MIT, copyright 2023–2024 Boston Dynamics AI Institute LLC |

The SplatPlan adapter imports upstream corridor/spline code from a separately
installed checkout. It provides a checked A* seed and validates returned motion;
it does not vendor or relicense Splat-Nav. The default planner is the independently
named `occupancy_astar_3d` backend.

The 16-position/eight-yaw camera schedule follows the layout convention used by
the author's `gs_sam2_instances` project. Camera expansion uses ordinary pinhole
camera and orthonormal-basis mathematics; no private project is required at
runtime or by the distributed tests.

Scene PLY files and saved search evidence are not distributed. Users supply their
own scenes and model weights. Runtime `environment.json` records installed
versions and available local Git revisions; a package version does not establish
a Git commit for a preinstalled wheel. See `requirements-tested.txt` for the
historically tested environment, including optional packages.
