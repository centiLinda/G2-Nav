# [CoRL 2026] G2-Nav: Grounded and Guarded Vision-Language Costmaps for Robot Social Navigation

## Getting started
This code has been tested on ROS1 Noetic.
Install [RAM++](https://github.com/xinyu1205/recognize-anything), [GroundingDINO](https://github.com/IDEA-Research/GroundingDINO), and [SAM](https://github.com/facebookresearch/segment-anything).
In `detection.py`, modify all "/path_to_your_xxx" to point to your local checkpoints and config files.

`run_scand.launch` is the main launch file that calls each component in `/scripts` folder.
For each python script in `/scripts`, you may modify the shebang to point to different environments (very useful if you're using conda).
Otherwise you may change them all to the default `#!/usr/bin/env python3` if everything is in your base environment.

`vlm_local.py` currently calls Qwen model hosted locally via [vLLM](https://docs.vllm.ai/en/latest/getting_started/quickstart/) server.
Follow [OpenAI API](https://developers.openai.com/api/docs/quickstart) instructions on how to call its online models, and remember to set `OPENAI_API_KEY` in your environment.

## Experiments

After building and sourcing the workspace, you may run the experiment by:
```
roslaunch g2nav run_scand.launch
rosbag play /path_to_your_scand_rosbag.bag
```
Rviz and a dashboard window will spawn.
The dashboard also pauses when you pause the rosbag.
Below are some useful topics to visualize in Rviz:
* /tracked_obj_annotated_img/compressed: tracked objects
* /costmap_safe_viz: vision-language costmap
* /plan_path: planned robot trajectory

This `g2nav` package works with the [SCAND](https://www.cs.utexas.edu/~xiao/SCAND/SCAND.html) dataset.
Change the names of the subscribed topics to work with other datasets or physical robot platforms.
You also need to change the camera intrinsics `CAM_INTRINSICS` and camera height `CAM_HEIGHT`.

Our real-world experiment was conducted on a Go2-W robot dog with ROS2 system.
The ROS1-ROS2 migration should be quite straightforward, but feel free to raise an issue if you face any difficulty.


## Citation
If you find this work useful, please cite [G2-Nav: Grounded and Guarded Vision-Language Costmaps for Robot Social Navigation](https://arxiv.org/abs/2607.16956) ([pdf](https://arxiv.org/abs/2607.16956), [video](https://youtu.be/1tlZk-4JfeE)):

```bibtex
@misc{liao2026g2navgroundedguardedvisionlanguage,
      title={G2-Nav: Grounded and Guarded Vision-Language Costmaps for Robot Social Navigation}, 
      author={Yuwen Liao and Yihang Lan and Yizhuo Yang and Ruimeng Liu and Xinhang Xu and Shenghai Yuan and Lihua Xie},
      year={2026},
      eprint={2607.16956},
      archivePrefix={arXiv},
      primaryClass={cs.RO},
      url={https://arxiv.org/abs/2607.16956}, 
}
```
