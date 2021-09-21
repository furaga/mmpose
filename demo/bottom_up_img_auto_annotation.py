# Copyright (c) OpenMMLab. All rights reserved.

"""
python demo/bottom_up_img_auto_annotation.py \
    configs/body/2d_kpt_sview_rgb_img/associative_embedding/crowdpose/higherhrnet_w32_anim_512x512_udp.py \
    checkpoints/best_AP_epoch_80.pth \
    --img_dir /mnt/d/workspace/anim/screenshots/RAW/rezero \
    --out_dir /mnt/d/workspace/anim/screenshots/posed/rezero/pose \
    --sample-span 30
"""

import os
import os.path as osp
import warnings
from pathlib import Path
from argparse import ArgumentParser
import json
import shutil

import mmcv

from mmpose.apis import inference_bottom_up_pose_model, init_pose_model, vis_pose_result
from mmpose.datasets import DatasetInfo


def main():
    """Visualize the demo images."""
    parser = ArgumentParser()
    parser.add_argument("pose_config", help="Config file for detection")
    parser.add_argument("pose_checkpoint", help="Checkpoint file")
    parser.add_argument(
        "--img_dir", type=str, help="Path to an image file or a image folder."
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="",
        help="Root of the output img file. "
        "Default not saving the visualization images.",
    )
    parser.add_argument("--device", default="cuda:0", help="Device used for inference")
    parser.add_argument(
        "--kpt-thr", type=float, default=0.5, help="Keypoint score threshold"
    )
    parser.add_argument(
        "--pose-nms-thr", type=float, default=0.9, help="OKS threshold for pose NMS"
    )

    parser.add_argument(
        "--sample-span", type=int, default=1, help="OKS threshold for pose NMS"
    )

    args = parser.parse_args()

    Path(args.out_dir).mkdir(exist_ok=True)

    # prepare image list
    image_list = [
        osp.join(args.img_dir, fn)
        for fn in os.listdir(args.img_dir)
        if fn.lower().endswith((".png", ".jpg", ".jpeg", ".tiff", ".bmp"))
    ]

    # build the pose model from a config file and a checkpoint file
    pose_model = init_pose_model(
        args.pose_config, args.pose_checkpoint, device=args.device.lower()
    )

    dataset = pose_model.cfg.data["test"]["type"]
    dataset_info = pose_model.cfg.data["test"].get("dataset_info", None)
    if dataset_info is None:
        warnings.warn(
            "Please set `dataset_info` in the config."
            "Check https://github.com/open-mmlab/mmpose/pull/663 for details.",
            DeprecationWarning,
        )
        assert dataset == "BottomUpCocoDataset"
    else:
        dataset_info = DatasetInfo(dataset_info)

    # optional
    return_heatmap = False

    # e.g. use ('backbone', ) to return backbone feature
    output_layer_names = None

    # process each image
    annotations = []

    image_list = image_list[:: args.sample_span]
    for i, image_name in enumerate(image_list):
        if i % 10 == 0:
            print(
                f"[{i+1}/{len(image_list)}] {Path(image_name).name}"
                f" ({len(annotations)} annotations found)"
            )
        # test a single image, with a list of bboxes.
        pose_results, returned_outputs = inference_bottom_up_pose_model(
            pose_model,
            image_name,
            dataset=dataset,
            dataset_info=dataset_info,
            pose_nms_thr=args.pose_nms_thr,
            return_heatmap=return_heatmap,
            outputs=output_layer_names,
        )

        people = []
        for res in pose_results:
            kps = res["keypoints"]
            n_kps = 0
            keypoints = []
            for x, y, score in kps:
                keypoints += [int(x), int(y), 1]
                if score > args.kpt_thr:
                    n_kps += 1
            if n_kps > 4:
                people.append(
                    {
                        "person_name": f"P{len(people)}",
                        "person_id": len(people),
                        "keypoints": keypoints,
                    }
                )

        out_img_path = Path(args.out_dir) / Path(image_name).name
        shutil.copy(image_name, out_img_path)
        annotations.append(
            {
                "image_name": Path(image_name).name,
                "image_id": i,
                "people": people,
            }
        )

    d = {
        "annotations": annotations,
    }

    out_json_path = Path(args.out_dir) / "keypoints.json"
    with open(out_json_path, "w") as f:
        json.dump(d, f)


if __name__ == "__main__":
    main()
