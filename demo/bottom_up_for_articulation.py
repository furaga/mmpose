"""
python demo/bottom_up_for_articulation.py \
    configs/body/2d_kpt_sview_rgb_img/associative_embedding/crowdpose/higherhrnet_w32_anim_512x512_udp.py \
    checkpoints/best_AP_epoch_120_20211003.pth \
    --img-path data/rendered.png \
    --out-path output.json
"""

from argparse import ArgumentParser
import json
from pathlib import Path


def main():
    parser = ArgumentParser()
    parser.add_argument("pose_config", help="Config file for detection")
    parser.add_argument("pose_checkpoint", help="Checkpoint file")
    parser.add_argument(
        "--img-path", type=Path, help="Path to an image file or a image folder."
    )
    parser.add_argument("--out-path", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0", help="Device used for inference")
    parser.add_argument(
        "--kpt-thr", type=float, default=0.5, help="Keypoint score threshold"
    )
    parser.add_argument(
        "--pose-nms-thr", type=float, default=0.9, help="OKS threshold for pose NMS"
    )

    args = parser.parse_args()

    # import 遅いのでなるべく後のタイミングでimport
    from mmpose.apis import inference_bottom_up_pose_model, init_pose_model
    from mmpose.datasets import DatasetInfo

    # build the pose model from a config file and a checkpoint file
    pose_model = init_pose_model(
        args.pose_config, args.pose_checkpoint, device=args.device.lower()
    )

    dataset = pose_model.cfg.data["test"]["type"]
    dataset_info = pose_model.cfg.data["test"].get("dataset_info", None)
    dataset_info = DatasetInfo(dataset_info)

    # optional
    return_heatmap = False
    output_layer_names = None

    pose_results, _ = inference_bottom_up_pose_model(
        pose_model,
        str(args.img_path),
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

    d = {
        "people": people,
    }

    with open(args.out_path, "w") as f:
        json.dump(d, f)

    print("Saved", str(args.out_path))


if __name__ == "__main__":
    main()
