"""
python tools/dataset/combine_crowdpose.py \
    --json_path1 /mnt/h/data/mmpose/anim/train/mmpose_anim_train.json \
    --json_path2 /mnt/h/data/mmpose/crowdpose/annotations/mmpose_crowdpose_trainval.json \
    --img_dir1 /mnt/h/data/mmpose/anim/train \
    --img_dir2 /mnt/h/data/mmpose/crowdpose/images \
    --out_dir /mnt/h/data/mmpose/anim_crowdpose/train
"""

import argparse
import shutil
from pathlib import Path
import json


def parse_args():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--json_path1", type=Path, required=True)
    parser.add_argument("--json_path2", type=Path, required=True)
    parser.add_argument("--img_dir1", type=Path, required=True)
    parser.add_argument("--img_dir2", type=Path, required=True)
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--name", type=str, default="combined")
    parser.add_argument("--type", type=str, default="train")
    args = parser.parse_args()
    return args


def main(args):
    args.out_dir.mkdir(exist_ok = True, parents=True)

    with open(args.json_path1) as f:
        d1 = json.loads(f.read())

    with open(args.json_path2) as f:
        d2 = json.loads(f.read())

    images = []
    annotations = []

    m_id = {}
    for i in range(len(d1["images"])):
        img_info = d1["images"][i]
        file_name = f"{len(images):05d}.jpg"
        # shutil.copy(
        #     args.img_dir1 / img_info["file_name"],
        #     args.out_dir / file_name,
        # )
        img_info["file_name"] = file_name
        m_id[img_info["id"]] = len(images)
        img_info["id"] = len(images)
        images.append(img_info)

    for i in range(len(d1["annotations"])):
        annot = d1["annotations"][i]
        annot["image_id"] = m_id[annot["image_id"]]
        annot["id"] = len(annotations)
        annotations.append(annot)

    m_id = {}
    for i in range(len(d2["images"])):
        img_info = d2["images"][i]
        file_name = f"{len(images):05d}.jpg"
        # shutil.copy(
        #     args.img_dir2 / img_info["file_name"],
        #     args.out_dir / file_name,
        # )
        img_info["file_name"] = file_name
        m_id[img_info["id"]] = len(images)
        img_info["id"] = len(images)
        images.append(img_info)

    for i in range(len(d2["annotations"])):
        annot = d2["annotations"][i]
        annot["image_id"] = m_id[annot["image_id"]]
        annot["id"] = len(annotations)
        annotations.append(annot)

    with open("data/crowdpose_base.json") as f:
        d = json.loads(f.read())
    d["images"] = images
    d["annotations"] = annotations

    out_json_path = args.out_dir / f"mmpose_{args.name}_{args.type}.json"
    print(f"Saving {len(d['annotations'])} Annotation to", str(out_json_path))
    with open(out_json_path, "w") as f:
        json.dump(d, f)

if __name__ == "__main__":
    args = parse_args()
    main(args)
