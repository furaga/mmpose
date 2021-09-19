import argparse
import csv
import os
import shutil
import time
import sys
from pathlib import Path
import json
import numpy as np

from glob import glob


def parse_args():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument(
        "--list_path", type=Path, default=Path("data/anim_data_list.txt")
    )
    parser.add_argument("--out_dir", type=Path, default=Path("out"))
    args = parser.parse_args()
    return args


def load_data_list(list_path):
    data_list = []
    with open(args.list_path) as f:
        line = f.readline()
        root = "."
        while line:
            if line.startswith("DATA_ROOT="):
                root = line.split("=")[1].strip()
            else:
                tokens = line.split(",")
                if len(tokens) == 2:
                    annot_path = Path(root) / tokens[0].strip()
                    img_dir = Path(root) / tokens[1].strip()
                    data_list.append((annot_path, img_dir))
            line = f.readline()
    return data_list


def make_crowdpose_json(annotations):
    with open("data/crowdpose_base.json") as f:
        d = json.loads(f.read())

    width, height = 1920, 1080
    annot_id = 0
    for ann in annotations:
        d["images"].append(
            {
                "file_name": ann["image_name"],
                "id": ann["image_id"],
                "height": height,
                "width": width,
                "crowdIndex": 0,  # 不要？
            }
        )

        for person in ann["people"]:
            d["annotations"].append(
                {
                    "num_keypoints": 0,
                    "iscrowd": 0,
                    "keypoints": 0,
                    "image_id": ann["image_id"],
                    "bbox": 0,
                    "category_id": 1,
                    "id": annot_id,
                }
            )
            annot_id += 1
    return d


def main(args):
    args.out_dir.mkdir(exist_ok=True)
    data_list = load_data_list(args.list_path)

    all_annotations = []
    image_counter = 0
    for annot_path, img_dir in data_list:
        with open(annot_path) as f:
            d = json.loads(f.read())
        annots = d["annotations"]
        for ann in annots:
            image_name = ann["image_name"]
            image_id = image_counter

            img_path = img_dir / image_name
            out_img_path = args.out_dir / f"{image_id:08d}.jpg"
            shutil.copy(img_path, out_img_path)

            people = ann["people"]
            new_ann = {
                "image_name": out_img_path.name,
                "image_id": image_id,
                "people": people,
            }
            all_annotations.append(new_ann)
            image_counter += 1

            print(new_ann)
            exit()

    d = make_crowdpose_json(all_annotations)
    with open(args.out_dir / f"mmpose_anim_{args.type}.json", "w") as f:
        json.dump(d, f)


if __name__ == "__main__":
    args = parse_args()
    main(args)
