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
    parser.add_argument("--type", type=str, default="train")
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


def get_kps_bbox(person, width, height):
    keypoints = np.array(person["keypoints"]).astype(int)
    keypoints = keypoints.reshape((14, 3))
    n_keypoints = 0
    new_keypoints = []
    for x, y, w in keypoints:
        valid = True
        if w <= 0:
            valid = False
        if x < 0 or width <= w or y < 0 or height <= y:
            valid = False
        if valid:
            new_keypoints.append((x, y, w))
            n_keypoints += 1
        else:
            new_keypoints.append((0, 0, 0))
    new_keypoints = np.array(new_keypoints)
    n_keypoints -= 2

    xmin = np.min(new_keypoints[:, 0])
    xmax = np.max(new_keypoints[:, 0])
    ymin = np.min(new_keypoints[:, 1])
    ymax = np.max(new_keypoints[:, 1])
    cx = (xmin + xmax) / 2
    cy = (ymin + ymax) / 2
    dx = cx - xmin
    dy = cy - ymin
    ratio = 1.1
    bbox = [
        float(cx - dx * ratio),
        float(cx + dx * ratio),
        float(cy - dy * ratio),
        float(cy + dy * ratio),
    ]

    new_keypoints = [int(v) for v in new_keypoints.ravel()]
    return n_keypoints, new_keypoints, bbox


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
            n_keypoints, keypoints, bbox = get_kps_bbox(person, width, height)
            d["annotations"].append(
                {
                    "num_keypoints": n_keypoints,
                    "iscrowd": 0,
                    "keypoints": keypoints,
                    "image_id": ann["image_id"],
                    "bbox": bbox,
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
            #shutil.copy(img_path, out_img_path)

            people = ann["people"]
            new_ann = {
                "image_name": out_img_path.name,
                "image_id": image_id,
                "people": people,
            }
            all_annotations.append(new_ann)
            image_counter += 1

    d = make_crowdpose_json(all_annotations)
    out_json_path = args.out_dir / f"mmpose_anim_{args.type}.json"
    with open(out_json_path, "w") as f:
        json.dump(d, f)
        print("Saved", out_json_path)


if __name__ == "__main__":
    args = parse_args()
    main(args)
