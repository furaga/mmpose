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

def to_crowdpose(unity_ann):
    return unity_ann

def main(args):
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
            out_img_path = args.out_dir / f"{image_id:08d}"
            people = ann["people"]
            unity_ann = {
                "image_name": out_img_path.name,
                "image_id": image_id,
                "people": people,
            }
            new_ann = to_crowdpose(unity_ann)
            all_annotations.append(new_ann)
            image_counter += 1

            print(new_ann)
            exit()


if __name__ == "__main__":
    args = parse_args()
    main(args)
