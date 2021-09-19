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
    #parser.add_argument("--out_path", type=Path, required=True)
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


def main(args):
    data_list = load_data_list(args.list_path)
    for annot_path, img_dir in data_list:
        print(str(annot_path))
        print(str(img_dir))
        assert annot_path.exists(), str(annot_path)
        assert img_dir.exists(), str(img_dir) + str(img_dir.exists())


if __name__ == "__main__":
    args = parse_args()
    main(args)
