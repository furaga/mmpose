"""
$ python tools/dataset/unity_to_crowdpose.py \
    --list_path data/anim_data_list_train.txt \
    --out_dir /mnt/h/data/mmpose/anim/train \
    --type train

$ python tools/dataset/unity_to_crowdpose.py \
    --list_path data/anim_data_list_test.txt \
    --out_dir /mnt/h/data/mmpose/anim/test \
    --type test

$ python tools/dataset/unity_to_crowdpose.py \
    --list_path data/tmp.txt \
    --out_dir /mnt/h/data/mmpose/anim/tmp \
    --type test
"""

import argparse
import cv2
import shutil
from pathlib import Path
import json
import numpy as np
from glob import glob
import random


def parse_args():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument(
        "--list_path", type=Path, default=Path("data/anim_data_list.txt")
    )
    parser.add_argument("--bg_dir", type=Path, default=Path("data/background"))
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
                    data_list.append((annot_path, img_dir, "real"))
                if len(tokens) == 3:
                    annot_path = Path(root) / tokens[0].strip()
                    img_dir = Path(root) / tokens[1].strip()
                    data_list.append((annot_path, img_dir, tokens[2].strip()))
            line = f.readline()
    return data_list


skeleton = [
    [12, 13],
    [13, 0],
    [13, 1],
    [0, 2],
    [2, 4],
    [1, 3],
    [3, 5],
    [13, 7],
    [13, 6],
    [7, 9],
    [9, 11],
    [6, 8],
    [8, 10],
]


def get_kps_bbox(person, width, height):
    def _is_valid(x, y, w):
        if w <= 0:
            return False
        if x < 0 or width <= w or y < 0 or height <= y:
            return False
        return True

    G = {}
    for s, t in skeleton:
        G.setdefault(s, []).append(t)
        G.setdefault(t, []).append(s)

    keypoints = np.array(person["keypoints"]).astype(int)
    keypoints = keypoints.reshape((14, 3))
    n_keypoints = 0
    new_keypoints = []
    for i, (x, y, w) in enumerate(keypoints):
        valid = True
        if not _is_valid(x, y, w):
            valid = False
            for j in G[i]:
                x2, y2, w2 = keypoints[j]
                # 隣接する関節がvalidなら有効
                if _is_valid(x2, y2, w2):
                    valid = True
                    break
        if valid:
            new_keypoints.append((x, y, w))
            n_keypoints += 1
        else:
            new_keypoints.append((0, 0, 0))
    new_keypoints = np.array(new_keypoints)
    # n_keypoints -= 2

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
        has_any = False
        for person in ann["people"]:
            n_keypoints, keypoints, bbox = get_kps_bbox(person, width, height)
            if n_keypoints > 0:
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
                has_any = True
        if has_any:
            d["images"].append(
                {
                    "file_name": ann["image_name"],
                    "id": ann["image_id"],
                    "height": height,
                    "width": width,
                    "crowdIndex": 0,  # 不要？
                }
            )

    return d


def save_synth_image(img_path, bg_path, out_img_path):
    img = cv2.imread(str(img_path))
    bg = cv2.imread(str(bg_path))
    mask = cv2.inRange(img, (0, 250, 0), (0, 255, 0))
    mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    front = cv2.bitwise_and(img, cv2.bitwise_not(mask))
    back = cv2.bitwise_and(bg, mask)
    out_img = cv2.bitwise_or(front, back)
    cv2.imwrite(str(out_img_path), out_img)


def main(args):
    args.out_dir.mkdir(exist_ok=True, parents=True)
    data_list = load_data_list(args.list_path)

    all_bg_paths = glob(str(args.bg_dir / "*.jpg"))
    print(f"Found {len(all_bg_paths)} BG images")

    all_annotations = []
    image_counter = 0

    save_image_info = {}
    for annot_path, img_dir, type in data_list:
        if type == "synth":
            # 画像ごとにjsonファイルがあるので統合する
            all_json_paths = glob(str(Path(annot_path) / "*.json"))
            d = {"annotations": []}
            for json_path in all_json_paths:
                with open(json_path) as f:
                    _d = json.loads(f.read())
                    d["annotations"] += _d["annotations"]
        else:
            with open(annot_path) as f:
                d = json.loads(f.read())
        annots = d["annotations"]
        for ann in annots:
            image_name = ann["image_name"]
            image_id = image_counter

            img_path = img_dir / image_name
            out_img_path = args.out_dir / f"{image_id:08d}.jpg"

            img_info = []
            if type == "synth":
                n_bg = len(all_bg_paths)
                bg_path = all_bg_paths[random.randint(0, n_bg - 1)]
                img_info = [img_path, bg_path, out_img_path]
            else:
                img_info = [img_path, "", out_img_path]

            people = ann["people"]
            new_ann = {
                "image_name": out_img_path.name,
                "image_id": image_id,
                "people": people,
            }
            all_annotations.append(new_ann)
            save_image_info[new_ann["image_name"]] = img_info
            image_counter += 1

    print(f"{image_counter} images found.")

    d = make_crowdpose_json(all_annotations)
    print(f"# Images: {len(d['images'])}")
    print(f"# Annotations: {len(d['annotations'])}")

    out_json_path = args.out_dir / f"mmpose_anim_{args.type}.json"
    print(f"Saving {len(d['annotations'])} Annotation to", str(out_json_path))
    with open(out_json_path, "w") as f:
        json.dump(d, f)

    print("Saving images to", str(args.out_dir))
    for i, e in enumerate(d["images"]):
        info = save_image_info[e["file_name"]]
        if i % 50 == 0:
            print(f"[{i+1}/{len(d['images'])}] {info[0]}")
        if info[1] == "":
            shutil.copy(info[0], info[2])
        else:
            save_synth_image(info[0], info[1], info[2])

    print("Done")


if __name__ == "__main__":
    args = parse_args()
    main(args)
