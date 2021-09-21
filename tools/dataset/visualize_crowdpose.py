"""
$ python tools/dataset/visualize_crowdpose.py \
    --json_path /mnt/h/data/mmpose/anim/train/mmpose_anim_train.json \
    --img_dir /mnt/h/data/mmpose/anim/train
"""

import argparse
import shutil
from pathlib import Path
import json
import numpy as np
import cv2

palette = np.array(
    [
        [255, 128, 0],
        [255, 153, 51],
        [255, 178, 102],
        [230, 230, 0],
        [255, 153, 255],
        [153, 204, 255],
        [255, 102, 255],
        [255, 51, 255],
        [102, 178, 255],
        [51, 153, 255],
        [255, 153, 153],
        [255, 102, 102],
        [255, 51, 51],
        [153, 255, 153],
        [102, 255, 102],
        [51, 255, 51],
        [0, 255, 0],
        [0, 0, 255],
        [255, 0, 0],
        [255, 255, 255],
    ]
)

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


def parse_args():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--json_path", type=Path, required=True)
    parser.add_argument("--img_dir", type=Path, required=True)
    args = parser.parse_args()
    return args


def main(args):
    with open(args.json_path) as f:
        d = json.loads(f.read())

    image_dict = {}
    for ann in d["images"]:
        image_dict[ann["id"]] = ann["file_name"]

    annotation_dict = {}
    for ann in d["annotations"]:
        img_name = image_dict[ann["image_id"]]
        annotation_dict.setdefault(img_name, []).append([ann["bbox"], ann["keypoints"]])

    all_img_paths = sorted(list(annotation_dict.keys()))
    for img_name in all_img_paths[200:]:
        bbox_kps = annotation_dict[img_name]
        img_path = args.img_dir / img_name
        img = cv2.imread(str(img_path))

        for bbox, kps in bbox_kps:
            # print(bbox)
            # cv2.rectangle(
            #     img,
            #     (int(bbox[0]), int(bbox[1])),
            #     (int(bbox[2]), int(bbox[3])),
            #     (255, 255, 255),
            #     3,
            # )

            kps = np.reshape(kps, (14, 3))
            for i, (x, y, _) in enumerate(kps):
                r, g, b = palette[i]
                cv2.circle(img, (x, y), 4, (int(b), int(g), int(r)))

            for b1, b2 in skeleton:
                if kps[b1][2] > 0 and kps[b2][2] > 0:
                    r, g, b = palette[b1]
                    cv2.line(
                        img,
                        (kps[b1][0], kps[b1][1]),
                        (kps[b2][0], kps[b2][1]),
                        (int(b), int(g), int(r)),
                        2,
                    )

        cv2.imshow("img", img)
        if cv2.waitKey(0) == ord("q"):
            break


if __name__ == "__main__":
    args = parse_args()
    main(args)
