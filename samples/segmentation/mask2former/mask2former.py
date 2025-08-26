import os
import sys

current_file_path = os.path.dirname(os.path.abspath(__file__))
common_path = os.path.join(current_file_path, "../../..")
sys.path.append(common_path)

import numpy as np
import cv2
import argparse
from common.mask2former import Mask2Former
import common.utils as utils
from pycocotools.mask import encode  # noqa
import json


def single_encode(x):
    """Encode predicted masks as RLE and append results to jdict."""
    rle = encode(np.asarray(x[:, :, None], order="F", dtype="uint8"))[0]
    rle["counts"] = rle["counts"].decode("utf-8")
    return rle


def argument_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m",
        "--model_prefix",
        default="/opt/vastai/vaststreamx/data/models/mask2former-fp16-none-1_3_1024_1024-vacc/mod",
        help="model prefix of the model suite files",
    )
    parser.add_argument(
        "--hw_config",
        default="",
        help="hw-config file of the model suite",
    )
    parser.add_argument(
        "--vdsp_params",
        default="./data/configs/mask2former_rgbplanar.json",
        help="vdsp preprocess parameter file",
    )
    parser.add_argument(
        "--label_file",
        default="../data/labels/coco2id.txt",
        help="label file",
    )
    parser.add_argument(
        "-d",
        "--device_id",
        default=0,
        type=int,
        help="device id to run",
    )
    parser.add_argument(
        "--threshold",
        default=0.6,
        type=float,
        help="detection threshold",
    )
    parser.add_argument(
        "--input_file",
        default="../../../data/images/cycling.jpg",
        help="input file",
    )
    parser.add_argument(
        "--output_file",
        default="./mask2former_result.jpg",
        help="output file",
    )
    parser.add_argument(
        "--dataset_filelist",
        default="",
        help="input dataset image list",
    )
    parser.add_argument(
        "--dataset_root",
        default="",
        help="input dataset root",
    )
    parser.add_argument(
        "--dataset_output_file",
        default="mask2former_predictions.json",
        help="dataset output file",
    )
    args = parser.parse_args()
    return args


colors = [
    [62, 140, 230],
    [255, 85, 0],
    [255, 170, 0],
    [255, 0, 85],
    [0, 255, 0],
    [85, 255, 0],
    [170, 255, 0],
    [0, 255, 85],
    [0, 255, 170],
    [0, 0, 255],
    [85, 0, 255],
    [170, 0, 255],
    [0, 85, 255],
    [0, 170, 255],
    [255, 255, 0],
    [255, 255, 85],
    [255, 255, 170],
    [255, 0, 170],
    [255, 0, 255],
    [255, 85, 255],
    [255, 170, 255],
    [0, 255, 255],
    [85, 255, 255],
    [170, 255, 255],
]


def save_mask(tensors, orgin_image, output_image, labels, threshold):
    classes, scores, boxes = tensors[:3]
    det_count = tensors[4][0]
    masks = tensors[3]
    mask_h, mask_w = masks.shape[-2:]
    cvmask = np.zeros((mask_h, mask_w, 3), dtype=np.uint8)
    for i in range(det_count):
        if scores[i] < threshold:
            continue
        m = masks[i][:, :, np.newaxis]
        c = i % len(colors)
        msk = m * colors[c]
        cvmask += msk.astype(np.uint8)

    cvorigin = cv2.imread(orgin_image)
    cvorigin = cvorigin * 0.5 + cvmask * 0.5
    for i in range(det_count):
        if scores[i] < threshold:
            continue
        color = colors[i % len(colors)]
        box = boxes[i]
        cls_scr = f"{labels[int(classes[i])]},{scores[i]:.2f}"
        cv2.rectangle(cvorigin, (box[0], box[1]), (box[2], box[3]), color)
        font = cv2.FONT_HERSHEY_SIMPLEX
        left, top = box[:2].astype(int)
        top += 15
        cv2.putText(cvorigin, cls_scr, (left, top), font, 0.5, color)

    cv2.imwrite(output_image, cvorigin.astype(np.uint8))


if __name__ == "__main__":
    args = argument_parser()
    batch_size = 1
    segmenter = Mask2Former(
        args.model_prefix,
        args.vdsp_params,
        batch_size,
        args.device_id,
        args.threshold,
        args.hw_config,
    )
    labels = utils.load_labels(args.label_file)
    image_format = segmenter.get_fusion_op_iimage_format()

    if args.dataset_filelist == "":
        cv_image = cv2.imread(args.input_file)
        assert cv_image is not None, f"Read image failed:{args.input_file}"
        vsx_image = utils.cv_bgr888_to_vsximage(cv_image, image_format, args.device_id)
        outputs = segmenter.process(vsx_image)
        classes, scores, boxes = outputs[:3]
        det_count = outputs[4][0]
        if det_count == 0:
            print("No object detected in image.")
        else:
            for i in range(det_count):
                print(
                    f"Object class: {labels[int(classes[i])]}, score: {scores[i]}, bbox: {boxes[i].tolist()}"
                )
            if args.output_file != "":
                save_mask(
                    outputs, args.input_file, args.output_file, labels, args.threshold
                )
    else:
        if args.dataset_output_file == "":
            print("The option 'dataset_output_file' is empty")
            exit(-1)
        filelist = []
        with open(args.dataset_filelist, "rt") as f:
            filelist = f.readlines()
        jdict = []
        coco_num = utils.coco80_to_coco91_class()
        for i, filename in enumerate(filelist):
            filename = os.path.join(args.dataset_root, filename.replace("\n", ""))
            print(f"{i+1}/{len(filelist)} {filename}")
            cv_image = cv2.imread(filename)
            assert cv_image is not None, f"Read image failed:{filename}"
            vsx_image = utils.cv_bgr888_to_vsximage(
                cv_image, image_format, args.device_id
            )
            outputs = segmenter.process(vsx_image)
            basename, _ = os.path.splitext(os.path.basename(filename))
            image_id = int(basename)

            classes = outputs[0]
            scores = outputs[1]
            boxes = outputs[2]
            masks = outputs[3]
            det_num = outputs[4][0]

            for i in range(det_num):
                box = boxes[i].tolist()
                box[2] = box[2] - box[0]
                box[3] = box[3] - box[1]

                jdict.append(
                    {
                        "image_id": image_id,
                        "category_id": coco_num[int(classes[i])],
                        "bbox": [x for x in box],
                        "score": scores[i].astype(float),
                        "segmentation": single_encode(masks[i]),
                    }
                )

        with open(args.dataset_output_file, "w") as f:
            json.dump(jdict, f)  # flatten and save
