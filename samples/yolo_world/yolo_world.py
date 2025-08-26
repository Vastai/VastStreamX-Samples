import os
import sys

current_file_path = os.path.dirname(os.path.abspath(__file__))
common_path = os.path.join(current_file_path, "../..")
sys.path.append(common_path)

from common.yolo_world import YoloWorld
import common.utils as utils
import vaststreamx as vsx
import cv2
import argparse
import json
from tqdm import tqdm


def argument_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--imgmod_prefix",
        default="/opt/vastai/vaststreamx/data/models/yolo_world_image-fp16-none-1_3_1280_1280_1203_512-vacc/mod",
        help="image model prefix of the model suite files",
    )
    parser.add_argument(
        "--imgmod_hw_config",
        help="image model hw-config file of the model suite",
        default="",
    )
    parser.add_argument(
        "--imgmod_vdsp_params",
        default="./data/configs/yolo_world_1280_1280_bgr888.json",
        help="vdsp preprocess parameter file",
    )
    parser.add_argument(
        "--txtmod_prefix",
        default=" /opt/vastai/vaststreamx/data/models/yolo_world_text-fp16-none-1_16_1_16-vacc/mod",
        help="text model prefix of the model suite files",
    )
    parser.add_argument(
        "--txtmod_hw_config",
        help="text model hw-config file of the model suite",
        default="",
    )
    parser.add_argument(
        "--txtmod_vdsp_params",
        default="./data/configs/clip_txt_vdsp.json",
        help="text model vdsp preprocess parameter file",
    )
    parser.add_argument(
        "--tokenizer_path",
        default="/opt/vastai/vaststreamx/data/tokenizer/clip-vit-base-patch32",
        help="tokenizer path",
    )
    parser.add_argument(
        "-d",
        "--device_id",
        default=0,
        type=int,
        help="device id to run",
    )
    parser.add_argument(
        "--max_per_image",
        default=300,
        type=int,
        help="max objects detected per image",
    )
    parser.add_argument(
        "--score_thres",
        default=0.001,
        type=float,
        help="object confidence threshold",
    )
    parser.add_argument(
        "--iou_thres",
        default=0.7,
        type=float,
        help="iou threshold",
    )
    parser.add_argument(
        "--nms_pre",
        default=30000,
        type=int,
        help="nms_pre",
    )
    parser.add_argument(
        "--label_file",
        default="data/labels/lvis_v1_class_texts.json",
        help="label file",
    )
    parser.add_argument(
        "--input_file",
        default="data/images/dog.jpg",
        help="input file",
    )
    parser.add_argument(
        "--output_file",
        default="",
        help="output file",
    )
    parser.add_argument(
        "--dataset_filelist",
        default="",
        help="input dataset filelist",
    )
    parser.add_argument(
        "--dataset_root",
        default="",
        help="input dataset root",
    )
    parser.add_argument(
        "--dataset_output_file",
        default="",
        help="dataset output file",
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = argument_parser()
    assert vsx.set_device(args.device_id) == 0

    batch_size = 1
    model = YoloWorld(
        args.imgmod_prefix,
        args.imgmod_vdsp_params,
        args.txtmod_prefix,
        args.txtmod_vdsp_params,
        args.tokenizer_path,
        batch_size,
        args.device_id,
        args.score_thres,
        args.nms_pre,
        args.iou_thres,
        args.max_per_image,
    )

    image_format = model.get_fusion_op_iimage_format()

    with open(args.label_file) as f:
        text_classes = json.load(f)
        text_classes = [x[0] for x in text_classes]

    if args.dataset_filelist == "":
        cv_image = cv2.imread(args.input_file)
        assert cv_image is not None, f"Failed to read input file: {args.input_file}"
        vsx_image = utils.cv_bgr888_to_vsximage(cv_image, image_format, args.device_id)

        result = model.process(image=vsx_image, texts=text_classes)

        scores = result["scores"]
        bboxes = result["bboxes"]
        labels = result["labels"]
        for label, score, box in zip(labels, scores, bboxes):
            print(
                f"Object class: {text_classes[label]}, score: {score:.4f}, bbox: {box}"
            )
        if args.output_file != "":
            for box in bboxes:
                cv2.rectangle(
                    cv_image,
                    pt1=(int(box[0]), int(box[1])),
                    pt2=(int(box[2]), int(box[3])),
                    color=(0, 0, 255),
                    thickness=1,
                )
            cv2.imwrite(args.output_file, cv_image)
    else:
        txt_features = model.process_texts(text_classes)

        filelist = []
        with open(args.dataset_filelist, "rt") as f:
            filelist = [line.strip() for line in f.readlines()]

        dataset_results = []
        for image_file in tqdm(filelist):
            fullname = os.path.join(args.dataset_root, image_file)
            cv_image = cv2.imread(fullname)
            assert cv_image is not None, f"Failed to read image:{fullname}"
            vsx_image = utils.cv_bgr888_to_vsximage(
                cv_image, image_format, args.device_id
            )
            result = model.process_image(vsx_image, txt_features)

            base_name, _ = os.path.splitext(os.path.basename(image_file))

            scores = result["scores"].astype(float)
            bboxes = result["bboxes"].astype(float)
            labels = result["labels"] + 1
            bboxes[:, 2] = bboxes[:, 2] - bboxes[:, 0]
            bboxes[:, 3] = bboxes[:, 3] - bboxes[:, 1]

            for label, score, box in zip(labels, scores, bboxes):
                obj = {}
                obj["image_id"] = int(base_name)
                obj["bbox"] = box.tolist()
                obj["score"] = score
                obj["category_id"] = int(label)
                dataset_results.append(obj)

        with open(args.dataset_output_file, mode="w") as json_file:
            json.dump(dataset_results, json_file)
