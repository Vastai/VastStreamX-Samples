#
# Copyright (C) 2025 Vastai-tech Company.
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
import vaststreamx as vsx
import numpy as np
import argparse
import glob
import cv2
from typing import List, Union
import json
from tqdm import tqdm
import torch

from pose_utils import (
    non_max_suppression,
    DFL,
    dist2bbox,
    make_anchors,
    scale_coords,
    coco80_to_coco91_class,
    coco_names,
    xyxy2xywh,
)

import os
import sys

current_file_path = os.path.dirname(os.path.abspath(__file__))
common_path = os.path.join(current_file_path, "../../../")
sys.path.append(common_path)

from common.yolov8_pose import Yolov8Pose
from common import utils

parse = argparse.ArgumentParser(description="RUN Pose WITH VSX")
parse.add_argument(
    "--file_path",
    type=str,
    default="/path/to/data/det_coco_val/",
    help="dir  path",
)
parse.add_argument(
    "--model_prefix_path",
    type=str,
    default="deploy_weights/ultralytics_yolov8_pose_run_stream_int8/mod",
    help="model info",
)
parse.add_argument(
    "--vdsp_params_info",
    type=str,
    default="../vacc_code/params_info/ultralytics-yolov8s_pose-vdsp_params.json",
    help="vdsp op info",
)
parse.add_argument(
    "--label_txt", type=str, default="/path/to/coco.txt", help="label txt"
)
parse.add_argument(
    "--draw_output",
    action="store_true",
    required=False,
    default=False,
    help="save output image or not",
)
parse.add_argument(
    "--threashold", type=float, default=0.01, help="threashold for postprocess"
)
parse.add_argument("--device_id", type=int, default=0, help="device id")
parse.add_argument("--batch", type=int, default=1, help="bacth size")
parse.add_argument("--save_dir", type=str, default="./output/", help="save_dir")
args = parse.parse_args()


class Segmenter:
    def __init__(
        self,
        model_size: Union[int, list],
        classes: Union[str, List[str]],
        model_output_shape: list,
        draw_output: bool = False,
        threashold: float = 0.01,
    ) -> None:
        if isinstance(classes, str):
            with open(classes) as f:
                classes = [cls.strip() for cls in f.readlines()]

        if isinstance(model_size, int):
            self.model_size = [model_size, model_size]
        else:
            # h,w
            assert len(model_size) == 2, "model_size ERROR."
            self.model_size = model_size  # h,w
        self.classes = classes
        self.threashold = threashold
        self.draw_output = draw_output
        self.output_shape = model_output_shape
        self.jdict = []

    def scale_coords(self, img1_shape, coords, img0_shape, ratio_pad=None):
        # Rescale coords (xyxy) from img1_shape to img0_shape
        if ratio_pad is None:  # calculate from img0_shape
            # gain  = old / new
            gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])
            pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (
                img1_shape[0] - img0_shape[0] * gain
            ) / 2  # wh padding
        else:
            gain = ratio_pad[0][0]
            pad = ratio_pad[1]
        coords[:, [0, 2]] -= pad[0]  # x padding
        coords[:, [1, 3]] -= pad[1]  # y padding
        coords[:, :4] /= gain

        self.clip_coords(coords, img0_shape)
        return coords

    def kpts_decode(self, kpts, anchors, strides):
        ndim = 3
        y = kpts.clone()
        # if ndim == 3:
        y[:, 2::3].sigmoid_()  # inplace sigmoid
        y[:, 0::ndim] = (y[:, 0::ndim] * 2.0 + (anchors[0] - 0.5)) * strides
        y[:, 1::ndim] = (y[:, 1::ndim] * 2.0 + (anchors[1] - 0.5)) * strides
        return y

    def clip_coords(self, boxes, shape):
        boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, shape[1])  # x1, x2
        boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, shape[0])  # y1, y2

    def postprocess(self, stream_ouput, classes_list, image_file, save_dir, **kwargs):
        # print(f"=====>>>>>process img-{image_file}")
        origin_img = cv2.imread(image_file, cv2.IMREAD_COLOR)
        height, width, _ = origin_img.shape
        file_name = os.path.basename(image_file)

        dfl = DFL(16)
        for i in range(9):
            stream_ouput[i] = torch.Tensor(
                stream_ouput[i].reshape(self.output_shape[i])
            )

        ## detect cat
        output = []
        for i in range(3):
            x = torch.cat((stream_ouput[i * 2], stream_ouput[i * 2 + 1]), 1)
            output.append(x)

        anchors, strides = (
            x.transpose(0, 1) for x in make_anchors(output, [8, 16, 32], 0.5)
        )

        x_cat = torch.cat([xi.view(1, 65, -1) for xi in output], 2)
        box, cls = x_cat.split((16 * 4, 1), 1)
        dbox = dist2bbox(dfl(box), anchors.unsqueeze(0), xywh=True, dim=1) * strides
        ty = torch.cat((dbox, cls.sigmoid()), 1)

        det_out = (ty, output)

        ## keypoints cat
        tkpt = []
        for i in range(6, 9):
            tkpt.append(stream_ouput[i].view(1, 51, -1))
        kpt = torch.cat(tkpt, -1)
        # print(strides)
        pkpt = self.kpts_decode(kpt, anchors, strides)
        kpt_out = (torch.cat([det_out[0], pkpt], 1), (det_out[1], kpt))
        pred = non_max_suppression(kpt_out)[0]
        # print(pred)
        npr = pred.shape[0]
        if npr != 0:
            predn = pred.clone()
            # print(f"npr-{npr}")
            pred_kpts = predn[:, 6:].view(npr, 17, -1)
            scale_r = self.model_size[0] / max(height, width)
            pad_h = (self.model_size[0] - height * scale_r) / 2
            pad_w = (self.model_size[0] - width * scale_r) / 2
            scale_coords(
                self.model_size,
                pred_kpts,
                (height, width),
                ratio_pad=((scale_r, scale_r), (pad_w, pad_h)),
            )
            if len(pred):
                pred[:, :4] = self.scale_coords(
                    self.model_size, pred[:, :4], [height, width, 3]
                ).round()

            pred = pred.numpy()
            res_length = len(pred)
            COLORS = np.random.uniform(0, 255, size=(len(classes_list), 3))

            # 画框
            box_list = []
            if res_length:
                for index in range(res_length):
                    label = classes_list[pred[index][5].astype(np.int8)]
                    score = pred[index][4]
                    bbox = pred[index][:4].tolist()
                    p1, p2 = (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3]))
                    cv2.rectangle(
                        origin_img,
                        p1,
                        p2,
                        (0, 255, 255),
                        thickness=1,
                        lineType=cv2.LINE_AA,
                    )
                    text = f"{label}: {round(score * 100, 2)}%"
                    y = (
                        int(int(bbox[1])) - 15
                        if int(int(bbox[1])) - 15 > 15
                        else int(int(bbox[1])) + 15
                    )
                    cv2.putText(
                        origin_img,
                        text,
                        (int(bbox[0]), y),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        COLORS[pred[index][5].astype(np.int8)],
                        2,
                    )
                    box_list.append(
                        f"{label} {score} {int(bbox[0])} {int(bbox[1])} {int(bbox[2])} {int(bbox[3])}"
                    )
                if self.draw_output:
                    cv2.imwrite(f"{save_dir}/{file_name}", origin_img)
            new_filename = os.path.splitext(file_name)[0]
            self.pred_to_json(box_list, new_filename, predn)

    def pred_to_json(self, box_list, file_name, predn):
        """Save one JSON result."""
        coco_num = coco80_to_coco91_class()
        image_id = int(file_name) if file_name.isnumeric() else file_name
        box = []
        label = []
        score = []
        for line in box_list:
            line = line.strip().split()
            label.append(coco_num[coco_names.index(" ".join(line[:-5]))])
            box.append([float(l) for l in line[-4:]])
            score.append(float(line[-5]))
        if len(box):
            box = xyxy2xywh(np.array(box))  # x1y1wh
            box[:, :2] -= box[:, 2:] / 2  # xy center to top-left corner
            kptlist = predn.tolist()

            for i in range(len(box.tolist())):
                self.jdict.append(
                    {
                        "image_id": image_id,
                        "category_id": label[i],
                        "bbox": [x for x in box[i].tolist()],
                        # 'keypoints': [x[6:] for x in predn.tolist()],
                        "keypoints": kptlist[i][6:],
                        "score": score[i],
                    }
                )

    def save_json(self, json_save_dir):
        with open(json_save_dir + "/predictions.json", "w") as f:
            json.dump(self.jdict, f)  # flatten and save

    def feature_decode(self, input_image_path: str, feature_maps: list, json_save_dir):
        stream_ouput = [vsx.as_numpy(o).astype(np.float32) for o in feature_maps]
        # post proecess
        self.postprocess(stream_ouput, self.classes, input_image_path, json_save_dir)
        return stream_ouput


if __name__ == "__main__":
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    pose_detector = Yolov8Pose(
        model_prefix=args.model_prefix_path,
        vdsp_config=args.vdsp_params_info,
        batch_size=args.batch,
        device_id=args.device_id,
    )
    model_input_shape = pose_detector.input_shape[0]

    feature_decoder = Segmenter(
        model_size=[model_input_shape[2], model_input_shape[3]],
        classes=args.label_txt,
        draw_output=args.draw_output,
        threashold=args.threashold,
        model_output_shape=pose_detector.output_shape,
    )
    image_format = pose_detector.get_fusion_op_iimage_format()
    if os.path.isfile(args.file_path):
        cv_image = cv2.imread(args.file_path)
        assert cv_image is not None, f"Read image failed:{args.file_path}"
        vsx_image = utils.cv_bgr888_to_vsximage(cv_image, image_format, args.device_id)
        outputs = pose_detector.process(vsx_image)
        feature_decoder.feature_decode(args.file_path, outputs, args.save_dir)
    else:
        images = glob.glob(os.path.join(args.file_path, "*"))
        for image in tqdm(images):
            cv_image = cv2.imread(image)
            assert cv_image is not None, f"Read image failed:{image}"
            vsx_image = utils.cv_bgr888_to_vsximage(
                cv_image, image_format, args.device_id
            )
            outputs = pose_detector.process(vsx_image)
            feature_decoder.feature_decode(image, outputs, args.save_dir)

    feature_decoder.save_json(args.save_dir)
