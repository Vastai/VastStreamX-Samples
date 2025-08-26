#
# Copyright (C) 2025 Vastai-tech Company.
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
import os
import sys

current_file_path = os.path.dirname(os.path.abspath(__file__))
common_path = os.path.join(current_file_path, "../..")
sys.path.append(common_path)

import common.utils as utils
import numpy as np
import cv2
import argparse
from common.text_det import TextDetector
from common.text_cls import TextClassifier
from common.text_rec import TextRecognizer
import copy
import vaststreamx as vsx
from easydict import EasyDict as edict
import time
import threading
import ast

attr = vsx.AttrKey


def argument_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--det_model",
        default="/opt/vastai/vaststreamx/data/models/dbnet_resnet50_vd-int8-kl_divergence-1_3_736_1280-vacc/mod",
        help="text detection model prefix of the model suite files",
    )
    parser.add_argument(
        "--det_vdsp_params",
        default="./data/configs/dbnet_rgbplanar.json",
        help="text detection vdsp preprocess parameter file",
    )
    parser.add_argument(
        "--det_box_type",
        default="quad",
        help="det box type, poly or quad",
    )
    parser.add_argument(
        "--det_elf_file",
        default="/opt/vastai/vaststreamx/data/elf/find_contours_ext_op",
        help="input file",
    )
    parser.add_argument(
        "--cls_model",
        default="/opt/vastai/vaststreamx/data/models/resnet34_vd-int8-max-1_3_32_100-vacc/mod",
        help="text detection model prefix of the model suite files",
    )
    parser.add_argument(
        "--cls_vdsp_params",
        default="./data/configs/crnn_rgbplanar.json",
        help="text detection vdsp preprocess parameter file",
    )
    parser.add_argument(
        "--cls_label_list",
        type=list,
        default=["0", "180"],
        help="text classification label list",
    )
    parser.add_argument(
        "--cls_thresh", type=float, default=0.9, help="text classification thresh"
    )
    parser.add_argument(
        "--rec_model",
        default="/opt/vastai/vaststreamx/data/models/resnet34_vd-int8-max-1_3_32_100-vacc/mod",
        help="text detection model prefix of the model suite files",
    )
    parser.add_argument(
        "--rec_vdsp_params",
        default="./data/configs/crnn_rgbplanar.json",
        help="text detection vdsp preprocess parameter file",
    )
    parser.add_argument(
        "--rec_label_file",
        default="../data/labels/key_37.txt",
        help="text recognizition label file",
    )
    parser.add_argument(
        "--rec_drop_score",
        type=float,
        default=0.5,
        help="text recogniztion drop score threshold",
    )
    parser.add_argument(
        "--use_angle_cls", type=bool, default=True, help="whether use angle classifier"
    )
    parser.add_argument(
        "--hw_config",
        default="",
        help="hw-config file of the model suite",
    )
    parser.add_argument(
        "--device_ids",
        default="[0]",
        type=str,
        help="device ids to run",
    )
    parser.add_argument(
        "--input_file",
        default="../../../data/images/detect.jpg",
        help="input file",
    )
    parser.add_argument(
        "--output_file",
        default="./ocr_result.jpg",
        help="output file",
    )
    parser.add_argument(
        "--dataset_filelist",
        default="",
        help="dataset filelist",
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


class OCR_e2e:
    def __init__(
        self,
        det_model,
        det_config,
        det_box_type,
        det_elf_file,
        cls_model,
        cls_config,
        cls_label_list,
        cls_thresh,
        rec_model,
        rec_config,
        rec_label_file,
        rec_drop_score,
        use_angle_cls,
        batch_size=1,
        device_id=0,
        hw_config="",
    ):
        self.text_det = TextDetector(
            det_model,
            det_config,
            batch_size,
            device_id,
            hw_config,
            elf_file=det_elf_file,
        )
        self.text_cls = TextClassifier(
            cls_model, cls_config, cls_label_list, batch_size, device_id, hw_config
        )
        self.text_rec = TextRecognizer(
            rec_model, rec_config, rec_label_file, batch_size, device_id, hw_config
        )
        self.det_box_type = det_box_type
        self.use_angle_cls = use_angle_cls
        self.cls_thresh = cls_thresh
        self.rec_drop_score = rec_drop_score
        self.device_id = device_id
        self.image_format = self.text_det.get_fusion_op_iimage_format()

    def process(self, image, cv_image):
        [dt_boxes, dt_scores] = self.text_det.process(image)
        if dt_boxes is None or dt_boxes.size == 0:
            return None
        img_crop_list = []
        vacc_img_crop_list = []

        for bno in range(len(dt_boxes)):
            tmp_box = copy.deepcopy(dt_boxes[bno])
            if args.det_box_type == "quad":
                img_crop = self.get_rotate_crop_image(cv_image, tmp_box)
            else:
                img_crop = self.get_minarea_rect_crop(cv_image, tmp_box)
            img_crop_list.append(img_crop)
            vacc_img_crop = utils.cv_bgr888_to_vsximage(
                img_crop, self.image_format, self.device_id
            )
            vacc_img_crop_list.append(vacc_img_crop)
        if self.use_angle_cls and self.text_cls:
            cls_result = self.text_cls.process(vacc_img_crop_list)
            for rno in range(len(cls_result)):
                label, score = cls_result[rno]
                if "180" in label and score > self.cls_thresh:
                    img_crop_list[rno] = cv2.rotate(img_crop_list[rno], 1)
                    vacc_img_crop_list[rno] = utils.cv_bgr888_to_vsximage(
                        img_crop_list[rno], self.image_format, self.device_id
                    )

        rec_res = self.text_rec.process(vacc_img_crop_list)

        filter_boxes, filter_rec_res = [], []
        for box, rec_result in zip(dt_boxes, rec_res):
            text, score = rec_result[0]
            if score >= self.rec_drop_score:
                filter_boxes.append(box)
                filter_rec_res.append(rec_result)
        return filter_boxes, filter_rec_res

    def get_rotate_crop_image(self, img, points):
        """
        img_height, img_width = img.shape[0:2]
        left = int(np.min(points[:, 0]))
        right = int(np.max(points[:, 0]))
        top = int(np.min(points[:, 1]))
        bottom = int(np.max(points[:, 1]))
        img_crop = img[top:bottom, left:right, :].copy()
        points[:, 0] = points[:, 0] - left
        points[:, 1] = points[:, 1] - top
        """
        assert len(points) == 4, "shape of points must be 4*2"
        points = np.array(points, dtype=np.float32)
        img_crop_width = int(
            max(
                np.linalg.norm(points[0] - points[1]),
                np.linalg.norm(points[2] - points[3]),
            )
        )
        img_crop_height = int(
            max(
                np.linalg.norm(points[0] - points[3]),
                np.linalg.norm(points[1] - points[2]),
            )
        )
        pts_std = np.float32(
            [
                [0, 0],
                [img_crop_width, 0],
                [img_crop_width, img_crop_height],
                [0, img_crop_height],
            ]
        )
        M = cv2.getPerspectiveTransform(points, pts_std)
        dst_img = cv2.warpPerspective(
            img,
            M,
            (img_crop_width, img_crop_height),
            borderMode=cv2.BORDER_REPLICATE,
            flags=cv2.INTER_CUBIC,
        )
        dst_img_height, dst_img_width = dst_img.shape[0:2]
        if dst_img_height * 1.0 / dst_img_width >= 1.5:
            dst_img = np.rot90(dst_img)
        return dst_img

    def get_minarea_rect_crop(self, img, points):
        bounding_box = cv2.minAreaRect(np.array(points).astype(np.int32))
        points = sorted(list(cv2.boxPoints(bounding_box)), key=lambda x: x[0])

        index_a, index_b, index_c, index_d = 0, 1, 2, 3
        if points[1][1] > points[0][1]:
            index_a = 0
            index_d = 1
        else:
            index_a = 1
            index_d = 0
        if points[3][1] > points[2][1]:
            index_b = 2
            index_c = 3
        else:
            index_b = 3
            index_c = 2

        box = [points[index_a], points[index_b], points[index_c], points[index_d]]
        crop_img = self.get_rotate_crop_image(img, np.array(box))
        return crop_img

    def get_fusion_op_iimage_format(self):
        return self.text_det.get_fusion_op_iimage_format()


def inference_thread(model, args, context, thread_index):
    vsx.set_device(model.device_id)
    image_format = model.get_fusion_op_iimage_format()

    if args.dataset_filelist == "":
        cv_image = cv2.imread(args.input_file)
        assert cv_image is not None, f"Failed to read input file: {args.input_file}"
        vsx_image = utils.cv_bgr888_to_vsximage(cv_image, image_format, model.device_id)
        ocr_res = model.process(vsx_image, cv_image)
        if ocr_res is None:
            print("Do not detect any text")
        else:
            boxes, rec_res = ocr_res
            for box, rec_result in zip(boxes, rec_res):
                out_str = "["
                for point in box:
                    out_str += f"[{point[0]},{point[1]}], "
                out_str = out_str[:-2] + "], "
                print(out_str, rec_result)
            if args.output_file != "":
                for box in boxes:
                    for i in range(len(box)):
                        t = (i + 1) % len(box)
                        pt1 = (box[i][0], box[i][1])
                        pt2 = (box[t][0], box[t][1])
                        cv2.line(cv_image, pt1, pt2, color=(0, 0, 255))
                dir, basename = os.path.split(args.output_file)
                save_file = os.path.join(dir, f"thread_{thread_index}_{basename}")
                cv2.imwrite(save_file, cv_image)
                print("save file ", save_file)

    else:
        costs = []
        filelist = []
        with open(args.dataset_filelist, "rt") as f:
            filelist = f.readlines()
        dir, basename = os.path.split(args.dataset_output_file)
        save_file = os.path.join(dir, f"thread_{thread_index}_{basename}")
        with open(save_file, "wt") as outfile:
            for filename in filelist:
                fullname = os.path.join(args.dataset_root, filename.replace("\n", ""))
                print("fullname:", fullname)
                cv_image = cv2.imread(fullname)
                assert cv_image is not None, f"Read image failed:{filename}"
                vsx_image = utils.cv_bgr888_to_vsximage(
                    cv_image, image_format, model.device_id
                )
                start = time.time()
                ocr_res = model.process(vsx_image, cv_image)
                costs.append(time.time() - start)
                result_str = ""
                if ocr_res is None:
                    result_str = "Do not detect any text"
                else:
                    boxes, rec_res = ocr_res
                    for box, rec_result in zip(boxes, rec_res):
                        res_str = "["
                        for point in box:
                            res_str += f"[{point[0]},{point[1]}], "
                        res_str = (
                            res_str[:-2]
                            + "], "
                            + rec_result[0][0]
                            + ", "
                            + str(rec_result[0][1])
                        )
                        result_str += res_str + "\n"
                basename, _ = os.path.splitext(os.path.basename(fullname))
                outfile.write(f"{basename}\n{result_str}\n")
        outfile.close()
        cost_sum = np.sum(costs)
        throughput = len(costs) / cost_sum

        context.merge_lock.acquire()
        context.costs += costs
        context.throughput += throughput
        context.merge_lock.release()


if __name__ == "__main__":
    args = argument_parser()
    device_ids = ast.literal_eval(args.device_ids)

    models = []
    for id in device_ids:
        model = OCR_e2e(
            args.det_model,
            args.det_vdsp_params,
            args.det_box_type,
            args.det_elf_file,
            args.cls_model,
            args.cls_vdsp_params,
            args.cls_label_list,
            args.cls_thresh,
            args.rec_model,
            args.rec_vdsp_params,
            args.rec_label_file,
            args.rec_drop_score,
            args.use_angle_cls,
            batch_size=1,
            device_id=id,
            hw_config=args.hw_config,
        )
        models.append(model)

    threads = []
    context = edict(merge_lock=threading.Lock(), costs=[], throughput=0)

    for mod, id in zip(models, device_ids):
        thread = threading.Thread(
            target=inference_thread, args=(mod, args, context, id)
        )
        thread.start()
        threads.append(thread)

    for thread in threads:
        thread.join()

    if args.dataset_filelist == "":
        exit(0)
    costs = context.costs
    throughput = context.throughput
    avg_cost = np.mean(costs)
    cost_sum = np.sum(costs) / len(device_ids)
    print(
        f"Image count: {len(costs)}, total cost: {cost_sum:.2f} s, throughput: {throughput:.2f} fps, average latency: {avg_cost:.3f} s"
    )
