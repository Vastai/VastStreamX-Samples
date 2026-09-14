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


import cv2
import argparse
from ppocr_async_cls import PPOCR_Async
import common.utils as utils

import vaststreamx as vsx
import threading
from easydict import EasyDict as edict

attr = vsx.AttrKey
import time
import ast


def argument_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--doc_ori_model",
        default="",
        help="document image orientation classification model prefix of the model suite files",
    )
    parser.add_argument(
        "--doc_ori_vdsp_params",
        default="",
        help="document image orientation classification model vdsp preprocess parameter file",
    )
    parser.add_argument(
        "--doc_ori_label_file",
        default="",
        help="doc image orientation classification label file",
    )
    parser.add_argument(
        "--use_doc_ori_cls", type=int, default=1, help="whether use document image orientation classifier"
    )
    parser.add_argument(
        "--det_model",
        default="/opt/vastai/vaststreamx/data/models/ppocr-v5-mobile/det_fp16_1-3-960-960/mod",
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
        "--det_box_thresh",
        type=float,
        default=0.6,
        help="text detection box threshold",
    )
    parser.add_argument(
        "--text_ori_model",
        default="/opt/vastai/vaststreamx/data/models/resnet34_vd-int8-max-1_3_32_100-vacc/mod",
        help="text detection model prefix of the model suite files",
    )
    parser.add_argument(
        "--text_ori_vdsp_params",
        default="./data/configs/crnn_rgbplanar.json",
        help="text detection vdsp preprocess parameter file",
    )
    parser.add_argument(
        "--text_ori_label_list",
        type=list,
        default=["0", "180"],
        help="text line orientation classification label list",
    )
    parser.add_argument(
        "--text_ori_thresh", type=float, default=0.9, help="text line orientation classification thresh"
    )
    parser.add_argument(
        "--use_text_ori_cls", type=int, default=1, help="whether use text line orientation classifier"
    )
    parser.add_argument(
        "--rec_model",
        default="/opt/vastai/vaststreamx/data/models/resnet34_vd-int8-max-1_3_32_100-vacc/mod",
        help="text recognition model prefix of the model suite files",
    )
    parser.add_argument(
        "--rec_vdsp_params",
        default="./data/configs/crnn_rgbplanar.json",
        help="text recognition vdsp preprocess parameter file",
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
    parser.add_argument(
        "--queue_size",
        type=int,
        default=1,
        help="queue size of the pipeline"
    )
    args = parser.parse_args()
    return args


def inference_async(model, args, context, thread_index):
    vsx.set_device(model.device_id)

    ticks = []
    tocks = []

    # get output thread
    def get_output_thread(model, filelist):
        vsx.set_device(model.device_id)
        index = 0
        while True:
            try:
                output, rot_angle = model.get_output()
                tocks.append(time.time())
                print(f"Thread:{thread_index},Get {filelist[index]} result")
                if len(filelist) == 1:
                    boxes, rec_res = output
                    for box, rec_result in zip(boxes, rec_res):
                        str = "["
                        for point in box:
                            str += f"[{point[0]},{point[1]}], "
                        str = str[:-2] + "], "
                        print(str, rec_result)
                    if args.output_file != "":
                        cv_mat = cv2.imread(filelist[index])
                        if rot_angle == 90:
                            cv_mat = cv2.rotate(cv_mat, cv2.ROTATE_90_COUNTERCLOCKWISE)
                        elif rot_angle == 180:
                            cv_mat = cv2.rotate(cv_mat, cv2.ROTATE_180)
                        elif rot_angle == 270:
                            cv_mat = cv2.rotate(cv_mat, cv2.ROTATE_90_CLOCKWISE)
                        for box in boxes:
                            for i in range(len(box)):
                                t = (i + 1) % len(box)
                                pt1 = (box[i][0], box[i][1])
                                pt2 = (box[t][0], box[t][1])
                                cv2.line(cv_mat, pt1, pt2, color=(0, 0, 255))
                        direc, filename = os.path.split(args.output_file)
                        outfile = os.path.join(
                            direc, f"thread_{thread_index}_{filename}"
                        )
                        if rot_angle == 90:
                            cv_mat = cv2.rotate(cv_mat, cv2.ROTATE_90_CLOCKWISE)
                        elif rot_angle == 180:
                            cv_mat = cv2.rotate(cv_mat, cv2.ROTATE_180)
                        elif rot_angle == 270:
                            cv_mat = cv2.rotate(cv_mat, cv2.ROTATE_90_COUNTERCLOCKWISE)

                        cv2.imwrite(outfile, cv_mat)
                        print(f"save file to {outfile}")
                index += 1
            except Exception:
                break
    # one image test
    if args.dataset_filelist == "":
        cv_mat = cv2.imread(args.input_file)
        assert cv_mat is not None, f"Failed to read image file: {args.input_file}"
        output_thread = threading.Thread(
            target=get_output_thread, args=(model, [args.input_file])
        )
        output_thread.start()
        model.process_async(cv_mat)

        model.stop()
        output_thread.join()
        return

    # dataset test
    filelist = []
    with open(args.dataset_filelist, "rt") as f:
        files = f.readlines()
    if args.dataset_root != "":
        for file in files:
            file = file.replace("\n", "")
            filelist.append(os.path.join(args.dataset_root, file))
    else:
        filelist = files

    output_thread = threading.Thread(target=get_output_thread, args=(model, filelist))
    output_thread.start()

    for i, file in enumerate(filelist):
        cv_image = cv2.imread(file)
        assert cv_image is not None, f"Failed to read image file:{file}"
        model.process_async(cv_image)
        ticks.append(time.time())

    model.stop()

    output_thread.join()
    context.merge_lock.acquire()
    context.ticks += ticks
    context.tocks += tocks
    context.merge_lock.release()


if __name__ == "__main__":
    args = argument_parser()
    device_ids = ast.literal_eval(args.device_ids)
    use_text_ori_cls = bool(args.use_text_ori_cls)

    doc_ori_labels={}
    if args.use_doc_ori_cls:
        lines = utils.load_labels(args.doc_ori_label_file)
        for line in lines:
            label, angle = line.strip().split()
            doc_ori_labels[int(label)] = int(angle)

    models = []
    for id in device_ids:
        model = PPOCR_Async(
            doc_ori_model=args.doc_ori_model,
            doc_ori_vdsp_params=args.doc_ori_vdsp_params,
            doc_ori_labels=doc_ori_labels,
            use_doc_ori_cls=args.use_doc_ori_cls,
            det_model=args.det_model,
            det_vdsp_params=args.det_vdsp_params,
            det_box_type=args.det_box_type,
            det_elf_file=args.det_elf_file,
            det_box_thresh=args.det_box_thresh,
            text_ori_model=args.text_ori_model,
            text_ori_vdsp_params=args.text_ori_vdsp_params,
            text_ori_label_list=args.text_ori_label_list,
            text_ori_thresh=args.text_ori_thresh,
            use_text_ori_cls=args.use_text_ori_cls,
            rec_model=args.rec_model,
            rec_vdsp_params=args.rec_vdsp_params,
            rec_label_file=args.rec_label_file,
            rec_drop_score=args.rec_drop_score,
            batch_size=1,
            device_id=id,
            hw_config=args.hw_config,
            queue_size=args.queue_size,
        )
        models.append(model)

    threads = []
    context = edict(
        merge_lock=threading.Lock(),
        ticks=[],
        tocks=[],
    )

    start = time.time()
    for mod, id in zip(models, device_ids):
        thread = threading.Thread(target=inference_async, args=(mod, args, context, id))
        thread.start()
        threads.append(thread)

    for thread in threads:
        thread.join()
    test_cost = time.time() - start

    if args.dataset_filelist == "":
        exit(0)

    ticks = context.ticks
    tocks = context.tocks

    cost_sum = 0
    assert len(ticks) == len(
        tocks
    ), f"ticks len = {len(ticks)}, tocks len = {len(tocks)}"
    for i in range(len(ticks)):
        cost_sum += tocks[i] - ticks[i]

    avg_cost = cost_sum / len(ticks)

    throughput = len(ticks) / test_cost

    print(
        f"Image count: {len(ticks)}, total cost: {test_cost:.2f} s, throughput: {throughput:.2f} fps, average latency: {avg_cost:.3f} s"
    )
