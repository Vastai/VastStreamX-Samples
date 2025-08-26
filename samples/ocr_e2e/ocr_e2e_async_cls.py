#
# Copyright (C) 2025 Vastai-tech Company.
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
import os
import sys
import queue
from threading import Thread
from easydict import EasyDict as edict
import concurrent.futures

current_file_path = os.path.dirname(os.path.abspath(__file__))
common_path = os.path.join(current_file_path, "../..")
sys.path.append(common_path)

import common.utils as utils
import numpy as np
import cv2
import time
from common.text_det_async import TextDetectorAsync
from common.text_cls import TextClassifier
from common.text_rec_async import TextRecognizerAsync
import copy
import vaststreamx as vsx

attr = vsx.AttrKey


INPUT_STOP = 1
DET_STOP = 2
DET_POST_STOP = 3
CLS_STOP = 4
REC_STOP = 5


class OCR_e2e_Async:
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
        queue_size=1,
    ):
        self.text_det = TextDetectorAsync(
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
        self.text_rec = TextRecognizerAsync(
            rec_model, rec_config, rec_label_file, batch_size, device_id, hw_config
        )
        self.det_box_type = det_box_type
        self.use_angle_cls = use_angle_cls
        self.cls_thresh = cls_thresh
        self.rec_drop_score = rec_drop_score
        self.input_image_format = self.text_det.get_fusion_op_iimage_format()
        self.device_id = device_id

        self.det_inputs = queue.Queue(queue_size)
        self.det_post_inputs = queue.Queue(queue_size)
        self.cls_inputs = queue.Queue(queue_size)
        self.rec_inputs = queue.Queue(queue_size)
        self.rec_outputs = queue.Queue(queue_size)
        self.stop_flag = 0

        self.get_timeout = 0.01

        self.det_thread = Thread(target=self.detect_thread)
        self.det_post_thread = Thread(target=self.detect_post_thread)
        self.cls_thread = Thread(target=self.classify_thread)
        self.rec_thread = Thread(target=self.recognize_thread)

        self.det_thread.start()
        self.det_post_thread.start()
        self.cls_thread.start()
        self.rec_thread.start()

    def process_async(self, cv_image):
        self.det_inputs.put(cv_image)

    def get_output(self):
        while True:
            try:
                out = self.rec_outputs.get(timeout=self.get_timeout)
                return out
            except Exception:
                if self.stop_flag == REC_STOP:
                    raise Exception("Inference is stop")

    def stop(self):
        self.stop_flag = INPUT_STOP

    def detect_thread(self):
        vsx.set_device(self.device_id)
        input_mats = queue.Queue(50)

        def output_thread(text_det, input_mats, det_post_inputs):
            vsx.set_device(self.device_id)
            while True:
                try:
                    det_result = text_det.get_output()
                    cv_image = input_mats.get()
                    det_post_inputs.put((cv_image, det_result))
                except Exception as e:
                    break

        thread = Thread(
            target=output_thread, args=(self.text_det, input_mats, self.det_post_inputs)
        )
        thread.start()
        image_format = self.text_det.get_fusion_op_iimage_format()
        while True:
            try:
                cv_mat = self.det_inputs.get(timeout=self.get_timeout)
                vsx_image = utils.cv_bgr888_to_vsximage(
                    cv_mat, image_format, self.text_det.device_id_
                )
                input_mats.put(cv_mat)
                self.text_det.process_async(vsx_image)
            except Exception as e:
                if self.stop_flag == INPUT_STOP:
                    self.text_det.close_input()
                    thread.join()
                    self.text_det.wait_until_done()
                    self.stop_flag = DET_STOP
                    break

    def det_post_process(self, post_input):
        vsx.set_device(self.text_cls.device_id_)
        cv_mat, [[dt_boxes, dt_scores]] = post_input
        img_crop_list = []
        vacc_img_crop_list = []
        format = self.text_cls.get_fusion_op_iimage_format()
        for bno in range(len(dt_boxes)):
            tmp_box = copy.deepcopy(dt_boxes[bno])
            if self.det_box_type == "quad":
                img_crop = self.get_rotate_crop_image(cv_mat, tmp_box)
            else:
                img_crop = self.get_minarea_rect_crop(cv_mat, tmp_box)
            img_crop_list.append(img_crop)

            vacc_img_crop = utils.cv_bgr888_to_vsximage(
                img_crop, format, self.text_cls.device_id_
            )
            vacc_img_crop_list.append(vacc_img_crop)
        return ([dt_boxes, dt_scores], img_crop_list, vacc_img_crop_list)

    def detect_post_thread(self):
        vsx.set_device(self.device_id)
        queue_futs = queue.Queue(10)
        context = edict(stopped=False, left=0)

        def cunsume_thread_func(context, queue_futs, cls_inputs):
            vsx.set_device(self.device_id)
            while not context.stopped or context.left > 0:
                try:
                    fut = queue_futs.get(timeout=self.get_timeout)
                    result = fut.result()
                    cls_inputs.put(result)
                    context.left -= 1
                except Exception:
                    pass

        cunsume_thread = Thread(
            target=cunsume_thread_func, args=(context, queue_futs, self.cls_inputs)
        )
        cunsume_thread.start()

        with concurrent.futures.ThreadPoolExecutor() as executor:
            while True:
                try:
                    post_in = self.det_post_inputs.get(timeout=self.get_timeout)
                    fut = executor.submit(self.det_post_process, post_in)
                    context.left += 1
                    queue_futs.put(fut)
                except Exception:
                    if self.stop_flag == DET_STOP:
                        context.stopped = True
                        cunsume_thread.join()
                        self.stop_flag = DET_POST_STOP
                        break

    def classify_thread(self):
        vsx.set_device(self.text_det.device_id_)
        while True:
            try:
                cls_input = self.cls_inputs.get(timeout=self.get_timeout)
                [dt_boxes, dt_scores], img_crop_list, vacc_img_crop_list = cls_input
                if self.use_angle_cls and self.text_cls and len(vacc_img_crop_list) > 0:
                    format = self.text_det.get_fusion_op_iimage_format()
                    cls_result = self.text_cls.process(vacc_img_crop_list)
                    for rno in range(len(cls_result)):
                        label, score = cls_result[rno]
                        if "180" in label and score > self.cls_thresh:
                            img_crop_list[rno] = cv2.rotate(img_crop_list[rno], 1)
                            vacc_img_crop_list[rno] = utils.cv_bgr888_to_vsximage(
                                img_crop_list[rno], format, self.text_det.device_id_
                            )
                self.rec_inputs.put(([dt_boxes, dt_scores], vacc_img_crop_list))
            except Exception:
                if self.stop_flag == DET_POST_STOP:
                    self.stop_flag = CLS_STOP
                    break

    def recognize_thread(self):
        vsx.set_device(self.device_id)
        det_results = queue.Queue(50)
        infer_flags = queue.Queue(50)

        def output_thread(text_rec, det_results, rec_outputs):
            vsx.set_device(self.device_id)
            while True:
                try:
                    flag = infer_flags.get()
                    if flag:
                        rec_res = text_rec.get_output()
                        dt_boxes = det_results.get()
                    filter_boxes, filter_rec_res = [], []
                    for box, rec_result in zip(dt_boxes, rec_res):
                        text, score = rec_result[0]
                        if score >= self.rec_drop_score:
                            filter_boxes.append(box)
                            filter_rec_res.append(rec_result)
                    rec_outputs.put((filter_boxes, filter_rec_res))
                except Exception:
                    break

        thread = Thread(
            target=output_thread, args=(self.text_rec, det_results, self.rec_outputs)
        )
        thread.start()
        while True:
            try:
                rec_input = self.rec_inputs.get(timeout=self.get_timeout)
                [dt_boxes, dt_scores], vacc_img_crop_list = rec_input
                if len(vacc_img_crop_list) > 0:
                    det_results.put(dt_boxes)
                    infer_flags.put(True)
                    self.text_rec.process_async(vacc_img_crop_list)
                else:
                    infer_flags.put(False)
            except Exception:
                if self.stop_flag == CLS_STOP:
                    infer_flags.put(True)
                    self.text_rec.close_input()
                    thread.join()
                    self.text_rec.wait_until_done()
                    self.stop_flag = REC_STOP
                    break

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
