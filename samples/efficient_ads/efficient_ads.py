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

from common.model_cv import ModelCV
import numpy as np
import cv2
import argparse
import common.utils as utils
import glob
import onnxruntime as ort
import torchvision.transforms.functional as F
import torch
from tqdm import tqdm


def argument_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m",
        "--model_prefix",
        default="/home/aico/Downloads/docker/xmu/EfficientAD-S/deploy_weights/official_efficientAD_run_stream_fp16/mod",
        help="model prefix of the model suite files",
    )
    parser.add_argument(
        "--hw_config",
        default="",
        help="hw-config file of the model suite",
    )
    parser.add_argument(
        "--vdsp_params",
        default="../../data/configs/efficient_ads_rgbplanar.json",
        help="vdsp preprocess parameter file",
    )
    parser.add_argument(
        "-d",
        "--device_id",
        default=0,
        type=int,
        help="device id to run",
    )
    parser.add_argument(
        "--input_file",
        default="/home/aico/Downloads/docker/xmu/EfficientAD-S/offical/datasets/MVTecAD/zipper/test/broken_teeth/011.png",
        help="input file",
    )
    parser.add_argument(
        "--output_file",
        default="./efficient_ads_vsx_output_py.npz",
        help="output file",
    )
    parser.add_argument(
        "--onnx_file",
        default="/home/aico/customer/efficient_ads/onnx_model/model.onnx",
        help="onnx file",
    )
    parser.add_argument(
        "--dataset_root_dir",
        default="",
        help="dataset root dir, such as zipper",
    )
    parser.add_argument(
        "--split",
        default="test",
        help="split of the dataset",
    )
    args = parser.parse_args()
    return args


class OnnxModel:
    def __init__(self, model_path):
        self.session = ort.InferenceSession(model_path)
        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [output.name for output in self.session.get_outputs()]

    def process(self, image_path):
        image = cv2.imread(image_path)
        assert image is not None, f"Failed to read input file: {image_path}"
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = image.transpose(2, 0, 1)
        new_w = 256
        new_h = 256
        image = F.resize(
            torch.from_numpy(image),
            [new_w, new_h],
            F.InterpolationMode.BILINEAR,
            None,
            True,
        )

        image = np.expand_dims(image.numpy(), axis=0)
        image = image.astype(np.float32) / 255.0

        outputs = self.session.run(self.output_names, {self.input_name: image})
        return outputs


def cos_sim(a, b):
    a = a.flatten()
    b = b.flatten()
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def get_outputs_cos_sim(vsx_outputs, onnx_outputs):

    return [cos_sim(vsx_outputs[i], onnx_outputs[i]) for i in range(len(vsx_outputs))]


if __name__ == "__main__":
    args = argument_parser()
    batch_size = 1
    model = ModelCV(
        args.model_prefix, args.vdsp_params, batch_size, args.device_id, args.hw_config
    )
    image_format = model.get_fusion_op_iimage_format()
    if args.dataset_root_dir == "":
        cv_image = cv2.imread(args.input_file)
        assert cv_image is not None, f"Failed to read input file: {args.input_file}"
        vsx_image = utils.cv_bgr888_to_vsximage(cv_image, image_format, args.device_id)
        output = model.process(vsx_image)
        np.savez(
            args.output_file,
            input_0=output[0],
            input_1=output[1],
            input_2=output[2],
        )
        print("++++++++++Inference result saved to", os.path.abspath(args.output_file))
    else:
        filelist = sorted(
            glob.glob(os.path.join(args.dataset_root_dir, args.split, "*", "*.png"))
        )
        onnx_model = OnnxModel(args.onnx_file)
        outputs_cos_sim = []
        for image_path in tqdm(filelist):
            # vsx output
            cv_image = cv2.imread(image_path)
            assert cv_image is not None, f"Failed to read input file: {image_path}"
            vsx_image = utils.cv_bgr888_to_vsximage(
                cv_image, image_format, args.device_id
            )
            vsx_outputs = model.process(vsx_image)
            vsx_outputs = [o.astype(np.float32) for o in vsx_outputs]
            # onnx output
            onnx_outputs = onnx_model.process(image_path)
            assert len(vsx_outputs) == len(
                onnx_outputs
            ), f"Outputs length mismatch for file: {image_path}"
            outputs_cos_sim.append(get_outputs_cos_sim(vsx_outputs, onnx_outputs))
        print("+++++++++Outputs Cosine Similarity:", np.mean(outputs_cos_sim, axis=0))
