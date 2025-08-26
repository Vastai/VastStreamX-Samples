import vaststreamx as vsx
from typing import Union, List
import numpy as np
import os
import sys

attr = vsx.AttrKey

current_file_path = os.path.dirname(os.path.abspath(__file__))
common_path = os.path.join(current_file_path, "../../..")
sys.path.append(common_path)

from common.model_profiler import ModelProfiler
from easydict import EasyDict as edict
import argparse
import ast
import common.utils as utils


def argument_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--op_config",
        default="",
        help="op config file",
    )
    parser.add_argument(
        "-d",
        "--device_ids",
        default="[0]",
        help="device ids to run",
    )
    parser.add_argument(
        "-i",
        "--instance",
        default=1,
        type=int,
        help="instance number for each device",
    )
    parser.add_argument(
        "--iterations",
        default=10240,
        type=int,
        help="iterations count for one profiling",
    )

    parser.add_argument(
        "--percentiles",
        default="[50, 90, 95, 99]",
        help="percentiles of latency",
    )
    parser.add_argument(
        "--input_host",
        default=0,
        type=int,
        help="cache input data into host memory",
    )
    args = parser.parse_args()
    return args


def cv_rgb_image_to_vastai(image_cv, device_id):
    assert len(image_cv.shape) >= 2
    h = image_cv.shape[0]
    w = image_cv.shape[1]
    if len(image_cv.shape) == 3:
        return vsx.create_image(
            image_cv, vsx.ImageFormat.BGR_INTERLEAVE, w, h, device_id
        )
    elif len(image_cv.shape) == 2:
        return vsx.create_image(image_cv, vsx.ImageFormat.GRAY, w, h, device_id)
    else:
        raise Exception("unsupported ndarray shape", image_cv.shape)


class BuildInOperatorProf:
    def __init__(self, op_config, device_id):
        self.device_id_ = device_id
        vsx.set_device(device_id)

        self.ops_ = vsx.Operator.load_ops_from_json_file(op_config)
        assert (
            len(self.ops_) == 1
        ), f"Only support 1 BuildIn Op. Now it's {len(self.ops_)}"
        self.buildin_op_ = self.ops_[0].cast_to_buildin_operator()

        assert (
            self.buildin_op_.op_type != vsx.OpType.BERT_EMBEDDING_OP
        ), "Unsupport operator BERT_EMBEDDING_OP"

        self.iimage_width_ = self.buildin_op_.get_attribute(attr.IIMAGE_WIDTH)
        self.iimage_height_ = self.buildin_op_.get_attribute(attr.IIMAGE_HEIGHT)
        self.iimage_width_pitch_ = self.buildin_op_.get_attribute(
            attr.IIMAGE_WIDTH_PITCH
        )
        self.iimage_height_pitch_ = self.buildin_op_.get_attribute(
            attr.IIMAGE_HEIGHT_PITCH
        )

        self.iimage_format_ = vsx.ImageFormat.YUV_NV12
        self.oimage_format_ = vsx.ImageFormat.YUV_NV12

        if "IIMAGE_FORMAT" in list(self.buildin_op_.attributes.keys()):
            imagetype = self.buildin_op_.get_attribute(attr.IIMAGE_FORMAT)
            self.iimage_format_ = utils.imagetype_to_vsxformat(imagetype)

        if "OIMAGE_FORMAT" in list(self.buildin_op_.attributes.keys()):
            imagetype = self.buildin_op_.get_attribute(attr.OIMAGE_FORMAT)
            self.oimage_format_ = utils.imagetype_to_vsxformat(imagetype)

        if self.buildin_op_.op_type == vsx.OpType.SINGLE_OP_CVT_COLOR:
            cvtcolor_code = self.buildin_op_.get_attribute(attr.COLOR_CVT_CODE)
            self.iimage_format_, self.oimage_format_, _ = (
                utils.cvtcolorcode_to_vsxformat(cvtcolor_code)
            )

        self.oimage_count_ = 1
        self.oimage_width_ = []
        self.oimage_height_ = []
        self.oimage_width_pitch_ = []
        self.oimage_height_pitch_ = []

        if self.buildin_op_.op_type == vsx.OpType.SINGLE_OP_BATCH_CROP_RESIZE:
            self.oimage_count_ = self.buildin_op_.get_attribute(attr.CROP_NUM)
            ow = self.buildin_op_.get_attribute(attr.OIMAGE_WIDTH)
            oh = self.buildin_op_.get_attribute(attr.OIMAGE_HEIGHT)
            owp = self.buildin_op_.get_attribute(attr.OIMAGE_WIDTH_PITCH)
            ohp = self.buildin_op_.get_attribute(attr.OIMAGE_HEIGHT_PITCH)
            for _ in range(self.oimage_count_):
                self.oimage_width_.append(ow)
                self.oimage_height_.append(oh)
                self.oimage_width_pitch_.append(owp)
                self.oimage_height_pitch_.append(ohp)
        elif self.buildin_op_.op_type == vsx.OpType.SINGLE_OP_SCALE:
            self.oimage_count_ = self.buildin_op_.get_attribute(attr.OIMAGE_CNT)
            for i in range(self.oimage_count_):
                ow = self.buildin_op_.get_attribute(attr.OIMAGE_WIDTH, i)
                oh = self.buildin_op_.get_attribute(attr.OIMAGE_HEIGHT, i)
                owp = self.buildin_op_.get_attribute(attr.OIMAGE_WIDTH_PITCH, i)
                ohp = self.buildin_op_.get_attribute(attr.OIMAGE_HEIGHT_PITCH, i)
                self.oimage_width_.append(ow)
                self.oimage_height_.append(oh)
                self.oimage_width_pitch_.append(owp)
                self.oimage_height_pitch_.append(ohp)
        elif self.buildin_op_.op_type == vsx.OpType.SINGLE_OP_COPY_MAKE_BORDER:
            ow = self.buildin_op_.get_attribute(attr.OIMAGE_WIDTH)
            oh = self.buildin_op_.get_attribute(attr.OIMAGE_HEIGHT)
            self.oimage_width_.append(ow)
            self.oimage_height_.append(oh)
        elif (
            self.buildin_op_.op_type
            == vsx.OpType.FUSION_OP_RGB_LETTERBOX_CVTCOLOR_NORM_TENSOR_EXT
        ):
            left = self.buildin_op_.get_attribute(attr.PADDING_LEFT)
            right = self.buildin_op_.get_attribute(attr.PADDING_RIGHT)
            top = self.buildin_op_.get_attribute(attr.PADDING_TOP)
            bottom = self.buildin_op_.get_attribute(attr.PADDING_BOTTOM)
            width = self.buildin_op_.get_attribute(attr.RESIZE_WIDTH)
            height = self.buildin_op_.get_attribute(attr.RESIZE_HEIGHT)
            self.oimage_width_.append(left + right + width)
            self.oimage_height_.append(top + bottom + height)
            self.oimage_width_pitch_.append(left + right + width)
            self.oimage_height_pitch_.append(top + bottom + height)
        elif self.buildin_op_.op_type == vsx.OpType.FUSION_OP_RGB_CVTCOLOR_NORM_TENSOR:
            self.oimage_width_.append(self.iimage_width_)
            self.oimage_height_.append(self.iimage_height_)
            self.oimage_width_pitch_.append(self.iimage_width_)
            self.oimage_height_pitch_.append(self.iimage_height_)
            self.iimage_format_ = vsx.ImageFormat.BGR_PLANAR
            self.oimage_format_ = vsx.ImageFormat.RGB_PLANAR
        elif (
            self.buildin_op_.op_type
            >= vsx.OpType.FUSION_OP_YUV_NV12_RESIZE_2RGB_NORM_TENSOR
        ):
            ow = self.buildin_op_.get_attribute(attr.OIMAGE_WIDTH)
            oh = self.buildin_op_.get_attribute(attr.OIMAGE_HEIGHT)
            self.oimage_width_.append(ow)
            self.oimage_height_.append(oh)
            self.oimage_width_pitch_.append(ow)
            self.oimage_height_pitch_.append(oh)
        else:
            ow = self.buildin_op_.get_attribute(attr.OIMAGE_WIDTH)
            oh = self.buildin_op_.get_attribute(attr.OIMAGE_HEIGHT)
            owp = self.buildin_op_.get_attribute(attr.OIMAGE_WIDTH_PITCH)
            ohp = self.buildin_op_.get_attribute(attr.OIMAGE_HEIGHT_PITCH)
            self.oimage_width_.append(ow)
            self.oimage_height_.append(oh)
            self.oimage_width_pitch_.append(owp)
            self.oimage_height_pitch_.append(ohp)

        self.odata_type_ = vsx.TypeFlag.UINT8
        if (
            self.buildin_op_.op_type
            >= vsx.OpType.FUSION_OP_YUV_NV12_RESIZE_2RGB_NORM_TENSOR
        ):
            self.odata_type_ = vsx.TypeFlag.FLOAT16
            self.oimage_format_ = vsx.ImageFormat.RGB_PLANAR

        # self.PrintParams()

    def process(
        self, input: Union[List[np.ndarray], List[vsx.Image], np.ndarray, vsx.Image]
    ):
        if isinstance(input, list):
            if isinstance(input[0], np.ndarray):
                return self.process(
                    [cv_rgb_image_to_vastai(x, self.device_id_) for x in input]
                )
            else:
                return self.process_impl(input)
        return self.process([input])[0]

    def get_test_data(self, dtype, input_shape, batch_size, ctx="VACC"):
        assert len(input_shape) >= 2
        height = input_shape[-2]
        width = input_shape[-1]

        if self.iimage_format_ == vsx.ImageFormat.YUV_NV12:
            dummy_cv = np.random.randn(height * 3 // 2, width).astype(dtype)
        elif self.iimage_format_ == vsx.ImageFormat.RGB_INTERLEAVE:
            dummy_cv = np.random.randn(height, width * 3).astype(dtype)
        elif self.iimage_format_ == vsx.ImageFormat.BGR_INTERLEAVE:
            dummy_cv = np.random.randn(height, width * 3).astype(dtype)
        elif self.iimage_format_ == vsx.ImageFormat.GRAY:
            dummy_cv = np.random.randn(height, width).astype(dtype)
        else:
            dummy_cv = np.random.randn(height, width, 3).astype(dtype)

        if ctx == "CPU":
            return [dummy_cv] * batch_size
        else:

            vacc_dummy = vsx.create_image(
                dummy_cv, self.iimage_format_, width, height, self.device_id_
            )
            return [vacc_dummy] * batch_size

    def process_impl(self, inputs):
        scale = 1
        if self.odata_type_ == vsx.TypeFlag.FLOAT16:
            scale = 2
        output_info = [
            (self.oimage_format_, ow * scale, oh)
            for ow, oh in zip(self.oimage_width_, self.oimage_height_)
        ] * len(inputs)
        outs = self.buildin_op_.execute(
            images=inputs,
            output_info=output_info,
        )
        return outs

    def PrintParams(self):
        print(f"op_type:{self.buildin_op_.op_type}")
        print(f"self.attributes:{self.buildin_op_.attributes}")
        print(f"attribute keys:{list(self.buildin_op_.attributes.keys())}")
        print(f"self.iimage_width_:{self.iimage_width_}")
        print(f"self.iimage_height_:{self.iimage_height_}")
        print(f"self.iimage_width_pitch_:{self.iimage_width_pitch_}")
        print(f"self.iimage_height_pitch_:{self.iimage_height_pitch_}")
        print(f"self.oimage_width_:{self.oimage_width_}")
        print(f"self.oimage_height_:{self.oimage_height_}")
        print(f"self.oimage_width_pitch_:{self.oimage_width_pitch_}")
        print(f"self.oimage_height_pitch_:{self.oimage_height_pitch_}")
        print(f"self.oimage_count_:{self.oimage_count_}")
        print(f"self.iimage_format_:{self.iimage_format_}")
        print(f"self.oimage_format_:{self.oimage_format_}")
        print(f"self.odata_type_:{self.odata_type_}")


if __name__ == "__main__":
    args = argument_parser()
    op_config = args.op_config
    device_ids = ast.literal_eval(args.device_ids)
    instance = args.instance
    iterations = args.iterations
    queue_size = 0
    batch_size = 1
    input_host = args.input_host
    percentiles = ast.literal_eval(args.percentiles)

    ops = []
    contexts = []

    for i in range(instance):
        device_id = device_ids[i % len(device_ids)]
        op = BuildInOperatorProf(op_config=op_config, device_id=device_id)
        ops.append(op)
        if input_host:
            contexts.append("CPU")
        else:
            contexts.append("VACC")

    shape = (1, 3, ops[0].iimage_height_, ops[0].iimage_width_)
    config = edict(
        {
            "instance": instance,
            "iterations": iterations,
            "batch_size": batch_size,
            "data_type": "uint8",
            "device_ids": device_ids,
            "contexts": contexts,
            "input_shape": shape,
            "percentiles": percentiles,
            "queue_size": queue_size,
        }
    )
    profiler = ModelProfiler(config, ops)
    print(profiler.profiling())
