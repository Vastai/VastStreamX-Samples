import vaststreamx as vsx
import numpy as np
import cv2
import argparse
import ast
import os
import sys

current_file_path = os.path.dirname(os.path.abspath(__file__))
common_path = os.path.join(current_file_path, "../../..")
sys.path.append(common_path)
import common.utils as utils

attr = vsx.AttrKey


def argument_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-d",
        "--device_id",
        default=0,
        type=int,
        help="device id to run",
    )
    parser.add_argument(
        "--input_file",
        default="data/images/dog.jpg",
        help="input image",
    )
    parser.add_argument(
        "--output_size1", help="resize output size [w,h]", default="[256, 256]"
    )
    parser.add_argument(
        "--output_size2", help="resize output size [w,h]", default="[512, 512]"
    )
    parser.add_argument(
        "--output_file1",
        default="./scale_result1.jpg",
        help="output image",
    )
    parser.add_argument(
        "--output_file2",
        default="./scale_result2.jpg",
        help="output image",
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = argument_parser()
    vsx.set_device(args.device_id)
    output_size1 = ast.literal_eval(args.output_size1)
    output_size2 = ast.literal_eval(args.output_size2)

    cv_image = cv2.imread(args.input_file)
    assert cv_image is not None, f"Failed to read {args.input_file}"

    vsx_image = utils.cv_bgr888_to_vsximage(
        cv_image, vsx.ImageFormat.YUV_NV12, args.device_id
    )

    # Build operator
    op = vsx.BuildInOperator(vsx.OpType.SINGLE_OP_SCALE)

    op.set_attribute(
        {
            attr.IIMAGE_WIDTH: vsx_image.width,
            attr.IIMAGE_HEIGHT: vsx_image.height,
            attr.IIMAGE_WIDTH_PITCH: vsx_image.width,
            attr.IIMAGE_HEIGHT_PITCH: vsx_image.height,
            attr.RESIZE_TYPE: vsx.ImageResizeType.BILINEAR,
            attr.OIMAGE_CNT: 2,
            attr.OIMAGE_WIDTH: [output_size1[0], output_size2[0]],
            attr.OIMAGE_HEIGHT: [output_size1[1], output_size2[1]],
            attr.OIMAGE_WIDTH_PITCH: [output_size1[0], output_size2[0]],
            attr.OIMAGE_HEIGHT_PITCH: [output_size1[1], output_size2[1]],
        }
    )

    output_list = op.execute(
        [vsx_image],
        [
            (vsx.ImageFormat.YUV_NV12, output_size1[0], output_size1[1]),
            (vsx.ImageFormat.YUV_NV12, output_size2[0], output_size2[1]),
        ],
    )

    outs_rgb888 = [utils.vsximage_to_cv_bgr888(out) for out in output_list]
    cv2.imwrite(args.output_file1, outs_rgb888[0])
    cv2.imwrite(args.output_file2, outs_rgb888[1])
    print(f"save result to {args.output_file1}, {args.output_file2}")
