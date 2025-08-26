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
    parser.add_argument("--output_size", help="output size [w,h]", default="[256, 256]")
    parser.add_argument(
        "--matrix",
        help="warp affine matrix, [x0,x1,x2,y0,y1,y2] ",
        default="[0.7890625, -0.611328125, 56.0, 0.611328125, 0.7890625, -416.0]",
    )
    parser.add_argument(
        "--output_file",
        default="./warpaffine_result.jpg",
        help="output image",
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = argument_parser()
    vsx.set_device(args.device_id)
    output_size = ast.literal_eval(args.output_size)
    matrix = ast.literal_eval(args.matrix)

    cv_image = cv2.imread(args.input_file)
    assert cv_image is not None, f"Failed to read {args.input_file}"

    vsx_image = utils.cv_bgr888_to_vsximage(
        cv_image, vsx.ImageFormat.YUV_NV12, args.device_id
    )
    # Build operator
    op = vsx.BuildInOperator(vsx.OpType.SINGLE_OP_WARP_AFFINE)

    op.set_attribute(
        {
            attr.IIMAGE_FORMAT: vsx.ImageType.YUV_NV12,
            attr.IIMAGE_WIDTH: vsx_image.width,
            attr.IIMAGE_HEIGHT: vsx_image.height,
            attr.IIMAGE_WIDTH_PITCH: vsx_image.width,
            attr.IIMAGE_HEIGHT_PITCH: vsx_image.height,
            attr.OIMAGE_WIDTH: output_size[0],
            attr.OIMAGE_HEIGHT: output_size[1],
            attr.OIMAGE_WIDTH_PITCH: output_size[0],
            attr.OIMAGE_HEIGHT_PITCH: output_size[1],
            attr.FLAGS: vsx.ImageWarpAffineMode.WARP_AFFINE_MODE_BILINEAR,
            attr.BORDER_VALUE: [114, 114, 114],
            attr.BORDER_MODE: vsx.ImagePaddingType.PADDING_TYPE_CONSTANT,
            attr.M: matrix,
        }
    )
    output_list = op.execute(
        [vsx_image],
        [(vsx.ImageFormat.YUV_NV12, output_size[0], output_size[1])],
    )

    outs_rgb888 = [utils.vsximage_to_cv_bgr888(out) for out in output_list]
    cv2.imwrite(args.output_file, outs_rgb888[0])

    print(f"save result to {args.output_file}")
