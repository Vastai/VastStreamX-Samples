import vaststreamx as vsx

import cv2
import argparse
import numpy as np
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
        help="input file",
    )
    parser.add_argument(
        "--output_file",
        default="cvtcolor_result.jpg",
        help="output file",
    )
    parser.add_argument(
        "--cvtcolor_code",
        default="bgr2rgb_interleave2planar",
        help="cvtcolor code",
    )
    args = parser.parse_args()
    return args


def parse_cvtcolor_code(cvtcolor_code):
    if cvtcolor_code.upper() == "YUV2RGB_NV12":
        return (
            vsx.ImageFormat.YUV_NV12,
            vsx.ImageFormat.RGB_PLANAR,
            vsx.ColorCvtCode.YUV2RGB_NV12,
        )
    elif cvtcolor_code.upper() == "YUV2BGR_NV12":
        return (
            vsx.ImageFormat.YUV_NV12,
            vsx.ImageFormat.BGR_PLANAR,
            vsx.ColorCvtCode.YUV2BGR_NV12,
        )
    elif cvtcolor_code.upper() == "BGR2RGB":
        return (
            vsx.ImageFormat.BGR_PLANAR,
            vsx.ImageFormat.RGB_PLANAR,
            vsx.ColorCvtCode.BGR2RGB,
        )
    elif cvtcolor_code.upper() == "RGB2BGR":
        return (
            vsx.ImageFormat.RGB_PLANAR,
            vsx.ImageFormat.BGR_PLANAR,
            vsx.ColorCvtCode.RGB2BGR,
        )
    elif cvtcolor_code.upper() == "BGR2RGB_INTERLEAVE2PLANAR":
        return (
            vsx.ImageFormat.BGR_INTERLEAVE,
            vsx.ImageFormat.RGB_PLANAR,
            vsx.ColorCvtCode.BGR2RGB_INTERLEAVE2PLANAR,
        )
    elif cvtcolor_code.upper() == "RGB2BGR_INTERLEAVE2PLANAR":
        return (
            vsx.ImageFormat.RGB_INTERLEAVE,
            vsx.ImageFormat.BGR_PLANAR,
            vsx.ColorCvtCode.RGB2BGR_INTERLEAVE2PLANAR,
        )
    elif cvtcolor_code.upper() == "BGR2BGR_INTERLEAVE2PLANAR":
        return (
            vsx.ImageFormat.BGR_INTERLEAVE,
            vsx.ImageFormat.BGR_PLANAR,
            vsx.ColorCvtCode.BGR2BGR_INTERLEAVE2PLANAR,
        )
    elif cvtcolor_code.upper() == "RGB2RGB_INTERLEAVE2PLANAR":
        return (
            vsx.ImageFormat.RGB_INTERLEAVE,
            vsx.ImageFormat.RGB_PLANAR,
            vsx.ColorCvtCode.RGB2RGB_INTERLEAVE2PLANAR,
        )
    elif cvtcolor_code.upper() == "YUV2GRAY_NV12":
        return (
            vsx.ImageFormat.YUV_NV12,
            vsx.ImageFormat.GRAY,
            vsx.ColorCvtCode.YUV2GRAY_NV12,
        )
    elif cvtcolor_code.upper() == "BGR2GRAY_INTERLEAVE":
        return (
            vsx.ImageFormat.BGR_INTERLEAVE,
            vsx.ImageFormat.GRAY,
            vsx.ColorCvtCode.BGR2GRAY_INTERLEAVE,
        )
    elif cvtcolor_code.upper() == "BGR2GRAY_PLANAR":
        return (
            vsx.ImageFormat.BGR_PLANAR,
            vsx.ImageFormat.GRAY,
            vsx.ColorCvtCode.BGR2GRAY_PLANAR,
        )
    elif cvtcolor_code.upper() == "RGB2GRAY_INTERLEAVE":
        return (
            vsx.ImageFormat.RGB_INTERLEAVE,
            vsx.ImageFormat.GRAY,
            vsx.ColorCvtCode.RGB2GRAY_INTERLEAVE,
        )
    elif cvtcolor_code.upper() == "RGB2GRAY_PLANAR":
        return (
            vsx.ImageFormat.RGB_PLANAR,
            vsx.ImageFormat.GRAY,
            vsx.ColorCvtCode.RGB2GRAY_PLANAR,
        )
    elif cvtcolor_code.upper() == "RGB2YUV_NV12_PLANAR":
        return (
            vsx.ImageFormat.RGB_PLANAR,
            vsx.ImageFormat.YUV_NV12,
            vsx.ColorCvtCode.RGB2YUV_NV12_PLANAR,
        )
    elif cvtcolor_code.upper() == "BGR2YUV_NV12_PLANAR":
        return (
            vsx.ImageFormat.BGR_PLANAR,
            vsx.ImageFormat.YUV_NV12,
            vsx.ColorCvtCode.BGR2YUV_NV12_PLANAR,
        )
    else:
        assert False, f"Unsuport cvtcolor code: {cvtcolor_code}"


if __name__ == "__main__":
    args = argument_parser()

    input_format, output_format, cvtcolor_code = parse_cvtcolor_code(args.cvtcolor_code)

    vsx.set_device(args.device_id)

    cv_image = cv2.imread(args.input_file)
    assert cv_image is not None, f"Failed to read: {args.input_file}"

    vsx_image = utils.cv_bgr888_to_vsximage(cv_image, input_format, args.device_id)

    # Build operator
    op = vsx.BuildInOperator(vsx.OpType.SINGLE_OP_CVT_COLOR)

    op.set_attribute(
        {
            attr.IIMAGE_WIDTH: vsx_image.width,
            attr.IIMAGE_HEIGHT: vsx_image.height,
            attr.IIMAGE_WIDTH_PITCH: vsx_image.width,
            attr.IIMAGE_HEIGHT_PITCH: vsx_image.height,
            attr.OIMAGE_WIDTH: vsx_image.width,
            attr.OIMAGE_HEIGHT: vsx_image.height,
            attr.OIMAGE_WIDTH_PITCH: vsx_image.width,
            attr.OIMAGE_HEIGHT_PITCH: vsx_image.height,
            attr.COLOR_CVT_CODE: cvtcolor_code,
            attr.COLOR_SPACE: vsx.ImageColorSpace.COLOR_SPACE_BT601,
        }
    )

    # run
    outputs = op.execute(
        [vsx_image], [(output_format, vsx_image.width, vsx_image.height)]
    )

    output = outputs[0]

    out_image_cv = utils.vsximage_to_cv_bgr888(outputs[0])

    # write
    cv2.imwrite(args.output_file, out_image_cv)
    print(f"Write result to args.output_file")
