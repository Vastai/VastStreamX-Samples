#
# Copyright (C) 2025 Vastai-tech Company.
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
import vaststreamx as vsx
import numpy as np
import cv2
import argparse
import ast

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
        "--output_file",
        default="crop_result.jpg",
        help="output image 1",
    )
    parser.add_argument(
        "--output_size",
        default="[640,640]",
        help="output size [width,height]",
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = argument_parser()
    vsx.set_device(args.device_id)
    cv_image = cv2.imread(args.input_file)
    assert cv_image is not None, f"Failed to read: {args.input_file}"
    vsx_image = vsx.create_image(
        cv_image,
        vsx.ImageFormat.BGR_INTERLEAVE,
        cv_image.shape[1],
        cv_image.shape[0],
        args.device_id,
    )
    # Build operator
    op = vsx.BuildInOperator(vsx.OpType.SINGLE_OP_COPY_MAKE_BORDER)

    output_size = ast.literal_eval(args.output_size)
    assert len(output_size) == 2

    oimage_width, oimage_height = output_size
    iimage_height, iimage_width = cv_image.shape[:2]

    scale_w = oimage_width / iimage_width
    scale_h = oimage_height / iimage_height
    scale = scale_w if scale_w < scale_h else scale_h

    left = int((oimage_width - iimage_width * scale) * 0.5)
    right = int(oimage_width - iimage_width * scale - left)
    top = int((oimage_height - iimage_height * scale) * 0.5)
    bottom = int(oimage_height - iimage_height * scale - top)
    print(f"left:{left},right:{right},top:{top},bottom:{bottom}")
    op.set_attribute(
        {
            attr.IIMAGE_FORMAT: vsx.ImageType.BGR888,
            attr.IIMAGE_WIDTH: iimage_width,
            attr.IIMAGE_HEIGHT: iimage_height,
            attr.IIMAGE_WIDTH_PITCH: iimage_width,
            attr.IIMAGE_HEIGHT_PITCH: iimage_height,
            attr.OIMAGE_WIDTH: oimage_width,
            attr.OIMAGE_HEIGHT: oimage_height,
            attr.RESIZE_TYPE: vsx.ImageResizeType.BILINEAR_CV,
            attr.PADDING: [114, 114, 114],
            attr.EDGE_LEFT: left,
            attr.EDGE_RIGHT: right,
            attr.EDGE_TOP: top,
            attr.EDGE_BOTTOM: bottom,
        }
    )
    # run
    output_list = op.execute(
        [vsx_image],
        [(vsx.ImageFormat.BGR_INTERLEAVE, oimage_width, oimage_height)],
    )

    out_images = [vsx.as_numpy(output).squeeze() for output in output_list]
    cv2.imwrite(args.output_file, out_images[0])

    print(f"save result to {args.output_file}")
