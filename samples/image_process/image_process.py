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
import os
import sys

current_file_path = os.path.dirname(os.path.abspath(__file__))
common_path = os.path.join(current_file_path, "../..")
sys.path.append(common_path)

import common.utils as utils


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
        default="",
        help="output file",
    )
    args = parser.parse_args()
    return args


def save_vsx_image(vsx_image, filename):
    image_np = vsx.as_numpy(vsx_image)


if __name__ == "__main__":
    args = argument_parser()
    vsx.set_device(args.device_id)

    cv_image = cv2.imread(args.input_file)
    try:
        cv_image.shape
    except:
        print(f"Failed to read input_file:{args.input_file}")
        exit()

    vsx_image_bgr_interleave = vsx.create_image(
        cv_image,
        vsx.ImageFormat.BGR_INTERLEAVE,
        cv_image.shape[1],
        cv_image.shape[0],
        args.device_id,
    )

    # cvtcolor sample
    # convert color to  rgb_planar yuv_nv12 gray
    vsx_image_rgb_planar = vsx.cvtcolor(
        vsx_image_bgr_interleave, vsx.ImageFormat.RGB_PLANAR
    )
    vsx_image_yuv_nv12 = vsx.cvtcolor(vsx_image_rgb_planar, vsx.ImageFormat.YUV_NV12)
    vsx_image_gray = vsx.cvtcolor(vsx_image_bgr_interleave, vsx.ImageFormat.GRAY)
    print(f"vsx_image_rgb_planar format is {vsx_image_rgb_planar.format}")
    print(f"vsx_image_yuv_nv12 format is {vsx_image_yuv_nv12.format}")
    print(f"vsx_image_gray format is {vsx_image_gray.format}")

    # resize sample
    # resize to other sizes
    vsx_image_resize_416_416 = vsx.resize(
        vsx_image_rgb_planar,
        vsx.ImageResizeType.BILINEAR_PILLOW,
        resize_width=416,
        resize_height=416,
    )
    vsx_image_resize_600_800 = vsx.resize(
        vsx_image_yuv_nv12,
        vsx.ImageResizeType.BILINEAR_CV,
        resize_width=600,
        resize_height=800,
    )
    print(
        f"vsx_image_resize_416_416 size is ( {vsx_image_resize_416_416.width} x  {vsx_image_resize_416_416.height} )"
    )
    print(
        f"vsx_image_resize_600_800 size is ( {vsx_image_resize_600_800.width} x  {vsx_image_resize_600_800.height} )"
    )

    # crop sample
    vsx_image_crop_224_224 = vsx.crop(vsx_image_bgr_interleave, (0, 0, 224, 224))
    print(
        f"vsx_image_crop_224_224 size is ( {vsx_image_crop_224_224.width} x  {vsx_image_crop_224_224.height} )"
    )

    # yuvflip sample, just support yuv_nv12 format image
    vsx_image_yuvflip_x_axis = vsx.yuvflip(
        vsx_image_yuv_nv12, vsx.ImageFlipType.FLIP_TYPE_X_AXIS
    )
    vsx_image_yuvflip_y_axis = vsx.yuvflip(
        vsx_image_yuv_nv12, vsx.ImageFlipType.FLIP_TYPE_Y_AXIS
    )
    print(
        f"vsx_image_yuvflip_x_axis size is ( {vsx_image_yuvflip_x_axis.width} x  {vsx_image_yuvflip_x_axis.height} )"
    )
    print(
        f"vsx_image_yuvflip_y_axis size is ( {vsx_image_yuvflip_y_axis.width} x  {vsx_image_yuvflip_y_axis.height} )"
    )

    # warpaffine sample
    vsx_image_warpaffine = vsx.warpaffine(
        vsx_image_yuv_nv12,
        matrix=[[0.5, 0, 0], [0, 0.5, 0]],
        mode=vsx.ImageWarpAffineMode.WARP_AFFINE_MODE_NEAREST,
        border_mode=vsx.ImagePaddingType.PADDING_TYPE_CONSTANT,
        value=(128, 128, 0),
    )
    print(
        f"vsx_image_warpaffine size is ( {vsx_image_warpaffine.width} x  {vsx_image_warpaffine.height} )"
    )

    # resize_copy_make_border sample, resize to (w x h ) = (512 x 5120 and padding to (w x h) = (600 x 800)
    # padding_edges = (top,bottom,left,right)
    vsx_image_resize_copy_make_border = vsx.resize_copy_make_border(
        image=vsx_image_bgr_interleave,
        resize_type=vsx.ImageResizeType.NEAREST,
        resize_width=512,
        resize_height=512,
        padding_type=vsx.ImagePaddingType.PADDING_TYPE_CONSTANT,
        padding_values=(128, 128, 0),
        padding_edges=(144, 144, 44, 44),
    )
    print(
        f"vsx_image_resize_copy_make_border size is ( {vsx_image_resize_copy_make_border.width} x  {vsx_image_resize_copy_make_border.height} )"
    )

    # batch_crop_resize sample ,crop image with rectangles and resize to specific size
    crop_rects = [[0, 0, 200, 200], [50, 100, 150, 200]]
    vsx_images_batch_crop_resize = vsx.batch_crop_resize(
        vsx_image_bgr_interleave,
        crop_rects,
        vsx.ImageResizeType.BILINEAR_CV,
        resize_width=224,
        resize_height=224,
    )
    assert len(vsx_images_batch_crop_resize) == len(crop_rects)
    for i, image in enumerate(vsx_images_batch_crop_resize):
        print(
            f"vsx_images_batch_crop_resize[{i}] size is ( {image.width} x  {image.height} )"
        )

    # scale sample
    scale_shape = [(224, 224), (416, 416), (800, 600)]
    vsx_images_scale = vsx.scale(
        vsx_image_yuv_nv12, vsx.ImageResizeType.BILINEAR, scale_shape
    )
    assert len(vsx_images_scale) == len(scale_shape)
    for i, image in enumerate(vsx_images_scale):
        print(f"vsx_images_scale[{i}] size is ( {image.width} x  {image.height} )")

    cv_mat_bgr888 = utils.vsximage_to_cv_bgr888(vsx_images_scale[2])
    cv2.imwrite(args.output_file, cv_mat_bgr888)
