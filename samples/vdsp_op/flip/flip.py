import vaststreamx as vsx
import cv2
import argparse
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
        "--output_file",
        default="flip_resul.jpg",
        help="output image",
    )
    parser.add_argument(
        "--flip_type",
        default="x",
        help="flip type x or y",
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = argument_parser()
    vsx.set_device(args.device_id)
    cv_bgr888 = cv2.imread(args.input_file)
    assert cv_bgr888 is not None, f"Failed to read: {args.input_file}"
    height, width = cv_bgr888.shape[:2]
    cv_nv12 = utils.cv_bgr888_to_nv12(cv_bgr888)
    vsx_image = vsx.create_image(
        cv_nv12,
        vsx.ImageFormat.YUV_NV12,
        width,
        height,
        args.device_id,
    )
    # Build operator
    op = vsx.BuildInOperator(vsx.OpType.SINGLE_OP_FLIP)

    if args.flip_type == "x" or args.flip_type == "X":
        flip_type = vsx.ImageFlipType.FLIP_TYPE_X_AXIS
    elif args.flip_type == "y" or args.flip_type == "Y":
        flip_type = vsx.ImageFlipType.FLIP_TYPE_Y_AXIS
    else:
        print(f'Unsupport flip type: "{args.flip_type}", only support x or y')
        exit(-1)

    op.set_attribute(
        {
            attr.IIMAGE_FORMAT: vsx.ImageType.YUV_NV12,
            attr.IIMAGE_WIDTH: vsx_image.width,
            attr.IIMAGE_HEIGHT: vsx_image.height,
            attr.IIMAGE_WIDTH_PITCH: vsx_image.width,
            attr.IIMAGE_HEIGHT_PITCH: vsx_image.height,
            attr.OIMAGE_WIDTH: vsx_image.width,
            attr.OIMAGE_HEIGHT: vsx_image.height,
            attr.OIMAGE_WIDTH_PITCH: vsx_image.width,
            attr.OIMAGE_HEIGHT_PITCH: vsx_image.height,
            attr.DIRECTION: flip_type,
        }
    )
    # run
    output_list = op.execute(
        [vsx_image],
        [(vsx.ImageFormat.YUV_NV12, vsx_image.width, vsx_image.height)],
    )

    out_images = [vsx.as_numpy(output).squeeze() for output in output_list]
    out_bgr888 = utils.cv_nv12_to_bgr888(out_images[0])
    cv2.imwrite(args.output_file, out_bgr888)

    print(f"save result to {args.output_file}")
