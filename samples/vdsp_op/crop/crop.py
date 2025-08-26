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
        "--crop_rect",
        default="[50,70,131,230]",
        help="crop rect [x,y,w,h]",
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
    op = vsx.BuildInOperator(vsx.OpType.SINGLE_OP_CROP)
    crop_rect = ast.literal_eval(args.crop_rect)

    crop_x, crop_y, crop_w, crop_h = crop_rect

    op.set_attribute(
        {
            attr.IIMAGE_FORMAT: vsx.ImageType.BGR888,
            attr.IIMAGE_WIDTH: vsx_image.width,
            attr.IIMAGE_HEIGHT: vsx_image.height,
            attr.IIMAGE_WIDTH_PITCH: vsx_image.width,
            attr.IIMAGE_HEIGHT_PITCH: vsx_image.height,
            attr.OIMAGE_WIDTH: crop_w,
            attr.OIMAGE_HEIGHT: crop_h,
            attr.OIMAGE_WIDTH_PITCH: crop_w,
            attr.OIMAGE_HEIGHT_PITCH: crop_h,
            attr.CROP_X: crop_x,
            attr.CROP_Y: crop_y,
        }
    )
    # run
    output_list = op.execute(
        [vsx_image],
        [(vsx.ImageFormat.BGR_INTERLEAVE, crop_w, crop_h)],
    )

    out_images = [vsx.as_numpy(output).squeeze() for output in output_list]
    cv2.imwrite(args.output_file, out_images[0])

    print(f"save result to {args.output_file}")
