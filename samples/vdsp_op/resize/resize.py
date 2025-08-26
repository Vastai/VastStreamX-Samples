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
        "--output_size", help="resize output size", default="[256, 256]"
    )
    parser.add_argument(
        "--output_file",
        default="./resize_result.jpg",
        help="output image",
    )
    args = parser.parse_args()
    return args


def resize_bgr888_to_bgr888(args):
    """
    resize image, output image format is the same as input image format
    """
    cv_image = cv2.imread(args.input_file)
    assert cv_image is not None, f"Failed to read {args.input_file}"
    vsx_image = vsx.create_image(
        cv_image,
        vsx.ImageFormat.BGR_INTERLEAVE,
        cv_image.shape[1],
        cv_image.shape[0],
        args.device_id,
    )
    # Build operator
    op = vsx.BuildInOperator(vsx.OpType.SINGLE_OP_RESIZE)

    output_size = ast.literal_eval(args.output_size)
    # set attri
    out_width = output_size[0]
    out_height = output_size[1]

    op.set_attribute(
        {
            attr.IIMAGE_WIDTH: vsx_image.width,
            attr.IIMAGE_HEIGHT: vsx_image.height,
            attr.IIMAGE_WIDTH_PITCH: vsx_image.width,
            attr.IIMAGE_HEIGHT_PITCH: vsx_image.height,
            attr.OIMAGE_WIDTH: out_width,
            attr.OIMAGE_HEIGHT: out_height,
            attr.OIMAGE_WIDTH_PITCH: out_width,
            attr.OIMAGE_HEIGHT_PITCH: out_height,
            attr.IIMAGE_FORMAT: vsx.ImageType.RGB888,
            attr.OIMAGE_FORMAT: vsx.ImageType.RGB888,
            attr.RESIZE_TYPE: vsx.ImageResizeType.BILINEAR_CV,
        }
    )

    # run
    output_list = op.execute(
        [vsx_image], [(vsx.ImageFormat.BGR_INTERLEAVE, out_width, out_height)]
    )

    out_image_np = vsx.as_numpy(output_list[0]).squeeze()
    # write
    out_image = out_image_np.reshape((out_height, out_width, 3))
    cv2.imwrite(args.output_file, out_image)
    print("save result to ", args.output_file)


def resize_bgr888_to_bgr_planar(args):
    """
    resize image, and cvtcolor from bgr888 to bgr_planar
    """
    cv_image = cv2.imread(args.input_file)
    assert cv_image is not None, f"Failed to read {args.input_file}"
    vsx_image = vsx.create_image(
        cv_image,
        vsx.ImageFormat.BGR_INTERLEAVE,
        cv_image.shape[1],
        cv_image.shape[0],
        args.device_id,
    )
    # Build operator
    op = vsx.BuildInOperator(vsx.OpType.SINGLE_OP_RESIZE)

    output_size = ast.literal_eval(args.output_size)
    # set attri
    out_width = output_size[0]
    out_height = output_size[1]

    op.set_attribute(
        {
            attr.IIMAGE_WIDTH: vsx_image.width,
            attr.IIMAGE_HEIGHT: vsx_image.height,
            attr.IIMAGE_WIDTH_PITCH: vsx_image.width,
            attr.IIMAGE_HEIGHT_PITCH: vsx_image.height,
            attr.OIMAGE_WIDTH: out_width,
            attr.OIMAGE_HEIGHT: out_height,
            attr.OIMAGE_WIDTH_PITCH: out_width,
            attr.OIMAGE_HEIGHT_PITCH: out_height,
            attr.IIMAGE_FORMAT: vsx.ImageType.RGB888,
            attr.OIMAGE_FORMAT: vsx.ImageType.RGB_PLANAR,
            attr.RESIZE_TYPE: vsx.ImageResizeType.BILINEAR_CV,
        }
    )

    # run
    output_list = op.execute(
        [vsx_image], [(vsx.ImageFormat.BGR_PLANAR, out_width, out_height)]
    )

    out_image_np = vsx.as_numpy(output_list[0]).squeeze()
    # write
    out_image = out_image_np.transpose((1, 2, 0))  # BGR_PLANAR to BGR_INTERLEAVE
    cv2.imwrite(args.output_file, out_image)
    print("save result to ", args.output_file)


if __name__ == "__main__":
    args = argument_parser()
    vsx.set_device(args.device_id)
    resize_bgr888_to_bgr888(args)
    # resize_bgr888_to_bgr_planar(args)
