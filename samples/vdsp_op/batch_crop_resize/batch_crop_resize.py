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
        "--output_size", help="resize output size [w,h]", default="[512, 512]"
    )
    parser.add_argument(
        "--output_file1",
        default="batch_crop_resize_result1.jpg",
        help="output image 1",
    )
    parser.add_argument(
        "--output_file2",
        default="batch_crop_resize_result2.jpg",
        help="output image 2 ",
    )
    parser.add_argument(
        "--crop_rect1",
        default="[50,70,131,230]",
        help="crop rect [x,y,w,h]",
    )
    parser.add_argument(
        "--crop_rect2",
        default="[60,90,150,211]",
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
    op = vsx.BuildInOperator(vsx.OpType.SINGLE_OP_BATCH_CROP_RESIZE)

    output_size = ast.literal_eval(args.output_size)
    crop_rect1 = ast.literal_eval(args.crop_rect1)
    crop_rect2 = ast.literal_eval(args.crop_rect2)

    rect_num = 2
    op.set_attribute(
        {
            attr.IIMAGE_FORMAT: vsx.ImageType.BGR888,
            attr.IIMAGE_WIDTH: vsx_image.width,
            attr.IIMAGE_HEIGHT: vsx_image.height,
            attr.IIMAGE_WIDTH_PITCH: vsx_image.width,
            attr.IIMAGE_HEIGHT_PITCH: vsx_image.height,
            attr.OIMAGE_WIDTH: output_size[0],
            attr.OIMAGE_HEIGHT: output_size[1],
            attr.OIMAGE_WIDTH_PITCH: output_size[0],
            attr.OIMAGE_HEIGHT_PITCH: output_size[1],
            attr.RESIZE_TYPE: vsx.ImageResizeType.BILINEAR_CV,
            attr.CROP_NUM: rect_num,
            attr.CROP_X: [crop_rect1[0], crop_rect2[0]],
            attr.CROP_Y: [crop_rect1[1], crop_rect2[1]],
            attr.CROP_WIDTH: [crop_rect1[2], crop_rect2[2]],
            attr.CROP_HEIGHT: [crop_rect1[3], crop_rect2[3]],
        }
    )
    print(f"get attribute(attr.CROP_NUM):{op.get_attribute(attr.CROP_NUM)}")
    # run
    output_list = op.execute(
        [vsx_image],
        [(vsx.ImageFormat.BGR_INTERLEAVE, output_size[0], output_size[1])] * rect_num,
    )

    out_images = [vsx.as_numpy(output).squeeze() for output in output_list]
    cv2.imwrite(args.output_file1, out_images[0])
    cv2.imwrite(args.output_file2, out_images[1])
    print(f"save result to {args.output_file1}, {args.output_file2}")
