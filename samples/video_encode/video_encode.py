import os
import sys
from threading import Thread
import vaststreamx as vsx

current_file_path = os.path.dirname(os.path.abspath(__file__))
common_path = os.path.join(current_file_path, "../..")
sys.path.append(common_path)

from common.vencoder import Vencoder
import argparse


def read_frames(input_file, width, height):
    frames = []
    frames_num = 0
    with open(input_file, "rb") as file:
        while True:
            data = file.read((int)(width * height * 3 / 2))
            if len(data) == 0:
                break
            frames.append(data[:])
            frames_num += 1
    return frames, frames_num


def argument_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--codec_type",
        default="H264",
        help="hw-config file of the model suite",
    )
    parser.add_argument(
        "-d",
        "--device_id",
        default=0,
        type=int,
        help="device id to run",
    )
    parser.add_argument(
        "--width",
        default=0,
        type=int,
        help="frame width",
    )
    parser.add_argument(
        "--height",
        default=0,
        type=int,
        help="frame height",
    )
    parser.add_argument(
        "--frame_rate",
        default=30,
        type=int,
        help="frame rate",
    )
    parser.add_argument(
        "--input_file",
        default="",
        help="video file",
    )
    parser.add_argument(
        "--output_file",
        default="",
        help="output file",
    )

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = argument_parser()
    codec_type = vsx.CodecType.CODEC_TYPE_H264
    if args.codec_type == "H264" or args.codec_type == "h264":
        codec_type = vsx.CodecType.CODEC_TYPE_H264
    elif args.codec_type == "H265" or args.codec_type == "h265":
        codec_type = vsx.CodecType.CODEC_TYPE_H265
    elif args.codec_type == "AV1" or args.codec_type == "av1":
        codec_type = vsx.CodecType.CODEC_TYPE_AV1
    else:
        print(f"undefined codec_type:{args.codec_type}")
        exit(-1)

    input_file = args.input_file
    output_file = args.output_file

    frames, frames_num = read_frames(input_file, args.width, args.height)

    vencoder = Vencoder(
        codec_type,
        args.device_id,
        frames,
        frames_num,
        args.width,
        args.height,
        vsx.ImageFormat.YUV_NV12,
        args.frame_rate,
    )

    def get_frame_thread():
        file = open(output_file, "wb")
        index = 0
        while True:
            try:
                video_data = vencoder.get_result()
                index += 1
                file.write(video_data)
            except ValueError as e:
                break

        file.close()

    recv_thread = Thread(target=get_frame_thread)
    recv_thread.start()

    while True:
        media_data = vencoder.get_test_data(False)
        if media_data is None:
            vencoder.stop()
            break
        else:
            vencoder.process(media_data)

    recv_thread.join()
