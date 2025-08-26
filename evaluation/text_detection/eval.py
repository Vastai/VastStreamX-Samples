import os
import glob
import argparse
import numpy as np
from tqdm import tqdm
from det_metric import DetMetric
from label_ops import DetLabelEncode
import cv2
import sys
import numpy as np

# load npz
def load_npz(npz_file):
    stream_ouput = []
    for i in range(1):
        stream_ouput.append(np.load(npz_file, allow_pickle=True)["output_" + str(i)])
        # print(stream_ouput[i].dtype , stream_ouput[i].shape)
    return stream_ouput


if __name__ == "__main__":

    parse = argparse.ArgumentParser(description="EVAL")
    parse.add_argument(
        "--test_image_path",
        type=str,
        default="/home/zhchen/vastpipe-samples/data/ch4_test_images",
    )
    parse.add_argument(
        "--boxes_npz_dir",
        type=str,
        default="/home/zhchen/vastpipe-samples/build/boxes_npz",
    )
    parse.add_argument(
        "--label_file",
        type=str,
        default="/home/zhchen/vastpipe-samples/samples/text_detection/calc_accrate/test_icdar2015_label.txt",
    )
    parse.add_argument("--draw_image", action="store_true")

    args = parse.parse_args()
    print(args)

    # 禁止显示 RuntimeWarning

    eval_class = DetMetric()
    det_lable_encode = DetLabelEncode()

    result_map = {}

    file_list = glob.glob(args.test_image_path + "/*.jpg")
    boxes_npz_list = glob.glob(args.boxes_npz_dir + "/*.npz")
    for image_path in file_list:
        # TODO : 获取文件名
        # print(image_path)
        id_name = os.path.splitext(os.path.basename(image_path))[0]
        # print(id_name)
        npz_file_path = os.path.join(args.boxes_npz_dir, id_name + ".npz")
        # print(npz_file_path)
        if npz_file_path not in boxes_npz_list:
            print(id_name)
            print("not exist")
            boxes_numpy = np.array([])
        else:
            boxes_numpy = load_npz(npz_file_path)[0]
        result_map[id_name + ".jpg"] = boxes_numpy

        # draw pic
        if args.draw_image:
            image = cv2.imread(image_path)
            for bbox in result_map[id_name + ".jpg"]:
                for i in range(len(bbox)):
                    t = (i + 1) % len(bbox)
                    pt1 = (bbox[i][0], bbox[i][1])
                    pt2 = (bbox[t][0], bbox[t][1])
                    cv2.line(image, pt1, pt2, color=(0, 0, 255))
            cv2.imwrite(os.path.join(args.boxes_npz_dir, id_name + ".jpg"), image)

    # eval
    with open(args.label_file, "r") as f:
        for line in tqdm(f.readlines(), desc="calc metric",file=sys.stdout):
            substr = line.strip("\n").split("\t")
            file_name = substr[0].split("/")[-1]
            # print("file_name: " ,  file_name)
            lable_str = substr[1]
            gt_polyons, _, ignore_tags = det_lable_encode(lable_str)
            det_polys = result_map[file_name]
            # print("det_ploys" , det_polys)
            eval_class(det_polys, gt_polyons, ignore_tags)

    metric = eval_class.get_metric()
    print("metric: ", metric)
    ########################################################################################################

""" 
dbnet_mobilenet_v3-int8-kl_divergence-3_736_1280-vacc
metric:  {'precision': 0.7654723127035831, 'recall': 0.6788637457871931, 'hmean': 0.7195713192140852}

dbnet_resnet50_vd-int8-kl_divergence-3_736_1280-vacc
metric:  {'precision': 0.8387909319899244, 'recall': 0.8016369764082811, 'hmean': 0.8197932053175776}
"""
