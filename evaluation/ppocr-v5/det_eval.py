import os
import glob
import argparse
import numpy as np
from tqdm import tqdm
import cv2
import sys
import numpy as np
from det_metric import TextDetMetric
import json

# load npz
def load_npz(npz_file):
    stream_ouput = []
    for i in range(1):
        stream_ouput.append(np.load(npz_file, allow_pickle=True)["output_" + str(i)])
        # print(stream_ouput[i].dtype , stream_ouput[i].shape)
    return stream_ouput

def load_dataset(test_image_path):
    dataset =[]
    metafile = os.path.join(test_image_path, "metadata.jsonl")
    with open(metafile, "r") as f:
        for line in f:
            data = json.loads(line)
            dataset.append(data)
    return dataset

if __name__ == "__main__":

    parse = argparse.ArgumentParser(description="EVAL")
    parse.add_argument(
        "--test_image_path",
        type=str,
        default="/opt/vastai/vaststreamx/data/datasets/ch4_test_images",
    )
    parse.add_argument(
        "--boxes_npz_dir",
        type=str,
        default="./boxes_npz",
    )

    args = parse.parse_args()
    print(args)

    gt_data = load_dataset(args.test_image_path) # auto download from hf

    content_vacc = []
    pred_vacc_path = "det_pred_vacc.txt"

    boxes_npz_list = glob.glob(args.boxes_npz_dir + "/*.npz")

    # infer image
    for i, one_data in enumerate(tqdm(gt_data, desc="infer images")):
        filename = one_data.get("file_name")
        id_name = os.path.splitext(os.path.basename(filename))[0]
        npz_file_path = os.path.join(args.boxes_npz_dir, id_name + ".npz")
        if npz_file_path not in boxes_npz_list:
            print(id_name)
            print("not exist")
            boxes_numpy = np.array([])
        else:
            boxes_numpy = load_npz(npz_file_path)[0]

        dt_boxes_vacc = [] if boxes_numpy is None else boxes_numpy.tolist()

        elapse = 0
        gt_boxes = [v["points"] for v in one_data["shapes"]]

        content_vacc.append(f"{dt_boxes_vacc}\t{gt_boxes}\t{elapse}")



    with open(pred_vacc_path, "w", encoding="utf-8") as f:
        for v in content_vacc:
            f.write(f"{v}\n")

    # eval metric
    metric = TextDetMetric()
    metric_res = metric(pred_vacc_path)
    print('metric: ', metric_res)
