import json
import os
import argparse
from rec_metric import TextRecMetric

def load_dataset(test_image_path):
    dataset =[]
    metafile = os.path.join(test_image_path, "metadata.jsonl")
    with open(metafile, "r") as f:
        for line in f:
            data = json.loads(line)
            dataset.append(data)
    return dataset

def load_pred(pred_file):
    predict = []
    with open(pred_file, "r", encoding='utf-8', errors='ignore') as f:
        lines = f.readlines()
        for line in lines:
            content = line.replace('\n','').split(' ')
            filename = content[0] + ".jpg"
            label = content[1] if len(content) > 1 else ""
            predict.append({"file_name": filename, "label": label})
    return predict

if __name__ == "__main__":
    parse = argparse.ArgumentParser(description="EVAL")
    parse.add_argument(
        "--test_image_path",
        type=str,
        default="/home/zhchen/vastpipe-samples/data/ppocr-v5/rec_test/",
    )
    parse.add_argument(
        "--pred_file",
        type=str,
        default="rec_pred_vacc.txt",
    )

    args = parse.parse_args()
    print(args)

    # dataset
    gt_data = load_dataset(args.test_image_path) # auto download from hf
    pred_data = load_pred(args.pred_file)

    pred_path_vacc = "rec_pred_vacc.txt"

    merged_data = {}
    for data in gt_data:
        filename = data.get("file_name")
        merged_data[filename] = {"gt": data.get("label", ""), "pred": ""}

    for data in pred_data:
        filename = data.get("file_name")
        if filename in merged_data:
            merged_data[filename]["pred"] = data.get("label", "")
        else:
            merged_data[filename] = {"gt": "", "pred": data.get("label", "")}

    content = []
    for filename, labels in merged_data.items():
        gt = labels["gt"]
        pred = labels["pred"]
        elapse = 0
        content.append(f"{pred}\t{gt}\t{elapse}")

    with open(pred_path_vacc, "w", encoding="utf-8") as f:
        for v in content:
            f.write(f"{v}\n")

    metric = TextRecMetric()
    metric_res = metric(pred_path_vacc)
    print('metric: ', metric_res)

