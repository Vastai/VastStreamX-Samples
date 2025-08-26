import numpy as np
import argparse
from pycocotools.coco import COCO  # noqa
from pycocotools.cocoeval import COCOeval  # noqa

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="EVAL COCO_SEG")
    parser.add_argument(
        "--prediction_file",
        default="/path/to/infer/prediction/file",
    )
    parser.add_argument("--gt", type=str, default="/path/to/instances_val2017.json")
    args = parser.parse_args()

    try:
        anno = COCO(str(args.gt))  # init annotations api
        pred = anno.loadRes(args.prediction_file)  # init predictions api (must pass string, not Path)
        imgIds = anno.getImgIds()
        print("get %d images" % len(imgIds))
        imgIds = sorted(imgIds)
        for i, eval in enumerate(
            [COCOeval(anno, pred, "bbox"), COCOeval(anno, pred, "segm")]
        ):
            eval.params.imgIds = imgIds
            eval.evaluate()
            eval.accumulate()
            eval.summarize()

    except Exception as e:
        print(f"pycocotools unable to run: {e}")
