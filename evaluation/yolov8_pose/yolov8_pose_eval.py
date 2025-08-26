import argparse
from pycocotools.coco import COCO  # noqa
from pycocotools.cocoeval import COCOeval  # noqa

parse = argparse.ArgumentParser(description="EVAL YOLOV8_POSE")
parse.add_argument("--gt", type=str, default="coco/annotations/person_keypoints_val2017.json")
parse.add_argument("--pred", type=str, default="TEMP/predictions.json")
args = parse.parse_args()
try:  
    
    anno = COCO(str(args.gt))  # init annotations api
    pred = anno.loadRes(str(args.pred))  # init predictions api (must pass string, not Path)
    imgIds = anno.getImgIds()
    print("get %d images" % len(imgIds))
    imgIds = sorted(imgIds)
    for i, eval in enumerate([COCOeval(anno, pred, 'bbox'), COCOeval(anno, pred, 'keypoints')]):
        eval.params.imgIds = imgIds
        eval.evaluate()
        eval.accumulate()
        eval.summarize()
        # idx = i * 4 + 2
        # stats[self.metrics.keys[idx + 1]], stats[
        #     self.metrics.keys[idx]] = eval.stats[:2]  # update mAP50-95 and mAP50
except Exception as e:
    print(f'pycocotools unable to run: {e}')

