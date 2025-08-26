#! /bin/bash

DEVICE=0

if [ $# -lt 1 ]; then  
  echo "error, Usage: bash precision_py.sh [model_prefix] [device_id=0]"  
  exit 1  
fi  

MODEL_PREFIX=$1

if [ $# -eq 2 ]; then  
    DEVICE=$2
fi  

mkdir -p mot_output
rm -f mot_output/*

python3 bytetracker.py  \
-m $MODEL_PREFIX \
--vdsp_params ../../data/configs/bytetrack_rgbplanar.json \
--device_id $DEVICE \
--detect_threshold 0.01 \
--track_thresh 0.6 \
--track_buffer 30 \
--match_thresh 0.9 \
--min_box_area 100 \
--label_file ../../data/labels/coco2id.txt \
--dataset_filelist /opt/vastai/vaststreamx/data/datasets/mot17/MOT17-02-FRCNN-filelist.txt \
--dataset_root /opt/vastai/vaststreamx/data/datasets/mot17/ \
--dataset_result_file ./mot_output/MOT17-02-FRCNN.txt 


python3 bytetracker.py  \
-m $MODEL_PREFIX \
--vdsp_params ../../data/configs/bytetrack_rgbplanar.json \
--device_id $DEVICE \
--detect_threshold 0.01 \
--track_thresh 0.6 \
--track_buffer 30 \
--match_thresh 0.9 \
--min_box_area 100 \
--label_file ../../data/labels/coco2id.txt \
--dataset_filelist /opt/vastai/vaststreamx/data/datasets/mot17/MOT17-04-FRCNN-filelist.txt \
--dataset_root /opt/vastai/vaststreamx/data/datasets/mot17/ \
--dataset_result_file ./mot_output/MOT17-04-FRCNN.txt 

python3 bytetracker.py  \
-m $MODEL_PREFIX \
--vdsp_params ../../data/configs/bytetrack_rgbplanar.json \
--device_id $DEVICE \
--detect_threshold 0.01 \
--track_thresh 0.6 \
--track_buffer 14 \
--match_thresh 0.9 \
--min_box_area 100 \
--label_file ../../data/labels/coco2id.txt \
--dataset_filelist /opt/vastai/vaststreamx/data/datasets/mot17/MOT17-05-FRCNN-filelist.txt \
--dataset_root /opt/vastai/vaststreamx/data/datasets/mot17/ \
--dataset_result_file ./mot_output/MOT17-05-FRCNN.txt 


python3 bytetracker.py  \
-m $MODEL_PREFIX \
--vdsp_params ../../data/configs/bytetrack_rgbplanar.json \
--device_id $DEVICE \
--detect_threshold 0.01 \
--track_thresh 0.6 \
--track_buffer 30 \
--match_thresh 0.9 \
--min_box_area 100 \
--label_file ../../data/labels/coco2id.txt \
--dataset_filelist /opt/vastai/vaststreamx/data/datasets/mot17/MOT17-09-FRCNN-filelist.txt \
--dataset_root /opt/vastai/vaststreamx/data/datasets/mot17/ \
--dataset_result_file ./mot_output/MOT17-09-FRCNN.txt


python3 bytetracker.py  \
-m $MODEL_PREFIX \
--vdsp_params ../../data/configs/bytetrack_rgbplanar.json \
--device_id $DEVICE \
--detect_threshold 0.01 \
--track_thresh 0.6 \
--track_buffer 30 \
--match_thresh 0.9 \
--min_box_area 100 \
--label_file ../../data/labels/coco2id.txt \
--dataset_filelist /opt/vastai/vaststreamx/data/datasets/mot17/MOT17-10-FRCNN-filelist.txt \
--dataset_root /opt/vastai/vaststreamx/data/datasets/mot17/ \
--dataset_result_file ./mot_output/MOT17-10-FRCNN.txt 


python3 bytetracker.py  \
-m $MODEL_PREFIX \
--vdsp_params ../../data/configs/bytetrack_rgbplanar.json \
--device_id $DEVICE \
--detect_threshold 0.01 \
--track_thresh 0.6 \
--track_buffer 30 \
--match_thresh 0.9 \
--min_box_area 100 \
--label_file ../../data/labels/coco2id.txt \
--dataset_filelist /opt/vastai/vaststreamx/data/datasets/mot17/MOT17-11-FRCNN-filelist.txt \
--dataset_root /opt/vastai/vaststreamx/data/datasets/mot17/ \
--dataset_result_file ./mot_output/MOT17-11-FRCNN.txt 

python3 bytetracker.py  \
-m $MODEL_PREFIX \
--vdsp_params ../../data/configs/bytetrack_rgbplanar.json \
--device_id $DEVICE \
--detect_threshold 0.01 \
--track_thresh 0.6 \
--track_buffer 25 \
--match_thresh 0.9 \
--min_box_area 100 \
--label_file ../../data/labels/coco2id.txt \
--dataset_filelist /opt/vastai/vaststreamx/data/datasets/mot17/MOT17-13-FRCNN-filelist.txt \
--dataset_root /opt/vastai/vaststreamx/data/datasets/mot17/ \
--dataset_result_file ./mot_output/MOT17-13-FRCNN.txt 

python3 ../../evaluation/mot/mot_eval.py \
-gt /opt/vastai/vaststreamx/data/datasets/mot17/test \
-r ./mot_output

