from vdsp_ops import RotateOp, RotateDegree
import sys 
import cv2
import os

import vaststreamx as vsx


current_file_path = os.path.dirname(os.path.abspath(__file__))
common_path = os.path.join(current_file_path, "../..")
sys.path.append(common_path)

import common.utils as utils

image = "../../data/images/ppocr.jpg"

device_id = 0

rotate_op = RotateOp(elf_file="./simple_rotate_debug",device_id=device_id)

image_cv = cv2.imread(image)
assert image_cv is not None, f"failed to read image: {image}"
degree_type = int(sys.argv[1])


print(f"rotate degree_type: {degree_type}")

rotate_angle = RotateDegree.ROTATE_DEGREE_90
if degree_type == 0:
    rotate_angle = RotateDegree.ROTATE_DEGREE_90
elif degree_type == 1:
    rotate_angle = RotateDegree.ROTATE_DEGREE_180
elif degree_type == 2:
    rotate_angle = RotateDegree.ROTATE_DEGREE_270
elif degree_type == 4:
    rotate_angle = RotateDegree.ROTATE_DEGREE_NEG90
elif degree_type == 3:
    rotate_angle = RotateDegree.ROTATE_DEGREE_NEG270
else:
    raise Exception(f"unsupported rotate degree_type: {degree_type}")


vsx_image = utils.cv_bgr888_to_vsximage(image_cv, vsx.ImageFormat.BGR_INTERLEAVE, device_id)
print(f"vsx_image shape:{vsx_image.shape}")
org_image = utils.vsximage_to_cv_bgr888(vsx_image)
print(f"org_image shape:{org_image.shape}")

cv2.imwrite("ori.jpg", org_image)

print(f"rotate_angle:{rotate_angle}")
output_vacc = rotate_op(vsx_image, rotate_angle)

print(f"output format:{output_vacc.format}")
cv_output = utils.vsximage_to_cv_bgr888(output_vacc)

cv2.imwrite("rotate_result.jpg",cv_output)

print(f"rotate_result.jpg saved")