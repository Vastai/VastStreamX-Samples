from vdsp_ops import WarpPerspectiveOp
import sys 
import cv2
import os
import numpy as np
import vaststreamx as vsx

current_file_path = os.path.dirname(os.path.abspath(__file__))
common_path = os.path.join(current_file_path, "../..")
sys.path.append(common_path)

import common.utils as utils

def get_matrix_and_size():
    matrixs = []
    sizes = []

    matrixs.append([ 1.00000000e+00,1.38613964e-17,-4.16000000e+02,
                    2.56977481e-17,1.00000000e+00,-4.00000000e+01,
                    6.57701147e-19,7.51658453e-19,1.00000000e+00])
    sizes.append((137,138))

    matrixs.append([ 1.00000000e+00, -2.51310209e-18, -6.02000000e+02,
                    3.86689924e-17,  1.00000000e+00, -8.30000000e+01,
                    1.02097856e-18, -1.00014226e-18,  1.00000000e+00])
    sizes.append((133,641))

    matrixs.append([ 9.99104745e-01,  2.77529096e-02, -4.27672337e+02,
                    -3.22291853e-02,9.99104745e-01,-1.68204118e+02,
                    4.04413119e-18, 1.44377954e-17, 1.00000000e+00])
    sizes.append((31, 216))

    matrixs.append([ 9.99937533e-01, -5.93434738e-03, -4.48986788e+02,
                    1.05256582e-02, 9.99937533e-01, -1.70726177e+02,
                    -4.43853413e-19, 2.63414488e-21, 1.00000000e+00])
    sizes.append((95,337))

    matrixs.append([ 9.93923764e-01,  0.00000000e+00,  0.00000000e+00,
                    -1.59642415e-02, 9.89782972e-01, -1.84099633e+02,
                    1.07931689e-06, -3.33889816e-05, 1.00000000e+00])
    sizes.append((248,120))

    matrixs.append([1.00000000e+00,1.43749145e-17,-1.29000000e+02,
                    0.00000000e+00,1.00000000e+00,-3.09000000e+02,
                    7.63203798e-18,1.86393612e-33 ,1.00000000e+00])
    sizes.append((49,19))

    matrixs.append([1.00000000e+00, 1.20670707e-17, -2.70000000e+01,
                    1.96011789e-15, 1.00000000e+00, -3.11000000e+02,
                    2.86509168e-17, -8.95341149e-19, 1.00000000e+00])
    sizes.append((31,16))


    matrixs.append([ 1.00000000e+00 ,8.92409333e-18,-9.40000000e+01,
                    0.00000000e+00 ,1.00000000e+00,-3.13000000e+02,
                   1.18819855e-17,-1.48524819e-18,1.00000000e+00])
    sizes.append((31,13))


    matrixs.append([ 1.00000000e+00, -2.32550547e-17, -3.79000000e+02,
                    0.00000000e+00, 1.00000000e+00, -4.24000000e+02,
                    2.27738056e-18, 3.79352666e-34, 1.00000000e+00])
    sizes.append((91, 240))

    matrixs.append([ 1.00000000e+00, -1.52861870e-19, -1.00000000e+00,
                    3.15796771e-15, 1.00000000e+00, -4.61000000e+02,
                    3.38353684e-17, -2.38506698e-35, 1.00000000e+00])
    sizes.append((35, 36))

    matrixs.append([9.99314599e-01, 2.46744345e-02, -3.42222070e+02,
                   -2.77587389e-02, 9.99314599e-01, -4.54493831e+02,
                   -1.95140868e-17, 7.36522437e-19, 1.00000000e+00])
    sizes.append((36, 81))

    return matrixs, sizes


def test_ppocr():
    image = "../../data/images/ppocr.jpg"
    image = "./ocr.jpg"

    device_id = 0

    warp_perspective_op = WarpPerspectiveOp(elf_file="./warp_perspective_debug", device_id=device_id)

    image_cv = cv2.imread(image)
    assert image_cv is not None, f"failed to read image: {image}"

    M=[1.00000000e+00, -2.33979443e-17, -2.20000000e+01,0.00000000e+00, 1.00000000e+00, -3.20000000e+01, 8.28138387e-19,  1.38023064e-19,  1.00000000e+00]
    M = np.asarray(M)
    img_crop_width=286
    img_crop_height=45

    vsx_image = utils.cv_bgr888_to_vsximage(image_cv, vsx.ImageFormat.RGB_PLANAR, device_id)

    output_vacc = warp_perspective_op.Process(vsx_image, M, 286, 45)

    print(f"output format:{output_vacc.format}")
    cv_output = utils.vsximage_to_cv_bgr888(output_vacc)

    cv2.imwrite("warp_perspective_result.jpg",cv_output)

    print(f"warp_perspective_result.jpg saved")


def test_ocr():
    image = "./ocr.jpg"

    device_id = 0

    warp_perspective_op = WarpPerspectiveOp(elf_file="./warp_perspective_debug", device_id=device_id)

    image_cv = cv2.imread(image)
    assert image_cv is not None, f"failed to read image: {image}"

    matrixs, sizes = get_matrix_and_size()

    for i, (mat, size) in enumerate(zip(matrixs, sizes)):
        M = np.asarray(mat)
        img_crop_width=size[0]
        img_crop_height=size[1]
        
        vsx_image = utils.cv_bgr888_to_vsximage(image_cv, vsx.ImageFormat.RGB_PLANAR, device_id)
        
        output_vacc = warp_perspective_op.Process(vsx_image, M, img_crop_width, img_crop_height)

        cv_output = utils.vsximage_to_cv_bgr888(output_vacc)
        
        cv2.imwrite(f"warp_perspective_result_{i}.jpg",cv_output)
        
        print(f"warp_perspective_result_{i}.jpg saved")




if __name__ == "__main__":
    test_ppocr()
    test_ocr()