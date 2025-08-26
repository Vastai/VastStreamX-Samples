import os
import cv2
import glob
import math
import argparse
import numpy as np

def calculate_psnr(img1, img2):
    img1_ = img1.copy().astype(np.float64)
    img2_ = img2.copy().astype(np.float64)
    mse = np.mean((img1_-img2_)**2)
    if mse == 0:
        return float('inf')
    return 20 * math.log10(255.0/math.sqrt(mse))


def ssim(img1, img2):
    C1 = (0.01 * 255)**2
    C2 = (0.03 * 255)**2

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1**2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                                            (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()


def calculate_ssim(img1, img2):
    '''calculate SSIM
    the same outputs as MATLAB's
    img1, img2: [0, 255]
    '''
    if not img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions.')
    if img1.ndim == 2:
        return ssim(img1, img2)
    elif img1.ndim == 3:
        if img1.shape[2] == 3:
            ssims = []
            for i in range(3):
                ssims.append(ssim(img1, img2))
            return np.array(ssims).mean()
        elif img1.shape[2] == 1:
            return ssim(np.squeeze(img1), np.squeeze(img2))
    else:
        raise ValueError('Wrong input image dimensions.')

if __name__ == '__main__':
    parse = argparse.ArgumentParser(description="SUPER RESOLUTION EVAL")
    parse.add_argument("--gt", type=str, default="/opt/vastai/vastpipe/data/images/DIV2K/DIV2K_valid_HR", help="gt HR images folder")
    parse.add_argument("--result", type=str, default="/home/lwyu/work/edsr/vacc_result", help="result images folder")
    args = parse.parse_args()

    image_files = sorted(glob.glob(os.path.join(args.result, "*.png")))

    psnr_list = []
    ssim_list = []
    for image_path in image_files:
        # eval
        output = cv2.imread(image_path)
        image_gt = cv2.imread(os.path.join(args.gt, os.path.basename(image_path).replace("x2", "")))
        image_gt = cv2.resize(image_gt, output.shape[:2][::-1]) # , interpolation=cv2.INTER_AREA

        vacc_psnr = calculate_psnr(image_gt, output)
        vacc_ssim = calculate_ssim(image_gt, output)
        # print("{} psnr: {}, ssim: {}".format(image_path, vacc_psnr, vacc_ssim))
        psnr_list.append(vacc_psnr)
        ssim_list.append(vacc_ssim)

    print("mean psnr: {}, mean ssim: {}".format(np.mean(psnr_list), np.mean(ssim_list)))
