import numpy as np
import argparse
from easydict import EasyDict
from scipy.integrate import simps

def compute_fr_and_auc(nmes, thres=0.10, step=0.0001):
    num_data = len(nmes)
    xs = np.arange(0, thres + step, step)
    ys = np.array([np.count_nonzero(nmes <= x) for x in xs]) / float(num_data)
    fr = 1.0 - ys[-1]
    auc = simps(ys, x=xs) / thres
    nme = np.mean(nmes)

    print("Face alignment evaluation result:")
    print("NME %: {}".format(np.mean(nmes)))
    print("FR_{}% : {}".format(thres,fr))
    print("AUC_{}: {}".format(thres,auc))
    return nme, fr, auc

def calc_nme(target_w_size, preds):
    """
        Args: 
                target_w_size: tensor in pytorch (n,98,2)
                pred_heatmap : tensor in pytorch (n,98,64,64) 
            
        Return:
                Sum_ION : the sum Ion of this batch data
    """
    config = EasyDict()
    config.norm_indices=[60, 72]
    config.heatmap_size=64
    ION = []

    # target_w_size and preds : n, 98 , 2
    # target_w_size *= config.heatmap_size
    target_w_size = np.expand_dims(target_w_size, axis=0)
    preds = np.expand_dims(preds, axis=0)
    target_np = target_w_size
    pred_np = preds

    for target, pred in zip(target_np,pred_np):
        diff = target - pred
        norm = np.linalg.norm(target[config.norm_indices[0]] - target[config.norm_indices[1]]) if config.norm_indices is not None else config.heatmap_size
        
        c_ION = np.sum(np.linalg.norm(diff,axis=1))/(diff.shape[0]*norm)
        ION.append(c_ION)

    Sum_ION = np.sum(ION) # the ion of this batch 
    # need div the dataset size to get nme
    return Sum_ION, ION
            
def get_label(label_path, ret_dict=False):
    with open(label_path, 'r') as f:
        labels = f.readlines()
    labels = [x.strip().split() for x in labels]
    if len(labels[0])==1:
        return labels
    if ret_dict:
        import os
        labels_new = {}
        for label in labels:
            image_name = label[0]
            target = label[1:197]
            target = np.array([float(x) for x in target])
            labels_new[os.path.basename(image_name)] = target.reshape(-1, 2)
        return labels_new
    labels_new = []
    for label in labels:
        image_name = label[0]
        target = label[1:]
        target = np.array([float(x) for x in target])
        labels_new.append([image_name, target])
    return labels_new


def argument_parser():
    parser = argparse.ArgumentParser(description="WFLW EVALUATION")
    parser.add_argument(
        "-r", "--result",
        default="",
        type=str,
        help="dataset output file",
    )
    parser.add_argument(
        "--gt",
        default="/opt/vastai/vastpipe/data/images/WFLW/list.txt",
        type=str,
        help="dataset ground truth",
    )
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = argument_parser()
    preds_label = get_label(args.result, ret_dict=True)
    gt_label = get_label(args.gt, ret_dict=True)
    
    IONs = None
    for filename in list(gt_label.keys()):
        gt_landmarks = gt_label[filename]
        preds_landmark = preds_label[filename]
        sum_ion, ion = calc_nme(gt_landmarks, preds_landmark)
        IONs = np.concatenate((IONs,ion),0) if IONs is not None else ion
    compute_fr_and_auc(IONs)  
