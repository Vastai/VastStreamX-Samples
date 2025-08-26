import sys
import os
import torch

# _dataset_path = "/opt/vastai/vastpipe/data/images/fastreid"
# os.environ["FASTREID_DATASETS"] = _dataset_path + "/datasets/"
# sys.path.append(_dataset_path + "/fast-reid/")
current_file_path = os.path.dirname(os.path.abspath(__file__))
repo_path = os.path.join(current_file_path, "./fast-reid")
sys.path.append(repo_path)

from fastreid.config import get_cfg
from collections import OrderedDict
from fastreid.engine import DefaultTrainer, default_argument_parser, default_setup

import numpy as np


def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(cfg, args)
    return cfg


if __name__ == "__main__":
    parse = default_argument_parser()
    parse.add_argument(
        "--result_folder", type=str, default="", help="fastreid output folder"
    )
    parse.add_argument(
        "--dataset", type=str, default="", help="dataset folder"
    )
    parse.add_argument(
        "--gt",
        type=str,
        default="/opt/vastai/vastpipe/data/images/fastreid/fast-reid/configs/Market1501/bagtricks_R50.yml",
        help="fastreid output folder",
    )
    args = parse.parse_args()
    args.config_file = args.gt
    os.environ["FASTREID_DATASETS"] = args.dataset

    cfg = setup(args)
    cfg.defrost()
    cfg.MODEL.BACKBONE.PRETRAIN = False
    cfg.MODEL.DEVICE = "cpu"

    results = OrderedDict()
    for _, dataset_name in enumerate(cfg.DATASETS.TESTS):
        print(f"dataset_name:{dataset_name}")
        data_loader, evaluator = DefaultTrainer.build_evaluator(cfg, dataset_name)
        for _, inputs in enumerate(data_loader):
            outputs = []
            for _, input_data in enumerate(inputs["img_paths"]):
                npz_file = os.path.join(
                    args.result_folder, os.path.basename(input_data)[:-4] + ".npz"
                )
                output = np.load(npz_file, allow_pickle=True)["output_0"].astype(
                    "float32"
                )
                outputs.append(output)
            outputs = np.array(outputs)
            outputs = torch.Tensor(outputs).squeeze(1)
            evaluator.process(inputs, outputs)

        results[dataset_name] = evaluator.evaluate()

    if len(results) == 1:
        results = list(results.values())[0]
    print(results)
