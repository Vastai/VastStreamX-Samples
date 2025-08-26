import argparse
import os
import glob
import motmetrics as mm
from collections import OrderedDict
from pathlib import Path

mm.lap.default_solver = 'lap'

def argument_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-r", "--result_dir", type=str, help="datset result dir",
        default="outputs/mot",
    )
    parser.add_argument(
        "-gt",
        "--gt_dir",
        type=str,
        help="input dataset label file",
        default="data/datasets/track/mot17/test",
    )
    args = parser.parse_args()
    return args

def compare_dataframes(gts, ts):
    accs = []
    names = []
    for k, tsacc in ts.items():
        if k in gts:
            print('Comparing {}...'.format(k))
            accs.append(
                mm.utils.compare_to_groundtruth(gts[k],
                                                tsacc,
                                                'iou',
                                                distth=0.5))
            names.append(k)
        else:
            print('No ground truth for {}, skipping.'.format(k))
    return accs, names

def mot_eval(gt_dir, result_dir):
    gt_type = ''
    print('gt_type', gt_type)
    gtfiles = glob.glob(
        os.path.join(gt_dir, '*/gt/gt{}.txt'.format(gt_type)))
    gtfiles.sort()
    print('gt_files', gtfiles)
    tsfiles = glob.glob(os.path.join(result_dir, '*.txt'))
    tsfiles.sort()

    print('Found {} groundtruths and {} test files.'.format(
        len(gtfiles), len(tsfiles)))
    print('Available LAP solvers {}'.format(mm.lap.available_solvers))
    print('Default LAP solver \'{}\''.format(mm.lap.default_solver))
    print('Loading files.')

    gt = OrderedDict([(Path(f).parts[-3],
                        mm.io.loadtxt(f, fmt='mot15-2D', min_confidence=1))
                        for f in gtfiles])
    ts = OrderedDict([(os.path.splitext(Path(f).parts[-1])[0],
                        mm.io.loadtxt(f, fmt='mot15-2D', min_confidence=-1))
                        for f in tsfiles])

    mh = mm.metrics.create()
    accs, names = compare_dataframes(gt, ts)

    print('Running metrics')
    metrics = [
        'recall', 'precision', 'num_unique_objects', 'mostly_tracked',
        'partially_tracked', 'mostly_lost', 'num_false_positives',
        'num_misses', 'num_switches', 'num_fragmentations', 'mota', 'motp',
        'num_objects'
    ]
    summary = mh.compute_many(accs,
                            names=names,
                            metrics=metrics,
                            generate_overall=True)
    div_dict = {
        'num_objects': [
            'num_false_positives', 'num_misses', 'num_switches',
            'num_fragmentations'
        ],
        'num_unique_objects':
        ['mostly_tracked', 'partially_tracked', 'mostly_lost']
    }
    for divisor in div_dict:
        for divided in div_dict[divisor]:
            summary[divided] = (summary[divided] / summary[divisor])
    fmt = mh.formatters
    change_fmt_list = [
        'num_false_positives', 'num_misses', 'num_switches',
        'num_fragmentations', 'mostly_tracked', 'partially_tracked',
        'mostly_lost'
    ]
    for k in change_fmt_list:
        fmt[k] = fmt['mota']
    print(
        mm.io.render_summary(summary,
                                formatters=fmt,
                                namemap=mm.io.motchallenge_metric_names))

    metrics = mm.metrics.motchallenge_metrics + ['num_objects']
    summary = mh.compute_many(accs,
                                names=names,
                                metrics=metrics,
                                generate_overall=True)
    print(
        mm.io.render_summary(summary,
                                formatters=mh.formatters,
                                namemap=mm.io.motchallenge_metric_names))
    print('Completed')

if __name__ == "__main__":
    args = argument_parser()
    mot_eval(args.gt_dir, args.result_dir)
