import pyted
import numpy as np

def evaluate_ted(seg, gt):

    assert seg.max() <= np.uint32(-1)
    assert gt.max() <= np.uint32(-1)

    seg = seg.astype(np.uint32)
    gt = gt.astype(np.uint32)

    print("Evaluating segmentation...")

    params = pyted.Parameters()
    params.distance_threshold = 5
    params.report_ted = True
    params.report_voi = True
    params.report_rand = False
    params.have_background = True
    params.ignore_background = True
    params.verbosity = 3

    ted = pyted.Ted(params)
    # set voxel size to be less tolerant to errors over time
    report = ted.create_report(gt, seg, np.array([2, 1, 1]))

    print(report)
    return report
