import sys
import h5py
import numpy as np
from segtra import evaluate_segtra

def evaluate_files(seg_file, gt_file):

    print("Reading volumes...")

    with h5py.File(seg_file, 'r') as f:
        seg = np.array(f['volumes/labels/cells'])

    with h5py.File(gt_file, 'r') as f:
        gt = np.array(f['volumes/labels/cells'])
        ignore = np.array(f['volumes/labels/ignore'])

    gt[ignore==1] = 0
    report = evaluate_segtra(seg, gt)

    # DEBUG
    with h5py.File('gt.hdf', 'w') as f:
        f.create_dataset(
            'ids',
            data = report['gt'],
            compression='gzip')
    with h5py.File('seg.hdf', 'w') as f:
        f.create_dataset(
            'ids',
            data = report['seg'],
            compression='gzip')

if __name__ == "__main__":

    assert len(sys.argv) == 3, "Usage: evaluate.py <seg_file> <gt_file>"

    seg_file = sys.argv[1]
    gt_file = sys.argv[2]
    evaluate_files(seg_file, gt_file)
