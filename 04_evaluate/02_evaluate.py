import sys
import h5py
import numpy as np
from segtra import evaluate_segtra

def evaluate_files(res_file, gt_file):

    print("Reading volumes...")

    with h5py.File(res_file, 'r') as f:
        track_graph_present = 'graphs/track_graph' in f

    if not track_graph_present:
        add_track_graph(res_file)

    with h5py.File(res_file, 'r') as f:
        res_tracks = np.array(f['volumes/labels/tracks'])
        res_track_graph = np.array(f['graphs/track_graph'])

    with h5py.File(gt_file, 'r') as f:
        gt_tracks = np.array(f['volumes/labels/tracks'])
        gt_track_graph = np.array(f['graphs/track_graph'])

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

    assert len(sys.argv) == 3, "Usage: evaluate.py <res_file> <gt_file>"

    res_file = sys.argv[1]
    gt_file = sys.argv[2]
    evaluate_files(res_file, gt_file)
