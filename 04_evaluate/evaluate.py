import sys
import h5py
import numpy as np
import json
from segtra import evaluate_segtra

def evaluate_files(res_file, gt_file):

    print("Reading volumes...")

    with h5py.File(res_file, 'r') as f:
        res_tracks = np.array(f['volumes/labels/tracks'])
        res_track_graph = np.array(f['graphs/track_graph'])

    with h5py.File(gt_file, 'r') as f:
        gt_tracks = np.array(f['volumes/labels/tracks'])
        gt_track_graph = np.array(f['graphs/track_graph'])

    report = evaluate_segtra(res_tracks, res_track_graph, gt_tracks, gt_track_graph)

    report_file = res_file[:-3] + 'json'
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2)

    print("Saved report %s in %s"%(report, report_file))

if __name__ == "__main__":

    assert len(sys.argv) == 3, "Usage: evaluate.py <res_file> <gt_file>"

    res_file = sys.argv[1]
    gt_file = sys.argv[2]
    evaluate_files(res_file, gt_file)
