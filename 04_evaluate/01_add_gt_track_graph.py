from __future__ import print_function
import sys
import h5py
from track_graph import add_track_graph

gt_files = [
    '../01_data/per01.hdf',
    '../01_data/per02.hdf',
    '../01_data/per03.hdf',
    '../01_data/pro01.hdf',
    '../01_data/pro02.hdf',
    '../01_data/pro03.hdf',
    '../01_data/pro04.hdf',
    '../01_data/pro05.hdf',
]

for gt_file in gt_files:

    with h5py.File(gt_file, 'r') as f:
        track_graph_present = 'graphs/track_graph' in f

    if not track_graph_present:

        print("Adding GT track graph to ", gt_file)
        add_track_graph(gt_file)

    else:

        print("Skipping ", gt_file)
