import sys
import h5py
import json
import numpy as np
import os
import time
import tempfile
from coordinate import Coordinate
from roi import Roi
from watershed import watershed
from agglomerate import agglomerate_with_waterz
from subprocess import check_call
from scipy.ndimage.measurements import center_of_mass

def find_centers(ids, all_ids):

    coms = center_of_mass(np.ones_like(ids), ids, all_ids)
    return { i: l for i, l in zip(all_ids, coms) }

def score_to_cut_prob(score):

    # scores from agglomeration are in [0,1], with 0 favouring merging, 1
    # favouring cutting -- nothing to do here
    return score

def write_mlt_problem_graph(
        nodes_file,
        edges_file,
        cells,
        region_graph):

    print("Writing MLT problem...")

    with open(nodes_file, 'w') as nf:
        with open(edges_file, 'w') as ef:

            # create a map from cells to 0-based indices per frame
            cell_frame_ids = {}
            for z in range(cells.shape[0]):

                frame = cells[z]
                frame_cells = np.unique(frame[frame>0])
                centers = find_centers(frame, frame_cells)

                cell_frame_ids.update({
                    f: (z, i)
                    for i, f in enumerate(frame_cells)
                })

                # write nodes file
                for i, cell in enumerate(frame_cells):

                    cx, cy = centers[cell]

                    # t id x y p_birth_termination
                    nf.write('%d %d %d %d 0.0\n'%(z, i, cx, cy))

            # t0 id0 t1 id1 weight
            for edge in region_graph:

                u = edge['u']
                v = edge['v']
                score = edge['score']

                cut_prob = score_to_cut_prob(score)

                z_u, i_u = cell_frame_ids[u]
                z_v, i_v = cell_frame_ids[v]

                assert abs(z_u - z_v) <= 1, (
                    "Edge (%d, %d) connects nodes in frames %d and %d"%(
                        u, v, z_u, z_v))

                ef.write('%d %d %d %d %f\n'%(z_u, i_u, z_v, i_v, cut_prob))

def solve_mlt(cells, region_graph, threshold, solver):

    binary = {
        'ilp': './track-ilp',
        'gla': './track-heuristic-GLA',
        'klb': './track-heuristic-KLB',
    }[solver]

    # create a temp dir
    # temp_dir = tempfile.mkdtemp()
    temp_dir = 'test'

    nodes_file = os.path.join(temp_dir, 'nodes')
    edges_file = os.path.join(temp_dir, 'edges')
    solution_file = os.path.join(temp_dir, 'solution')

    try:

        write_mlt_problem_graph(
                nodes_file,
                edges_file,
                cells,
                region_graph)

        print("Solving MLT problem...")
        check_call([
            binary,
            '-n', nodes_file,
            '-e', edges_file,
            '-s', solution_file,
            '-t', str(score_to_cut_prob(threshold)),
            '-b', str(score_to_cut_prob(threshold))
        ])

    finally:

        # shutil.rmtree(temp_dir)
        pass

def track_lineages(
        affs,
        ignore_mask,
        thresholds,
        roi,
        resolution,
        output_basenames,
        solver,
        **kwargs):

    # prepare affinities for slice and lineage merging

    affs[:,ignore_mask==1] = 0

    print("Extracting initial fragments...")
    fragments = watershed(affs, 'maxima_distance')
    fragments[ignore_mask==1] = 0

    outfiles = [ h5py.File(n + '.hdf', 'w') for n in output_basenames ]

    # get the region graph
    for cells, region_graph in agglomerate_with_waterz(
            affs,
            [0],
            fragments,
            return_region_graph=True,
            **kwargs):
        pass

    for i in range(len(thresholds)):

        lineages = solve_mlt(cells, region_graph, thresholds[i], solver)

        print("Storing 2D segmentation...")
        f = outfiles[i]

        ds = f.create_dataset(
            'volumes/labels/cells',
            data=cells,
            compression="gzip",
            dtype=np.uint64)
        ds.attrs['offset'] = roi.get_offset()
        ds.attrs['resolution'] = resolution

        print("Storing lineages...")
        ds = f.create_dataset(
            'volumes/labels/lineages',
            data=lineages,
            compression="gzip")
        ds.attrs['offset'] = roi.get_offset()
        ds.attrs['resolution'] = resolution

    for f in outfiles:
        f.close()


def agglomerate(
        setup,
        iteration,
        sample,
        thresholds,
        output_basenames,
        first_frame=None,
        last_frame=None,
        *args,
        **kwargs):

    thresholds = list(thresholds)

    aff_data_dir = os.path.join(os.getcwd(), 'processed', setup, str(iteration))
    affs_filename = os.path.join(aff_data_dir, sample + '.hdf')
    gt_data_dir = os.path.join(os.getcwd(), '../01_data')
    gt_filename = os.path.join(gt_data_dir, sample + '.hdf')

    print "Running MLT on " + sample + " with " + setup + ", iteration " + str(iteration) + " at thresholds " + str(thresholds)

    print "Reading affinities..."
    with h5py.File(affs_filename, 'r') as affs_file:
        affs = np.array(affs_file['volumes/predicted_affs'][:,first_frame:last_frame])
        affs_offset_nm = Coordinate(affs_file['volumes/predicted_affs'].attrs['offset'])
        resolution = Coordinate(affs_file['volumes/predicted_affs'].attrs['resolution'])
        affs_roi = Roi(affs_offset_nm, resolution*affs.shape[1:])
        print "affs ROI: " + str(affs_roi)

    print "Reading ignore mask..."
    with h5py.File(gt_filename, 'r') as gt_file:
        ignore_mask = np.array(gt_file['volumes/labels/ignore'][first_frame:last_frame])

    start = time.time()
    track_lineages(
        affs,
        ignore_mask,
        thresholds,
        affs_roi,
        resolution,
        output_basenames,
        **kwargs)
    print "Finished agglomeration in " + str(time.time() - start) + "s"

if __name__ == "__main__":

    args_file = sys.argv[1]
    with open(args_file, 'r') as f:
        args = json.load(f)
    agglomerate(**args)
