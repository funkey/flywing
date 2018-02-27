import sys
import h5py
import json
import numpy as np
import os
import time
import tempfile
import shutil
from coordinate import Coordinate
from roi import Roi
from watershed import watershed
from agglomerate import agglomerate_with_waterz
from subprocess import check_call, CalledProcessError
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
        fragments,
        region_graph):

    print("Writing MLT problem...")

    with open(nodes_file, 'w') as nf:
        with open(edges_file, 'w') as ef:

            # create a map from fragments to 0-based indices per frame
            fragment_frame_ids = {}
            for z in range(fragments.shape[0]):

                frame = fragments[z]
                frame_fragments = list(np.unique(frame[frame>0]))
                centers = find_centers(frame, frame_fragments)

                fragment_frame_ids.update({
                    f: (z, i)
                    for i, f in enumerate(frame_fragments)
                })

                # write nodes file
                for i, fragment in enumerate(frame_fragments):

                    cx, cy = centers[fragment]

                    # t id x y p_birth_termination
                    nf.write('%d %d %d %d 0.0\n'%(z, i, cx, cy))

            # t0 id0 t1 id1 weight
            for edge in region_graph:

                u = edge['u']
                v = edge['v']
                score = edge['score']

                cut_prob = score_to_cut_prob(score)

                z_u, i_u = fragment_frame_ids[u]
                z_v, i_v = fragment_frame_ids[v]

                assert abs(z_u - z_v) <= 1, (
                    "Edge (%d, %d) connects nodes in frames %d and %d"%(
                        u, v, z_u, z_v))

                ef.write('%d %d %d %d %f\n'%(z_u, i_u, z_v, i_v, cut_prob))

    return fragment_frame_ids

def replace(array, old_values, new_values):

    values_map = np.arange(int(array.max() + 1), dtype=new_values.dtype)
    values_map[old_values] = new_values

    return values_map[array]

def read_mlt_solution(fragments, fragment_frame_ids, solution_file):

    frame_ids_fragment = {}
    for (f, (z, i)) in fragment_frame_ids.items():
        frame_ids_fragment[(z, i)] = f

    old_values = []
    new_values = []

    # read lineages
    for line in open(solution_file + '-fragment-node-labels.txt', 'r'):

        z, i, label = (int(x) for x in line.split())
        fragment = frame_ids_fragment[(z, i)]

        old_values.append(fragment)
        new_values.append(label)

    lineages = replace(
        fragments,
        np.array(old_values),
        np.array(new_values).astype(np.uint64))

    old_values = []
    new_values = []

    # read tracks
    track_id = 1
    track_start_end = [ None ] # track id 0 not used
    for line in open(solution_file + '-cell-nodes.txt', 'r'):

        # get all (t, id) pairs part of the same track
        cell_nodes = [
            tuple(int(x) for x in z_id.split())
            for z_id in line.strip().strip('()').split(') (')
        ]

        # get all fragments part of the same track
        track_fragments = [ frame_ids_fragment[x] for x in cell_nodes ]

        # replace all fragments with their track id
        old_values += track_fragments
        new_values += [track_id]*len(track_fragments)

        # get start and end of track
        frames = [ z for z, i in cell_nodes ]
        track_start_end.append((min(frames), max(frames)))

        track_id += 1

    num_tracks = track_id - 1

    tracks = replace(
        fragments,
        np.array(old_values),
        np.array(new_values).astype(np.uint64))

    # read track graph edges
    parents = {}
    for line in open(solution_file + '-cell-edges.txt', 'r'):

        # add +1, our ids start at 1, mlt output at 0 (zero is reserved in track
        # graph later for "no parent")
        parent, child = (int(x) + 1 for x in line.split())

        assert child not in parents
        parents[child] = parent

    # create track graph ready to write out
    track_graph = np.array([
        [
            track_id,
            track_start_end[track_id][0],
            track_start_end[track_id][1],
            parents[track_id] if track_id in parents else 0
        ]
        for track_id in range(1, num_tracks + 1)
    ], dtype=np.uint64)

    return tracks, track_graph, lineages

def solve_mlt(fragments, region_graph, threshold, merge_function):

    binary = {
        'mlt-ilp': './track-ilp',
        'mlt-gla': './track-heuristic-GLA',
        'mlt-klb': './track-heuristic-KLB',
    }[merge_function]

    # create a temp dir
    temp_dir = tempfile.mkdtemp()

    nodes_file = os.path.join(temp_dir, 'nodes')
    edges_file = os.path.join(temp_dir, 'edges')
    solution_file = os.path.join(temp_dir, 'solution')

    try:

        fragment_frame_ids = write_mlt_problem_graph(
            nodes_file,
            edges_file,
            fragments,
            region_graph)

        print("Solving MLT problem...")
        try:

            cmd = [
                binary,
                '-n', nodes_file,
                '-e', edges_file,
                '-s', solution_file,
                '-t', str(score_to_cut_prob(threshold)),
                '-b', str(score_to_cut_prob(threshold))
            ]
            check_call(cmd)

        except CalledProcessError as e:

            print("Failed to call\n\t" + " ".join(cmd))
            raise e

        tracks, track_graph, lineages = read_mlt_solution(
            fragments,
            fragment_frame_ids,
            solution_file)

    finally:

        shutil.rmtree(temp_dir)
        pass

    return tracks, track_graph, lineages

def track_lineages(
        affs,
        ignore_mask,
        thresholds,
        roi,
        resolution,
        output_basenames,
        merge_function,
        **kwargs):

    # prepare affinities for slice and lineage merging

    affs[:,ignore_mask==1] = 0

    print("Extracting initial fragments...")
    fragments = watershed(affs, 'maxima_distance')
    fragments[ignore_mask==1] = 0

    # get the region graph
    for fragments, region_graph in agglomerate_with_waterz(
            affs,
            [0],
            fragments,
            return_region_graph=True,
            **kwargs):
        pass

    for i in range(len(thresholds)):

        tracks, track_graph, lineages = solve_mlt(
            fragments,
            region_graph,
            thresholds[i],
            merge_function)

        print("Storing solution...")
        with h5py.File(output_basenames[i] + '.hdf', 'w') as f:

            print("Storing fragments...")
            ds = f.create_dataset(
                'volumes/labels/fragments',
                data=fragments,
                compression="gzip",
                dtype=np.uint64)
            ds.attrs['offset'] = roi.get_offset()
            ds.attrs['resolution'] = resolution

            print("Storing tracks...")
            ds = f.create_dataset(
                'volumes/labels/tracks',
                data=tracks,
                compression="gzip")
            ds.attrs['offset'] = roi.get_offset()
            ds.attrs['resolution'] = resolution

            print("Storing track graph...")
            f.create_dataset(
                'graphs/track_graph',
                data=track_graph,
                compression="gzip")

            print("Storing lineages...")
            ds = f.create_dataset(
                'volumes/labels/lineages',
                data=lineages,
                compression="gzip",
                dtype=np.uint64)
            ds.attrs['offset'] = roi.get_offset()
            ds.attrs['resolution'] = resolution

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
