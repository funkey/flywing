import h5py
import json
import numpy as np
import os
import time
import waterz
from coordinate import Coordinate
from roi import Roi
from watershed import watershed
from track_graph import find_edges, contract, relabel

scoring_functions = {

        'mean_aff': 'OneMinus<MeanAffinity<RegionGraphType, ScoreValue>>',
        'max_aff': 'OneMinus<MaxAffinity<RegionGraphType, ScoreValue>>',
        'max_10': 'OneMinus<MeanMaxKAffinity<RegionGraphType, 10, ScoreValue>>',

        # quantile merge functions, initialized with max affinity
        '15_aff_maxinit': 'OneMinus<QuantileAffinity<RegionGraphType, 15, ScoreValue>>',
        '15_aff_maxinit_histograms': 'OneMinus<HistogramQuantileAffinity<RegionGraphType, 15, ScoreValue, 256>>',
        '25_aff_maxinit': 'OneMinus<QuantileAffinity<RegionGraphType, 25, ScoreValue>>',
        '25_aff_maxinit_histograms': 'OneMinus<HistogramQuantileAffinity<RegionGraphType, 25, ScoreValue, 256>>',
        'median_aff_maxinit': 'OneMinus<QuantileAffinity<RegionGraphType, 50, ScoreValue>>',
        'median_aff_maxinit_histograms': 'OneMinus<HistogramQuantileAffinity<RegionGraphType, 50, ScoreValue, 256>>',
        '75_aff_maxinit': 'OneMinus<QuantileAffinity<RegionGraphType, 75, ScoreValue>>',
        '75_aff_maxinit_histograms': 'OneMinus<HistogramQuantileAffinity<RegionGraphType, 75, ScoreValue, 256>>',
        '85_aff_maxinit': 'OneMinus<QuantileAffinity<RegionGraphType, 85, ScoreValue>>',
        '85_aff_maxinit_histograms': 'OneMinus<HistogramQuantileAffinity<RegionGraphType, 85, ScoreValue, 256>>',

        # quantile merge functions, initialized with quantile
        '15_aff': 'OneMinus<QuantileAffinity<RegionGraphType, 15, ScoreValue, false>>',
        '15_aff_histograms': 'OneMinus<HistogramQuantileAffinity<RegionGraphType, 15, ScoreValue, 256, false>>',
        '25_aff': 'OneMinus<QuantileAffinity<RegionGraphType, 25, ScoreValue, false>>',
        '25_aff_histograms': 'OneMinus<HistogramQuantileAffinity<RegionGraphType, 25, ScoreValue, 256, false>>',
        'median_aff': 'OneMinus<QuantileAffinity<RegionGraphType, 50, ScoreValue, false>>',
        'median_aff_histograms': 'OneMinus<HistogramQuantileAffinity<RegionGraphType, 50, ScoreValue, 256, false>>',
        '75_aff': 'OneMinus<QuantileAffinity<RegionGraphType, 75, ScoreValue, false>>',
        '75_aff_histograms': 'OneMinus<HistogramQuantileAffinity<RegionGraphType, 75, ScoreValue, 256, false>>',
        '85_aff': 'OneMinus<QuantileAffinity<RegionGraphType, 85, ScoreValue, false>>',
        '85_aff_histograms': 'OneMinus<HistogramQuantileAffinity<RegionGraphType, 85, ScoreValue, 256, false>>',
}

def agglomerate_with_waterz(
        affs,
        thresholds,
        fragments,
        histogram_quantiles=False,
        discrete_queue=False,
        merge_function='median_aff',
        init_with_max=True,
        return_merge_history=False):

    if init_with_max:
        merge_function += '_maxinit'
    if histogram_quantiles:
        merge_function += '_histograms'

    discretize_queue = 0
    if discrete_queue:
        discretize_queue = 256

    return waterz.agglomerate(
            affs,
            thresholds,
            fragments=fragments,
            scoring_function=scoring_functions[merge_function],
            discretize_queue=discretize_queue,
            return_merge_history=return_merge_history)

def get_unique_pairs(a, b):

    return np.unique([a.flatten(), b.flatten()], axis=1).transpose()

def agglomerate_tracks(
        affs,
        thresholds,
        roi,
        resolution,
        output_basenames,
        **kwargs):

    # prepare affinities for slice and track merging

    affs_xy = np.array(affs)
    affs_xy[0] = 0

    affs_z = np.array(affs)
    affs_z[1, 2] = 0

    print("Extracting initial fragments...")
    fragments = watershed(affs, 'maxima_distance')

    outfiles = [ h5py.File(n + '.hdf', 'w') for n in output_basenames ]

    i = 0
    # outer loop: merge in 2D only
    for slices in agglomerate_with_waterz(
            affs_xy,
            thresholds,
            fragments,
            **kwargs):

        print("Storing 2D segmentation...")
        f = outfiles[i]

        ds = f.create_dataset(
            'volumes/labels/slices',
            data=slices,
            compression="gzip",
            dtype=np.uint64)
        ds.attrs['offset'] = roi.get_offset()
        ds.attrs['resolution'] = resolution

        # inner loop: merge across z only
        print("Agglomerating in z...")
        merge_z_args = kwargs
        merge_z_args['merge_function'] = 'max_aff'
        for tracks_history in agglomerate_with_waterz(
            affs_z,
            [thresholds[i]],
            slices.copy(),
            return_merge_history=True,
            **merge_z_args):

            tracks = tracks_history[0]
            history = tracks_history[1]

            merge_list = np.array([ [h['a'], h['b'], h['c']] for h in history ])
            merge_scores = np.array([ h['score'] for h in history ])

            print("Storing merge-history from slices to tracks...")
            f.create_dataset(
                'volumes/graphs/tracks_merge_history',
                data=merge_list,
                compression="gzip")
            f.create_dataset(
                'volumes/graphs/tracks_merge_scores',
                data=merge_scores,
                compression="gzip")

            # transform tracks into track graph with one node per 1D track

            print("Extracting track graph...")
            edges = find_edges(tracks, slices)
            track_graph = contract(edges, slices)
            track_graph_labels = relabel(slices, track_graph)

            track_graph_data = np.array([
                [
                    t.label,
                    t.start,
                    t.end,
                    t.parent.label if t.parent is not None else 0
                ]
                for t in track_graph
            ], dtype=np.uint64)

            print("Storing track graph...")
            f.create_dataset(
                'volumes/labels/tracks',
                data=track_graph_labels,
                compression="gzip")
            f.create_dataset(
                'volumes/labels/lineages',
                data=tracks,
                compression="gzip")
            f.create_dataset(
                'volumes/graphs/tracks_graph',
                data=track_graph_data,
                compression="gzip")

        i += 1

    for f in outfiles:
        f.close()


def agglomerate(
        setup,
        iteration,
        sample,
        thresholds,
        output_basenames,
        *args,
        **kwargs):

    thresholds = list(thresholds)

    aff_data_dir = os.path.join(os.getcwd(), 'processed', setup, str(iteration))
    affs_filename = os.path.join(aff_data_dir, sample + ".hdf")

    print "Agglomerating " + sample + " with " + setup + ", iteration " + str(iteration) + " at thresholds " + str(thresholds)

    print "Reading affinities..."
    affs_file = h5py.File(affs_filename, 'r')
    affs = affs_file['volumes/predicted_affs']
    affs_offset_nm = Coordinate(affs_file['volumes/predicted_affs'].attrs['offset'])
    resolution = Coordinate(affs_file['volumes/predicted_affs'].attrs['resolution'])
    affs_roi = Roi(affs_offset_nm, resolution*affs.shape[1:])
    print "affs ROI: " + str(affs_roi)

    start = time.time()
    agglomerate_tracks(
        affs,
        thresholds,
        affs_roi,
        resolution,
        output_basenames,
        **kwargs)
    print "Finished agglomeration in " + str(time.time() - start) + "s"

if __name__ == '__main__':

    thresholds = [0.2, 0.5]

    agglomerate(
            setup='setup01_both_pro01',
            iteration=100000,
            sample='pro05',
            thresholds=thresholds,
            output_basenames=['test_%3f'%t for t in thresholds],
            merge_function='mean_aff',
            init_with_max=False,
            histogram_quantiles=False,
            discrete_queue=True)

