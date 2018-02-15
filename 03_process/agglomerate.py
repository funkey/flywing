import sys
import h5py
import json
import numpy as np
import os
import time
import waterz
from coordinate import Coordinate
from roi import Roi
from watershed import watershed

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
        return_merge_history=False,
        return_region_graph=False,
        **kwargs):

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
            return_merge_history=return_merge_history,
            return_region_graph=return_region_graph)

def get_unique_pairs(a, b):

    return np.unique([a.flatten(), b.flatten()], axis=1).transpose()

def agglomerate_lineages(
        affs,
        ignore_mask,
        thresholds,
        roi,
        resolution,
        output_basenames,
        **kwargs):

    # prepare affinities for slice and lineage merging

    affs[:,ignore_mask==1] = 0

    affs_xy = np.array(affs)
    affs_xy[0] = 0

    affs_z = np.array(affs)
    affs_z[1, 2] = 0

    print("Extracting initial fragments...")
    fragments = watershed(affs, 'maxima_distance')
    fragments[ignore_mask==1] = 0

    outfiles = [ h5py.File(n + '.hdf', 'w') for n in output_basenames ]

    i = 0
    # outer loop: merge in 2D only
    for cells in agglomerate_with_waterz(
            affs_xy,
            thresholds,
            fragments,
            **kwargs):

        print("Storing 2D segmentation...")
        f = outfiles[i]

        ds = f.create_dataset(
            'volumes/labels/cells',
            data=cells,
            compression="gzip",
            dtype=np.uint64)
        ds.attrs['offset'] = roi.get_offset()
        ds.attrs['resolution'] = resolution

        # inner loop: merge across z only
        print("Agglomerating in z...")
        merge_z_args = kwargs
        merge_z_args['merge_function'] = 'max_aff'
        for lineages_history in agglomerate_with_waterz(
            affs_z,
            [thresholds[i]],
            cells.copy(),
            return_merge_history=True,
            **merge_z_args):

            lineages = lineages_history[0]
            history = lineages_history[1]

            merge_list = np.array([ [h['a'], h['b'], h['c']] for h in history ])
            merge_scores = np.array([ h['score'] for h in history ])

            print("Storing lineages...")
            f.create_dataset(
                'volumes/labels/lineages',
                data=lineages,
                compression="gzip")

            print("Storing merge-history from cells to lineages...")
            f.create_dataset(
                'graphs/lineages_merge_history',
                data=merge_list,
                compression="gzip")
            f.create_dataset(
                'graphs/lineages_merge_scores',
                data=merge_scores,
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
    affs_filename = os.path.join(aff_data_dir, sample + '.hdf')
    gt_data_dir = os.path.join(os.getcwd(), '../01_data')
    gt_filename = os.path.join(gt_data_dir, sample + '.hdf')

    print "Agglomerating " + sample + " with " + setup + ", iteration " + str(iteration) + " at thresholds " + str(thresholds)

    print "Reading affinities..."
    with h5py.File(affs_filename, 'r') as affs_file:
        affs = np.array(affs_file['volumes/predicted_affs'])
        affs_offset_nm = Coordinate(affs_file['volumes/predicted_affs'].attrs['offset'])
        resolution = Coordinate(affs_file['volumes/predicted_affs'].attrs['resolution'])
        affs_roi = Roi(affs_offset_nm, resolution*affs.shape[1:])
        print "affs ROI: " + str(affs_roi)

    print "Reading ignore mask..."
    with h5py.File(gt_filename, 'r') as gt_file:
        ignore_mask = np.array(gt_file['volumes/labels/ignore'])

    start = time.time()
    agglomerate_lineages(
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
