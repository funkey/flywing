import numpy as np
import tempfile
import os
from PIL import Image
from subprocess import check_output, CalledProcessError
from track_graph import ids_to_track_graph

def write_track_file(tracks, filename):

    with open(filename, 'w') as f:
        for track in tracks:
            f.write(
                    '%d %d %d %d\n'%(
                        track.label,
                        track.start,
                        track.end,
                        track.parent.label if track.parent is not None else 0))

def evaluate_segtra(seg, gt):

    print("Converting SEG to track graph...")
    seg_tracks, seg = ids_to_track_graph(seg)
    print("Converting GT to track graph...")
    gt_tracks, gt = ids_to_track_graph(gt)

    # holy cow, they need 16-bit encodings!
    assert seg.max() <= np.uint16(-1)
    assert gt.max() <= np.uint16(-1)
    seg = seg.astype(np.uint16)
    gt = gt.astype(np.uint16)

    # create a temp dir
    # dataset_dir = tempfile.mkdtemp()
    dataset_dir = 'test'

    seg_dir = os.path.join(dataset_dir, '01_RES')
    gt_dir = os.path.join(dataset_dir, '01_GT', 'SEG')
    gt_track_dir = os.path.join(dataset_dir, '01_GT', 'TRA')

    os.makedirs(seg_dir)
    os.makedirs(gt_dir)
    os.makedirs(gt_track_dir)

    # store seg and gt as stack of tif files...
    assert seg.shape[0] == gt.shape[0]

    # FORMAT:
    #
    # GT segmentation:
    #   * background 0
    #   * objects with IDs >=1, 16bit...
    #   -> this is what we already have
    #
    # RES segmentation:
    #   * background 0
    #   * objects with unique IDs >=1 in 2D, change between frames
    #     (hope this is not necessary, we will run out of IDs due to 16-bit
    #     encoding...)

    print("Preparing files for evaluation binaries...")
    for z in range(seg.shape[0]):

        seg_outfile = os.path.join(seg_dir, 'mask%03d.tif'%z)
        gt_outfile = os.path.join(gt_dir, 'man_seg%03d.tif'%z)
        gt_track_outfile = os.path.join(gt_track_dir, 'man_track%03d.tif'%z)

        seg_im = Image.fromarray(seg[z])
        gt_im = Image.fromarray(gt[z])

        seg_im.save(seg_outfile)
        gt_im.save(gt_outfile)
        gt_im.save(gt_track_outfile)

    print("Computing SEG score...")
    try:

        seg_output = check_output([
            './segtra_measure/Linux/SEGMeasure',
            dataset_dir,
            '01'
        ])

    except CalledProcessError as exc:

        print("Calling SEGMeasure failed: ", exc.returncode, exc.output)
        seg_score = 0

    else:

        seg_score = float(seg_output.split()[2])

    print("SEG score: %f"%seg_score)

    write_track_file(seg_tracks, os.path.join(seg_dir, 'res_track.txt'))
    write_track_file(gt_tracks, os.path.join(gt_track_dir, 'man_track.txt'))

    print("Computing TRA score...")
    try:

        tra_output = check_output([
            './segtra_measure/Linux/TRAMeasure',
            dataset_dir,
            '01'
        ])

    except CalledProcessError as exc:

        print("Calling TRAMeasure failed: ", exc.returncode, exc.output)
        tra_score = 0

    else:

        tra_score = float(tra_output.split()[2])

    print("TRA score: %f"%tra_score)

    return {
            'seg_score': seg_score,
            'tra_score': tra_score,
            'gt': gt,
            'gt_tracks': gt_tracks,
            'seg': seg,
            'seg_tracks': seg_tracks
    }

if __name__ == "__main__":

    seg = np.zeros((10, 100, 200), dtype=np.uint64)
    gt = np.zeros((10, 100, 200), dtype=np.uint64)

    gt[:,50:55,100:105] = 1
    seg[:,50:55,100:105] = 2

    evaluate_segtra(seg, gt)
