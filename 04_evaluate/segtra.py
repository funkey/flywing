import numpy as np
import tempfile
import os
from PIL import Image
from subprocess import check_output, CalledProcessError

def write_track_file(tracks, filename):

    with open(filename, 'w') as f:
        for track in tracks:
            f.write(
                    '%d %d %d %d\n'%(
                        track.label,
                        track.start,
                        track.end,
                        track.parent.label if track.parent is not None else 0))

def evaluate_segtra(res_tracks, res_track_graph, gt_tracks, gt_track_graph):

    # holy cow, they need 16-bit encodings!
    assert res_tracks.max() <= np.uint16(-1)
    assert gt_tracks.max() <= np.uint16(-1)
    res_tracks = res_tracks.astype(np.uint16)
    gt_tracks = gt_tracks.astype(np.uint16)

    # create a temp dir
    # dataset_dir = tempfile.mkdtemp()
    dataset_dir = 'test'

    res_dir = os.path.join(dataset_dir, '01_RES')
    gt_dir = os.path.join(dataset_dir, '01_GT', 'SEG')
    gt_track_dir = os.path.join(dataset_dir, '01_GT', 'TRA')

    os.makedirs(res_dir)
    os.makedirs(gt_dir)
    os.makedirs(gt_track_dir)

    # store seg and gt as stack of tif files...
    assert res_tracks.shape[0] == gt_tracks.shape[0]

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

        res_outfile = os.path.join(res_dir, 'mask%03d.tif'%z)
        gt_outfile = os.path.join(gt_dir, 'man_seg%03d.tif'%z)
        gt_track_outfile = os.path.join(gt_track_dir, 'man_track%03d.tif'%z)

        res_im = Image.fromarray(res_tracks[z])
        gt_im = Image.fromarray(gt_tracks[z])

        res_im.save(res_outfile)
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

    write_track_file(res_track_graph, os.path.join(res_dir, 'res_track.txt'))
    write_track_file(gt_track_graph, os.path.join(gt_track_dir, 'man_track.txt'))

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
    }

if __name__ == "__main__":

    seg = np.zeros((10, 100, 200), dtype=np.uint64)
    gt = np.zeros((10, 100, 200), dtype=np.uint64)

    gt[:,50:55,100:105] = 1
    seg[:,50:55,100:105] = 2

    evaluate_segtra(seg, gt)
