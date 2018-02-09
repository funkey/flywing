import h5py
import numpy as np
from scipy.ndimage.morphology import binary_dilation

# number of frames a lineage should be in to be considered for GT (after merging
# of lineages)
solid_lineage_threshold = 30

# used to temporarily mark ignore area
ignore_label = None

samples = [

  'proper_disc/Ecd_20141010_P2.cleaned',
  'proper_disc/Ecd_20150310_P2.cleaned',
  'proper_disc/Ecd_20150418_P1.cleaned',
  'proper_disc/Ecd_20150608_P1.cleaned',
  'proper_disc/Ecd_20161213_F0.cleaned',
  'peripodial/Ecd_20141010_P2.cleaned',
  'peripodial/Ecd_20150310_P2.cleaned',
  'peripodial/Ecd_20150418_P1.cleaned',
]

def replace(array, old_values, new_values):

    values_map = np.arange(int(array.max() + 1), dtype=new_values.dtype)
    values_map[old_values] = new_values

    return values_map[array]

def merge_lineages(cells, divisions, ignore_mask):

    global ignore_label
    if ignore_label is None:
        ignore_label = int(cells.max() + 1)

    # find all labels to be merged
    merge_labels = cells.copy()
    merge_labels[divisions==0] = 0

    old_values = []
    new_values = []
    ignore = []

    for z in range(cells.shape[0]):

        print("Processing frame %d"%z)

        # get all labels that are marked as "to merge" in the current frame
        merge_candidates = set(np.unique(merge_labels[z]))
        merge_candidates.remove(0)

        if len(merge_candidates) == 0:
            continue

        # get their overlap with the previous frame
        assert z > 0, "There are divisions in the first frame."

        current = merge_labels[z] # only cells marked as dividing
        prev = cells[z-1] # all cells

        prev_labels = set(np.unique(prev))
        masters = set([ l for l in merge_candidates if l in prev_labels ])

        # assert len(masters) == len(merge_candidates)/2, (
            # "Number of masters %d does not match number of division cells "
            # "%d/2"%(len(masters), len(merge_candidates)))

        # keep only non-masters
        merge_candidates = merge_candidates - masters

        # get overlap of merge candidates with previous section
        overlay = np.array([current.flatten(), prev.flatten()])
        matches, counts = np.unique(overlay, axis=1, return_counts=True)
        matches = list(zip(matches[0], matches[1]))
        # co-sort matches by size
        counts, matches = (list(t) for t in zip(*sorted(zip(counts, matches))))

        # find matches of merge candidates to masters
        for c, m in zip(counts, matches):

            merge_candidate, master = m

            if master not in masters:
                continue
            if merge_candidate not in merge_candidates:
                continue

            # assign merge_candidate to master
            old_values.append(merge_candidate)
            new_values.append(master)

            merge_candidates.remove(merge_candidate)
            masters.remove(master)

        print("Unmatched merge candidates: %s"%merge_candidates)
        print("Unmatched masters: %s"%masters)

        for l in merge_candidates.union(masters):
            ignore.append(l)

    # merge
    lineages = replace(cells, np.array(old_values, dtype=np.uint64), np.array(new_values, dtype=np.uint64))

    # update ignore mask
    ignored = replace(lineages, np.array(ignore, dtype=np.uint64), np.array([ignore_label], dtype=np.uint64))
    ignored = (ignored==ignore_label).astype(np.uint8)
    ignore_mask |= ignored

    return lineages, ignore_mask

def ignore_isolated_lineages(lineages):

    global ignore_label
    if ignore_label is None:
        ignore_label = int(lineages.max() + 1)

    print("Finding isolated labels...")

    # find number of sections each lineage is in
    span = {}

    for z in range(lineages.shape[0]):
        for lineage_id in np.unique(lineages[z]):
            if lineage_id in span:
                span[lineage_id] += 1
            else:
                span[lineage_id] = 1

    print("Masking isolated labels...")
    ignore_lineages = np.array([ i for (i, s) in span.items() if s < solid_lineage_threshold ], dtype=np.uint64)
    ignore_mask = replace(lineages, ignore_lineages, np.array([ignore_label], dtype=np.uint64))
    ignore_mask = (ignore_mask==ignore_label).astype(np.uint8)
    print("done.")

    return ignore_mask

def close_ignore_mask(cells, ignore_mask):

    print("Closing phantom boundaries in ignore mask...")

    # close isolated boundaries in ignore mask
    for z in range(cells.shape[0]):

        # get a foreground mask
        fg = np.logical_and(cells[z]>0, ignore_mask[z]==0)

        # dilate it by one
        fg = binary_dilation(fg)

        # every pixel not on dilated foreground and on background is false boundary
        ignore_mask[z] |= np.logical_and(np.logical_not(fg), cells[z]==0)

def label_unique_2D(cells):
    '''Assign each cell a unique ID in each frame.'''

    next_id = 1
    for z in range(cells.shape[0]):

        frame = cells[z]
        old_values = np.unique(frame[frame>0])
        new_values = np.arange(next_id, next_id + len(old_values))
        cells[z] = replace(frame, old_values, new_values)
        next_id += len(old_values)

for sample in samples:

    print("Merging lineages in %s"%sample)

    with h5py.File(sample + '.hdf', 'r') as infile:

        raw = np.array(infile['volumes/raw'])
        divisions = np.array(infile['volumes/labels/divisions'])
        boundaries = np.array(infile['volumes/labels/boundaries'])
        cells = np.array(infile['volumes/labels/cells'])
        ignore_mask = np.array(infile['volumes/labels/ignore'])

        lineages, ignore_mask = merge_lineages(cells, divisions, ignore_mask)

        label_unique_2D(cells)

        ignore_mask |= ignore_isolated_lineages(lineages)
        accepted_lineages= lineages.copy()
        accepted_lineages[ignore_mask==1] = 0

        close_ignore_mask(accepted_lineages, ignore_mask)

        print("Writing merged dataset")
        with h5py.File(sample + '.merged.hdf', 'w') as outfile:

            outfile.create_dataset(
                'volumes/raw',
                data = raw,
                compression='gzip')
            outfile.create_dataset(
                'volumes/labels/divisions',
                data = divisions,
                compression='gzip')
            outfile.create_dataset(
                'volumes/labels/boundaries',
                data = boundaries,
                compression='gzip')
            outfile.create_dataset(
                'volumes/labels/cells',
                data = cells,
                compression='gzip')
            outfile.create_dataset(
                'volumes/labels/lineages',
                data = lineages,
                compression='gzip')
            outfile.create_dataset(
                'volumes/labels/ignore',
                data = ignore_mask,
                compression='gzip')
