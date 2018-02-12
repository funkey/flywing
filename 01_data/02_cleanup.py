import h5py
import numpy as np
import malis
from scipy.ndimage.morphology import binary_dilation

# regions above this size (in 2D) are considered background
proper_disc_bg_size_threshold = 1000
peripodial_bg_size_threshold = 10000

# how many times boundary cells should be removed
remove_boundary_cells=2

# remove labels that appear in disconnected components
allow_disconnected_components = True

# used to temporarily mark ignore area
ignore_label = None

samples = [

  'proper_disc/Ecd_20141010_P2',
  'proper_disc/Ecd_20150310_P2',
  'proper_disc/Ecd_20150418_P1',
  'proper_disc/Ecd_20150608_P1',
  'proper_disc/Ecd_20161213_F0',
  'peripodial/Ecd_20141010_P2',
  'peripodial/Ecd_20150310_P2',
  'peripodial/Ecd_20150418_P1',
]

def replace(array, old_values, new_values):

    values_map = np.arange(int(array.max() + 1), dtype=new_values.dtype)
    values_map[old_values] = new_values

    return values_map[array]

def ignore_disconnected_components(cells):

    global ignore_label
    if ignore_label is None:
        ignore_label = int(cells.max() + 1)

    print("Relabelling connected components...")
    simple_neighborhood = malis.mknhood3d()
    affinities = malis.seg_to_affgraph(
        cells,
        simple_neighborhood)
    relabelled, _ = malis.connected_components_affgraph(
        affinities,
        simple_neighborhood)

    print("Creating overlay...")
    overlay = np.array([cells.flatten(), relabelled.flatten()])
    print("Finding unique pairs...")
    matches = np.unique(overlay, axis=1)

    print("Finding disconnected labels...")
    orig_to_new = {}
    disconnected = set()
    for orig_id, new_id in zip(matches[0], matches[1]):
        if orig_id == 0 or new_id == 0:
            continue
        if orig_id not in orig_to_new:
            orig_to_new[orig_id] = [new_id]
        else:
            orig_to_new[orig_id].append(new_id)
            disconnected.add(orig_id)

    print("Masking %d disconnected labels..."%len(disconnected))
    ignore_mask = replace(cells, np.array([l for l in disconnected]), np.array([ignore_label], dtype=np.uint64))
    ignore_mask = (ignore_mask==ignore_label).astype(np.uint8)
    print("done.")

    return ignore_mask

def ignore_boundary_cells(cells, boundaries, iterations):
    '''Assumes that background is already labelled with 0 in cells.'''

    current_cells = cells.copy()

    global ignore_label
    if ignore_label is None:
        ignore_label = int(cells.max() + 1)

    ignore_mask = np.zeros(cells.shape, dtype=np.uint8)

    for i in range(iterations):

        print("Removing boundary cells...")

        ignore = []

        for z in range(current_cells.shape[0]):

            c = current_cells[z]
            b = boundaries[z]

            # get a background mask
            background = np.logical_and((c==0), (b==0))

            # grow background by 2 steps, such that it overlaps with cell labels at the
            # boundary
            background = binary_dilation(background, iterations=2)

            # get overlap of cells with background
            overlay = np.array([c.flatten(), background.flatten()])
            matches = np.unique(overlay, axis=1)

            # ignore all cells that overlap with background
            for label, overlaps_with_background in zip(matches[0], matches[1]):
                if label != 0 and overlaps_with_background:
                    ignore.append(label)

        # update ignore mask
        ignore_labelled = replace(cells, np.array(ignore, dtype=np.uint64), np.array([ignore_label], dtype=np.uint64))
        ignore_mask |= (ignore_labelled==ignore_label).astype(np.uint8)

        # mark ignored cells as background for next iteration
        current_cells[ignore_mask==ignore_label] = 0

    # ignore cells which are at the border of the image
    t,w,h = cells.shape
    frame = np.zeros((w,h), dtype=bool)
    frame[0:3,:] = frame[w-3:,:] = frame[:,0:3] = frame[:,h-3:] = True
    ignore = []
    for z in range(current_cells.shape[0]):
        c = current_cells[z]
        overlay = c[frame]
        ignore.append(np.unique(overlay[overlay != 0]))
    
    # update ignore mask
    ignore_labelled = replace(cells, np.concatenate(ignore).astype(np.uint64), np.array([ignore_label], dtype=np.uint64))
    ignore_mask |= (ignore_labelled==ignore_label).astype(np.uint8)

    return ignore_mask

def mark_background(cells, bg_size_threshold):

    print("Marking background...")

    background = []
    for z in range(cells.shape[0]):
        components, counts = np.unique(cells[z], return_counts=True)
        for component, count in zip(components, counts):
            if count > bg_size_threshold:
                background.append(component)

    return replace(cells, np.array(background, dtype=np.uint64), np.array([0], dtype=np.uint64))

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

for sample in samples:

    print("Cleaning GT in %s"%sample)

    if 'proper_disc' in sample:
        bg_size_threshold = proper_disc_bg_size_threshold
    else:
        bg_size_threshold = peripodial_bg_size_threshold

    with h5py.File(sample + '.hdf', 'r') as infile:

        raw = np.array(infile['volumes/raw'])
        divisions = np.array(infile['volumes/labels/divisions'])
        boundaries = np.array(infile['volumes/labels/boundaries'])
        cells = np.array(infile['volumes/labels/cells'])

        cells = mark_background(cells, bg_size_threshold)

        ignore_mask = np.zeros(cells.shape, dtype=np.uint8)

        ignore_mask |= ignore_boundary_cells(cells, boundaries, iterations=remove_boundary_cells)
        cells[ignore_mask==1] = 0

        if not allow_disconnected_components:
            ignore_mask |= ignore_disconnected_components(cells)
            cells[ignore_mask==1] = 0

        close_ignore_mask(cells, ignore_mask)

        print("Writing cleaned dataset")
        with h5py.File(sample + '.cleaned.hdf', 'w') as outfile:

            outfile.create_dataset(
                'volumes/raw',
                data=raw,
                compression='gzip')
            outfile.create_dataset(
                'volumes/labels/divisions',
                data=divisions,
                compression='gzip')
            outfile.create_dataset(
                'volumes/labels/boundaries',
                data=boundaries,
                compression='gzip')
            outfile.create_dataset(
                'volumes/labels/cells',
                data=cells,
                compression='gzip')
            outfile.create_dataset(
                'volumes/labels/ignore',
                data=ignore_mask,
                compression='gzip')
