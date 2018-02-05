import h5py
import glob
from PIL import Image
import numpy as np
import os

def to_array(files, is_label):

    print("Converting images to numpy array...")

    images = np.array([ np.array(Image.open(f)) for f in files ])
    print(images.shape)

    if is_label:
        images = images.astype(np.uint64)

    if len(images.shape) == 4:

        if is_label:
            print("Converting from RGB(?) to label array...")
            images = images[:,:,:,0] + 256*images[:,:,:,1] + 256*256*images[:,:,:,2]
        else:
            print("The source images contain %d channels, keeping only first one..."%images.shape[3])
            images = images[:,:,:,0]

    return images

for dataset in ['proper_disc', 'peripodial']:

    samples = [ s for s in glob.glob(dataset + '/*') if os.path.isdir(s) ]
    for sample in samples:

        if os.path.isfile(sample + '.hdf'):
            print("Skipping sample %s, already processed"%sample)
            continue

        print("Processing %s, %s"%(dataset, sample))

        frames = [ f for f in glob.glob(sample + '/Segmentation/*') if os.path.isdir(f) ]
        frames.sort()
        print("Found frames %s..."%(frames[:3]))

        print("Parsing original")
        raw = to_array([ f + '/original.png' for f in frames ], False).astype(np.uint8)
        print("Parsing handCorrection")
        boundaries = to_array([ f + '/handCorrection.tif' for f in frames], True).astype(np.uint8)
        print("Parsing tracked_cells_resized")
        cells = to_array([ f + '/tracked_cells_resized.tif' for f in frames ], True)
        print("Parsing dividing_cells")
        dividing_cells = to_array([ f + '/dividing_cells.tif' for f in frames ], True)

        boundaries[boundaries>0] = 1
        cells[boundaries==1] = 0
        dividing_cells[boundaries==1] = 0
        dividing_cells = (dividing_cells>0).astype(np.uint8)

        with h5py.File(sample + '.hdf', 'w') as f:

            f.create_dataset(
                'volumes/raw',
                data = raw,
                compression='gzip')
            f.create_dataset(
                'volumes/labels/divisions',
                data = dividing_cells,
                compression='gzip')
            f.create_dataset(
                'volumes/labels/boundaries',
                data = boundaries,
                compression='gzip')
            f.create_dataset(
                'volumes/labels/cells',
                data = cells,
                compression='gzip')
