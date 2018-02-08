from __future__ import print_function
import sys
from gunpowder import *
from gunpowder.tensorflow import *
import malis
import os
import math
import json
import tensorflow as tf

data_dir = '../../01_data'
samples = [
    # 'pro01',
    # 'pro02',
    # 'pro03',
    # 'pro04',
    # 'pro05',
    # 'per01', # test
    'per02',
    # 'per03', # validation
]

class IgnoreToMask(BatchFilter):
    '''Turn a mask marking ignore areas with 1 into one a mask marking
    foreground with 1.'''

    def setup(self):
        self.updates(VolumeTypes.GT_MASK, self.spec[VolumeTypes.GT_MASK])

    def process(self, batch, request):
        batch.volumes[VolumeTypes.GT_MASK].data = 1 - batch.volumes[VolumeTypes.GT_MASK].data

def train_until(max_iteration):

    if tf.train.latest_checkpoint('.'):
        trained_until = int(tf.train.latest_checkpoint('.').split('_')[-1])
    else:
        trained_until = 0
    if trained_until >= max_iteration:
        return

    with open('net_io_names.json', 'r') as f:
        net_io_names = json.load(f)

    register_volume_type('RAW')
    register_volume_type('GT_LABELS')
    register_volume_type('GT_MASK')
    register_volume_type('GT_AFFINITIES')
    register_volume_type('GT_AFFINITIES_MASK')
    register_volume_type('GT_AFFINITIES_SCALE')
    register_volume_type('PREDICTED_AFFS')
    register_volume_type('LOSS_GRADIENT')

    input_size = Coordinate((29, 188, 188))
    output_size = Coordinate((1, 100, 100))

    request = BatchRequest()
    request.add(VolumeTypes.RAW, input_size)
    request.add(VolumeTypes.GT_LABELS, output_size)
    request.add(VolumeTypes.GT_MASK, output_size)
    request.add(VolumeTypes.GT_AFFINITIES, output_size)
    request.add(VolumeTypes.GT_AFFINITIES_MASK, output_size)
    request.add(VolumeTypes.GT_AFFINITIES_SCALE, output_size)

    snapshot_request = BatchRequest({
        VolumeTypes.PREDICTED_AFFS: request[VolumeTypes.GT_AFFINITIES],
        VolumeTypes.LOSS_GRADIENT: request[VolumeTypes.GT_AFFINITIES]
    })

    data_sources = tuple(
        Hdf5Source(
            os.path.join(data_dir, sample + '.hdf'),
            datasets = {
                VolumeTypes.RAW: 'volumes/raw',
                VolumeTypes.GT_LABELS: 'volumes/labels/lineages',
                VolumeTypes.GT_MASK: 'volumes/labels/ignore',
            },
            volume_specs = {
                VolumeTypes.GT_MASK: VolumeSpec(interpolatable=False)
            }
        ) +
        IgnoreToMask() +
        Normalize() +
        RandomLocation() +
        Reject()
        for sample in samples
    )

    train_pipeline = (
        data_sources +
        RandomProvider() +
        SplitAndRenumberSegmentationLabels() +
        GrowBoundary(
            steps=1,
            only_xy=True) +
        ElasticAugment(
            [1,10,10],
            [0,1,1],
            [0,math.pi/2.0],
            prob_slip=0.05,
            prob_shift=0.05,
            max_misalign=3,
            subsample=8) +
        SimpleAugment(transpose_only_xy=True) +
        AddGtAffinities(
            malis.mknhood3d(),
            gt_labels_mask=VolumeTypes.GT_MASK) +
        BalanceLabels(
            labels=VolumeTypes.GT_AFFINITIES,
            scales=VolumeTypes.GT_AFFINITIES_SCALE,
            mask=VolumeTypes.GT_AFFINITIES_MASK) +
        IntensityAugment(0.9, 1.1, -0.1, 0.1, z_section_wise=True) +
        IntensityScaleShift(2,-1) +
        PreCache(
            cache_size=40,
            num_workers=10) +
        Train(
            'unet',
            optimizer=net_io_names['optimizer'],
            loss=net_io_names['loss'],
            inputs={
                net_io_names['raw']: VolumeTypes.RAW,
                net_io_names['gt_affs']: VolumeTypes.GT_AFFINITIES,
                net_io_names['loss_weights']: VolumeTypes.GT_AFFINITIES_SCALE,
            },
            outputs={
                net_io_names['affs']: VolumeTypes.PREDICTED_AFFS
            },
            gradients={
                net_io_names['affs']: VolumeTypes.LOSS_GRADIENT
            }) +
        IntensityScaleShift(0.5, 0.5) +
        Snapshot({
                VolumeTypes.RAW: 'volumes/raw',
                VolumeTypes.GT_LABELS: 'volumes/labels/lineages',
                VolumeTypes.GT_MASK: 'volumes/labels/mask',
                VolumeTypes.GT_AFFINITIES: 'volumes/labels/affinities',
                VolumeTypes.PREDICTED_AFFS: 'volumes/labels/pred_affinities',
                VolumeTypes.LOSS_GRADIENT: 'volumes/loss_gradient',
            },
            every=100,
            output_filename='batch_{iteration}.hdf',
            additional_request=snapshot_request) +
        PrintProfilingStats(every=10)
    )

    print("Starting training...")
    with build(train_pipeline) as b:
        for i in range(max_iteration - trained_until):
            b.request_batch(request)
    print("Training finished")

if __name__ == "__main__":
    set_verbose(False)
    iteration = int(sys.argv[1])
    train_until(iteration)
