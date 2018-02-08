import h5py

samples = [
    'proper_disc/Ecd_20141010_P2.cleaned.merged',
    'proper_disc/Ecd_20150310_P2.cleaned.merged',
    'proper_disc/Ecd_20150418_P1.cleaned.merged',
    'proper_disc/Ecd_20150608_P1.cleaned.merged',
    'proper_disc/Ecd_20161213_F0.cleaned.merged',
    'peripodial/Ecd_20141010_P2.cleaned.merged',
    'peripodial/Ecd_20150310_P2.cleaned.merged',
    'peripodial/Ecd_20150418_P1.cleaned.merged',
]

datasets = [
    'volumes/raw',
    'volumes/labels/divisions',
    'volumes/labels/boundaries',
    'volumes/labels/cells',
    'volumes/labels/lineages',
    'volumes/labels/ignore',
]

for sample in samples:
    print("Adding attributes to %s..."%sample)
    with h5py.File(sample + '.hdf', 'r+') as f:
        for dataset in datasets:
            f[dataset].attrs['offset'] = (0, 0, 0)
            f[dataset].attrs['resolution'] = (1, 1, 1)
