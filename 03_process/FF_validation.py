import luigi

import sys
import os
sys.path.append('../src/caffe_luigi')
import numpy as np
from tasks import *

if __name__ == '__main__':

    setup_validation = {
        'setup01_pro01': 'pro05',
        'setup01_pro02': 'pro01',
        'setup01_pro03': 'pro02',
        'setup01_pro04': 'pro03',
        'setup01_pro05': 'pro04',
        'setup01_per01': 'per03',
        'setup01_per02': 'per01',
        'setup01_per03': 'per02',
        'setup01_both_pro01': 'pro05',
        'setup01_both_pro02': 'pro01',
        'setup01_both_pro03': 'pro02',
        'setup01_both_pro04': 'pro03',
        'setup01_both_pro05': 'pro04',
        'setup01_both_per01': 'per03',
        'setup01_both_per02': 'per01',
        'setup01_both_per03': 'per02',
    }

    jobs = []

    # DEBUG
    # for setup, validation in setup_validation.items()[:1]:
    for setup, validation in setup_validation.items():

        combinations = {
            'experiment': 'FF',
            'setups': [ setup ],
            'iterations': [50000, 100000, 150000],
            'samples': [ validation ],
            'thresholds': [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
            'merge_functions': ['mean_aff', 'max_aff'],
            'init_with_maxs': [False],
            'custom_fragments': True,
            'histogram_quantiless': [False],
            'discrete_queues': [True],
            'keep_segmentation': True,
            'dilate_mask': 0,
            'mask_fragments': True}

        range_keys = [
            'setups',
            'iterations',
            'samples',
            'merge_functions',
            'init_with_maxs',
            'histogram_quantiless',
            'discrete_queues']

        jobs.append(EvaluateCombinations(combinations, range_keys))

    set_base_dir(os.path.abspath('..'))

    luigi.build(
            jobs,
            workers=50,
            scheduler_host='slowpoke1.int.janelia.org',
            logging_conf_file='/groups/saalfeld/home/funkej/.luigi/logging.conf'
    )
