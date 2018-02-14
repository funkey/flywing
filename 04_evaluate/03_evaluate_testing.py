import luigi

import os
from tasks import *

if __name__ == '__main__':

    setup_testing = {
        'setup01_pro01': 'pro01',
        'setup01_pro02': 'pro02',
        'setup01_pro03': 'pro03',
        'setup01_pro04': 'pro04',
        'setup01_pro05': 'pro05',
        'setup01_per01': 'per01',
        'setup01_per02': 'per02',
        'setup01_per03': 'per03',
        'setup01_both_pro01': 'pro01',
        'setup01_both_pro02': 'pro02',
        'setup01_both_pro03': 'pro03',
        'setup01_both_pro04': 'pro04',
        'setup01_both_pro05': 'pro05',
        'setup01_both_per01': 'per01',
        'setup01_both_per02': 'per02',
        'setup01_both_per03': 'per03',
    }

    base_configuration = {
        'experiment': 'FF',
        'init_with_max': False,
        'custom_fragments': True,
        'histogram_quantiles': False,
        'discrete_queue': True,
        'keep_segmentation': True,
        'dilate_mask': 0,
        'mask_fragments': True,
        'test_run': 1
    }

    setup_configurations = {
        "setup01_both_pro05": {
            "merge_function": "mean_aff", 
            "thresholds": [
                0.8
            ], 
            "iteration": 50000
        }, 
        "setup01_both_per03": {
            "merge_function": "mean_aff", 
            "thresholds": [
                0.9
            ], 
            "iteration": 150000
        }, 
        "setup01_both_pro04": {
            "merge_function": "mean_aff", 
            "thresholds": [
                0.8
            ], 
            "iteration": 150000
        }, 
        "setup01_both_per01": {
            "merge_function": "mean_aff", 
            "thresholds": [
                0.9
            ], 
            "iteration": 100000
        }, 
        "setup01_per03": {
            "merge_function": "mean_aff", 
            "thresholds": [
                0.7
            ], 
            "iteration": 100000
        }, 
        "setup01_per02": {
            "merge_function": "mean_aff", 
            "thresholds": [
                0.8
            ], 
            "iteration": 50000
        }, 
        "setup01_per01": {
            "merge_function": "mean_aff", 
            "thresholds": [
                0.9
            ], 
            "iteration": 150000
        }, 
        "setup01_both_per02": {
            "merge_function": "mean_aff", 
            "thresholds": [
                0.9
            ], 
            "iteration": 150000
        }, 
        "setup01_pro05": {
            "merge_function": "mean_aff", 
            "thresholds": [
                0.8
            ], 
            "iteration": 150000
        }, 
        "setup01_pro04": {
            "merge_function": "mean_aff", 
            "thresholds": [
                0.8
            ], 
            "iteration": 100000
        }, 
        "setup01_pro03": {
            "merge_function": "mean_aff", 
            "thresholds": [
                0.9
            ], 
            "iteration": 150000
        }, 
        "setup01_pro02": {
            "merge_function": "mean_aff", 
            "thresholds": [
                0.8
            ], 
            "iteration": 150000
        }, 
        "setup01_pro01": {
            "merge_function": "mean_aff", 
            "thresholds": [
                0.8
            ], 
            "iteration": 100000
        }, 
        "setup01_both_pro01": {
            "merge_function": "mean_aff", 
            "thresholds": [
                0.8
            ], 
            "iteration": 150000
        }, 
        "setup01_both_pro02": {
            "merge_function": "mean_aff", 
            "thresholds": [
                0.8
            ], 
            "iteration": 100000
        }, 
        "setup01_both_pro03": {
            "merge_function": "mean_aff", 
            "thresholds": [
                0.8
            ], 
            "iteration": 150000
        }
    }

    jobs = []

    for setup, testing in setup_testing.items():

        combination = {}
        combination.update(base_configuration)
        combination.update(setup_configurations[setup])
        combination['setup'] = setup
        combination['sample'] = testing

        range_keys = []

        jobs.append(EvaluateCombinations(combination, range_keys))

    set_base_dir(os.path.abspath('..'))

    luigi.build(
            jobs,
            workers=50,
            scheduler_host='slowpoke1.int.janelia.org',
            logging_conf_file='/groups/saalfeld/home/funkej/.luigi/logging.conf'
    )
