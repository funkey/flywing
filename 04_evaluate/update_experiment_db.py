from time import time
import glob
import json
import os
import pandas
import re
import report
import warnings
warnings.filterwarnings('ignore', category=pandas.io.pytables.PerformanceWarning)

def dicts_to_data_frame(dicts):

    columns = set()
    for r in dicts:
        for k in r:
            columns.add(k)

    data_frame = pandas.DataFrame(
            index=range(len(dicts)),
            columns=list(columns))
    for i in range(len(dicts)):
        data_frame.loc[i] = dicts[i]

    return data_frame

def parse_config_file(filename, tokens):

    values = {}
    for line in open(filename):
        for token in tokens:
            # find 'token =' lines, ignore comments
            m = re.search('^[^#]*' + token + '\s*=\s*(.*)$', line)
            if m is not None and len(m.groups()) > 0:
                try:
                    v = eval(m.group(1))
                except:
                    print("%s can not be evaluated -- ignoring this value for token %s"%(m.group(1), token))
                    continue
                # lambdas can't be pickled, we use the strings instead
                if callable(v):
                    v = m.group(1)
                values[token] = v

    return values

def parse_train_until_py(setup_dir):

    tokens = [
            'type',
            'base_lr',
            'momentum',
            'momentum2',
            'delta',
            'weight_decay',
            'lr_policy',
            'gamma',
            'power',
            'component_erosion_steps',
            'simple_augment',
            # older setups store this, no harm done if not found
            # newer setup write this into trainig_options.json
            'data_dir',
            'samples',
            'training_augmentations',
            'testing_augmentations',
    ]

    for f in ['train_until.py', 'train.py']:
        filename = os.path.join(setup_dir, f)
        if not os.path.isfile(filename):
            continue
        return parse_config_file(filename, tokens)

    print("%s does not contain train[_until].py"%setup_dir)
    return {}

def parse_mknet_py(setup_dir):

    tokens = [
            'malis_split_component_phases',
            'ignore_conv_buffer',
            'use_batchnorm',
            'dropout',
            'fmap_start',
            'unet_depth',
            'unet_fmap_inc_rule',
            'unet_fmap_dec_rule',
            'unet_downsampling_strategy',
            'input_shape',
            'output_shape',
    ]

    try:
        return parse_config_file(os.path.join(setup_dir, 'mknet.py'), tokens)
    except IOError:
        print("%s does not contain mknet.py"%setup_dir)
        return {}


def parse_iterations(setup_dir):

    parameters = {}
    iterations = glob.glob(os.path.join(setup_dir, 'net_iter_*.solverstate'))
    iterations = [ os.path.basename(x) for x in iterations ]
    iterations = [ int(x.lstrip('net_iter_').rstrip('.solverstate')) for x in iterations ]
    iterations.sort()
    trained_until = iterations[-1] if len(iterations) > 0 else None
    parameters['trained_until'] = trained_until
    parameters['snapshot_iterations'] = iterations

    return parameters

def parse_train_parameters(setup_dir):

    parameters = {}

    parameters.update(parse_train_until_py(setup_dir))
    parameters.update(parse_mknet_py(setup_dir))
    parameters.update(parse_iterations(setup_dir))

    return parameters

def parse_loss_files(files):
    '''Get a dictionary of iteration -> loss, read from the given files.

    Lines with the same iteration overwrite existing values in the order of files given.'''

    losses = {}
    iter_loss_re   = re.compile(r'.*teration.([0-9]+),* loss\s*=\s*([0-9.]+).*$')
    for f in files:
        if not os.path.exists(f):
            continue
        # print("Parsing " + f)
        for l in open(f):
            result = iter_loss_re.match(l)
            if result is not None:
                losses[int(result.group(1))] = float(result.group(2))

    return losses

def parse_train_losses(experiment, setup):

    # old-style training losses are here
    loss_files = [os.path.join(experiment, '02_train', setup, 'train.err')]
    # old luigi-training losses are here
    loss_files += glob.glob(os.path.join(experiment, '02_train', 'log', 'TrainTask_' + setup + '_CREMI_*.err'))
    # new luigi-training losses are here
    loss_files += glob.glob(os.path.join(experiment, '02_train', setup, 'train_*.err'))
    # or here
    loss_files += glob.glob(os.path.join(experiment, '02_train', setup, 'train_*.out'))
    # or here
    loss_files += glob.glob(os.path.join(experiment, '02_train', setup, 'train.log'))

    return parse_loss_files(loss_files)

def read_setups(experiment):
    '''Returns two tables in the form of dictionaries:

        (setups, train_losses)

    'setups' contains a description of the setups.
    'train_losses' the training loss for each experiment, setup, and iteration.
    '''

    setups_dir = os.path.join(experiment, '02_train')
    setups = [ os.path.basename(x) for x in glob.glob(os.path.join(setups_dir, '*')) ]
    setups = [ x for x in setups if x.startswith('setup') ]

    print("Found setups " + str(setups))

    setup_records = []
    train_losses_dfs = []

    for setup in setups:

        setup_dir = os.path.join(setups_dir, setup)

        record = {}
        record['experiment'] = experiment
        record['setup'] = setup

        # general train options
        try:
            with open(os.path.join(setup_dir, 'train_options.json'), 'r') as f:
                record.update(json.load(f))
        except:
            # That's fine, older setups do that
            # print("Setup %s does not contain train_options.json"%setup)
            pass

        # training parameters
        record.update(parse_train_parameters(setup_dir))

        setup_records.append(record)

        # training losses
        losses = parse_train_losses(experiment, setup)
        df = pandas.DataFrame({
            'setup': [setup]*len(losses),
            'experiment': [experiment]*len(losses),
            'iteration': losses.keys(),
            'loss': losses.values(),
        })
        train_losses_dfs.append(df)

    print("Creating data frames...")
    setups_df = dicts_to_data_frame(setup_records)
    if len(train_losses_dfs) > 0:
        losses_df = pandas.concat(train_losses_dfs, ignore_index=True)
    else:
        losses_df = dicts_to_data_frame([{}])

    return (setups_df, losses_df)

if __name__ == '__main__':

    for experiment in ['cremi_gunpowder', 'snemi_gunpowder', 'fib25', 'cremi', 'segem']:

        setups = []
        losses = []
        results = []

        print
        print("Parsing experiment " + experiment)
        print("--------------------------------------------------")

        setups_df, losses_df = read_setups(experiment)
        setups.append(setups_df)
        losses.append(losses_df)

        for stage in ['03_process_training', '04_process_testing', '03_process']:
            process_dir = os.path.join(experiment, stage, 'processed')
            df = report.read_all_results(data_dir=process_dir)
            df['experiment'] = [experiment]*len(df)
            results.append(df)

        setups = pandas.concat(setups, ignore_index=True)
        losses = pandas.concat(losses, ignore_index=True)
        results = pandas.concat(results, ignore_index=True)

        try:
            os.remove('.experiments.hdf')
        except:
            pass

        print("Storing tables in temporary HDF file...")
        setups.to_hdf('.experiments.hdf', 'setups')
        losses.to_hdf('.experiments.hdf', 'losses')
        results.to_hdf('.experiments.hdf', 'results')
        print("Done.")

        start = time()
        setups = pandas.read_hdf('.experiments.hdf', 'setups')
        losses = pandas.read_hdf('.experiments.hdf', 'losses')
        results = pandas.read_hdf('.experiments.hdf', 'results')
        print("Read back DB in %fs"%(time()-start))
        print("Replacing previous DB with update")
        os.rename('.experiments.hdf', 'experiments_%s.hdf'%experiment)
        print("Done, experiment DB updated.")
