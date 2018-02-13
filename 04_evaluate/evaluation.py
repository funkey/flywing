import report
reload(report)
import numpy as np
import pandas

# Unfortunately, pandas does not handle NaN as we want it in group-by 
# operations. So we replce it with this magic value and convert back later.
NAN_PLACEHOLDER = -np.inf

metric_keys = ['threshold', 'voi_split', 'voi_merge', 'voi_sum', 'rand_split', 'rand_merge', 'arand', 'cremi_score']
nonrelevant_keys = ['raw', 'gt', 'affinities', 'segmentation', 'offset' ]

def find_best(results, score='cremi_score', k=3):
    '''Find the best configurations for each of the given samples in one experiment.
    '''

    # nan values lead to exclusion in group-by operations
    results = results.replace(np.nan, NAN_PLACEHOLDER)

    # sort by score
    #   the sorting is actually crucial: assuming that pandas is not changing 
    #   the row order when grouping, this ensures that we find the best scores 
    #   as the first entries in each group
    results = results.sort_values(by=score)

    # for each configuration, select the row with the minimal score
    configuration_keys = list(set(results.keys()) - set(metric_keys) - set(nonrelevant_keys))
    configuration_results = results.groupby(configuration_keys).first()

    # now we have a dataframe with hierarchical index, flatten it for further 
    # processing
    configuration_results = configuration_results.reset_index()

    # sort again by score
    #   the grouping keeps the rows inside a group sorted, but between groups 
    #   the values are not sorted anymore (and can't be in general), so we sort 
    #   again to make sure the next group command has the minimal values at the 
    #   top
    configuration_results = configuration_results.sort_values(by=score)

    # group again by sample, take first k per sample
    sample_results = configuration_results.groupby('sample').head(k)

    # sort again by group
    #   this seems necessary, as otherwise pandas shuffles the samples by 
    #   sorting by score
    sample_results = sample_results.sort_values(by=['sample', score])

    # convert nan back
    sample_results = sample_results.replace(NAN_PLACEHOLDER, np.nan)

    return sample_results

def find_best_average(results, score='cremi_score', k=3):
    '''Find the configurations that perform best on average over all samples.'''

    # nan values lead to trouble in group-by aggregations
    results = results.replace(np.nan, NAN_PLACEHOLDER)

    # replace sample results with averages
    configuration_keys = list(
        set(results.keys()) -
        set(metric_keys) -
        set(nonrelevant_keys))
    configuration_keys.remove('sample')
    configuration_keys.append('threshold')

    agg_keys = list(metric_keys)
    agg_keys.remove('threshold')
    agg_functions = { k: 'mean' for k in agg_keys}
    agg_functions['sample'] = lambda x: "average_over_%d"%len(x)
    results = results.groupby(configuration_keys).agg(agg_functions)
    results = results.reset_index()
    # results['sample'] = 'average'

    return find_best(results, score, k)

def curate_value(v):
    # holy cow, this null/nan handling is really broken...
    if pandas.isnull(v):
        return np.nan
    return v

def create_evaluation_report(results, k=5, setups=None, score='cremi_score', show_plots=False):

    best_results = []
    # show average only if there is more than one sample
    if len(np.unique(results['sample'])) > 1:
        best_results.append(find_best_average(results, k=k, score=score))
    best_results.append(find_best(results, k=k, score=score))

    print("Evaluated setups: " + str(np.unique(results['setup'])))

    for best in best_results:

        samples = np.unique(best['sample'])
        configuration_keys = list(set(results.keys()) - set(metric_keys) - set(nonrelevant_keys))

        print("%d best configurations:"%k)
        for sample in samples:

            sample_best = best[best['sample']==sample]

            first_table_keys = [
                'sample',
                'setup',
                'iteration',
                'merge_function',
                'threshold',
                'cremi_score',
                'voi_split',
                'voi_merge',
                'voi_sum',
                'rand_split',
                'rand_merge',
                'arand']

            table_keys = first_table_keys + [
                key
                for key in results.keys()
                if key not in first_table_keys and key not in nonrelevant_keys
            ]
            table_frame = sample_best.loc[:,table_keys]
            if setups is not None and len(setups) > 0:
                table_frame = pandas.merge(table_frame, setups, how='left', on='setup')
            report.render_table(table_frame)

            if show_plots:
                groups = [
                    {
                        'sample': sample,
                    }
                ]
                figures = [
                    {'x_axis': 'threshold',  'y_axis': 'voi_sum', 'title': 'VOI'},
                    {'x_axis': 'threshold',  'y_axis': 'arand', 'title': 'ARAND'},
                    {'x_axis': 'threshold',  'y_axis': 'cremi_score', 'title': 'CREMI score'},
                    {'x_axis': 'voi_split',  'y_axis': 'voi_merge', 'title': 'VOI'},
                    {'x_axis': 'rand_split', 'y_axis': 'rand_merge', 'title': 'RAND'},
                ]
                configurations = [
                    # for all thresholds
                    dict(
                        {
                            c: curate_value(row[1][c]) for c in configuration_keys
                        },
                        **{'style':'line'}
                    )
                    for row in sample_best.iterrows()
                ] + [
                    # for best threshold only
                    {
                        c: curate_value(row[1][c]) for c in configuration_keys + ['threshold']
                    }
                    for row in sample_best.iterrows()
                ]
                report.plot(groups, figures, configurations, results.sort_values(by='threshold'))
