import report
reload(report)
import numpy as np
import pandas

# Unfortunately, pandas does not handle NaN as we want it in group-by 
# operations. So we replce it with this magic value and convert back later.
NAN_PLACEHOLDER = -np.inf

metric_keys = ['threshold', 'seg_score', 'tra_score']
nonrelevant_keys = ['raw', 'gt', 'affinities', 'segmentation', 'offset', 'thresholds', 'error', 'output_basenames' ]

def find_best(results, per='sample', score='tra_score', k=3, lower_is_better=True):
    '''Find the best configurations for each of the given samples in one experiment.
    '''
    drop = list(set(results.keys()).intersection(set(nonrelevant_keys)))
    results = results.drop(columns=drop)

    # nan values lead to exclusion in group-by operations
    results = results.replace(np.nan, NAN_PLACEHOLDER)

    # sort by score
    #   the sorting is actually crucial: assuming that pandas is not changing 
    #   the row order when grouping, this ensures that we find the best scores 
    #   as the first entries in each group
    results = results.sort_values(by=score, ascending=lower_is_better)

    # for each configuration, select the row with the minimal score
    configuration_keys = list(set(results.keys()) - set(metric_keys) - set(nonrelevant_keys))
    configuration_groups = results.groupby(configuration_keys)
    # for key, item in configuration_groups:
        # print(configuration_groups.get_group(key), "\n")
    configuration_results = configuration_groups.first()

    # now we have a dataframe with hierarchical index, flatten it for further 
    # processing
    configuration_results = configuration_results.reset_index()

    # sort again by score
    #   the grouping keeps the rows inside a group sorted, but between groups 
    #   the values are not sorted anymore (and can't be in general), so we sort 
    #   again to make sure the next group command has the minimal values at the 
    #   top
    configuration_results = configuration_results.sort_values(
        by=score,
        ascending=lower_is_better)

    # group again by 'per', take first k per 'per'
    sample_results = configuration_results.groupby(per).head(k)

    # sort again by group
    #   this seems necessary, as otherwise pandas shuffles the samples by 
    #   sorting by score
    sample_results = sample_results.sort_values(
        by=[per, score],
        ascending=lower_is_better)

    # convert nan back
    sample_results = sample_results.replace(NAN_PLACEHOLDER, np.nan)

    return sample_results

def condense(values):

    for i in range(1, len(values)):
        if values.iloc[i] != values.iloc[i-1]:
            return 'n/a'
    return values.iloc[0]

def average(results, group_by):

    drop = list(set(results.keys()).intersection(set(nonrelevant_keys)))
    results = results.drop(columns=drop)

    # nan values lead to trouble in group-by aggregations
    results = results.replace(np.nan, NAN_PLACEHOLDER)

    # tell pandas how to agglomerate values in groups
    agg_functions = { k: condense for k in results.keys() }
    agg_keys = list(metric_keys)
    agg_keys.remove('threshold')
    agg_functions.update({ k: 'mean' for k in agg_keys})
    agg_functions['sample'] = lambda x: "average over %d"%len(x)

    # group results by configurations
    groups = results.groupby(group_by)

    # agglomerate
    results = groups.agg(agg_functions)
    results = results.drop(columns=group_by)
    results = results.reset_index()

    return results

def curate_value(v):
    # holy cow, this null/nan handling is really broken...
    if pandas.isnull(v):
        return np.nan
    return v

def create_evaluation_report(
        results,
        k=5,
        setups=None,
        score='tra_score',
        show_plots=False,
        lower_is_better=True):

    drop = list(set(results.keys()).intersection(set(nonrelevant_keys)))
    results = results.drop(columns=drop)

    best_results = []
    # show average only if there is more than one sample
    if len(np.unique(results['sample'])) > 1:
        best_results.append(find_best_average(results, k=k, score=score,
            lower_is_better=lower_is_better))
    best_results.append(find_best(results, k=k, score=score,
        lower_is_better=lower_is_better))

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
                'seg_score',
                'tra_score']

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
                    {'x_axis': 'threshold',  'y_axis': 'seg_score', 'title': 'SEG'},
                    {'x_axis': 'threshold',  'y_axis': 'tra_score', 'title': 'TRA'},
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
