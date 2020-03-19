"""Utility functions to be used by analysis tasks."""
import os
import re
import collections
import json
import numpy as np
import pandas as pd
import matplotlib as mpl
from matplotlib import pyplot as plt
import seaborn as sns


#File Utils
def load_metadata(directory):
    """Load the metadata file 'experiment_metadata.json' from experiment dir"""
    val = None
    try:
        fname = f"{directory}/experiment_metadata.json"
        with open(fname) as fin:
            val = json.load(fin)
    except IOError as ex:
        print(ex)
        val = None
    return val

def validate_required_fields(metadata):
    """Check if the metadata can be used on analysis. This is evolving"""
    required = ['RESULTS-BALANCED_ACCURACY']
    val = True
    for field in required:
        val = val & (field in metadata)
    return val

def load_result_directories(directories):
    """
    Walks a directory list in which every directory contains experiment
    results, loading a metadata file for each directory.
    Validates and returns the metadata dict
    """
    dirs = []
    for search_dir in directories:
        _, directory, _ = list(os.walk(search_dir, followlinks=True))[0]
        dirs += [f"{search_dir}{subdir}" for subdir in directory]

    results = {}
    for directory in dirs:
        metadata = load_metadata(directory)
        if metadata is None or not validate_required_fields(metadata):
            print(f"WARNING: Invalid metadata: {directory}")
        else:
            results[directory] = metadata
    return results


#Data Utils
def load_full_dataset(results):
    """Converts a results dict to a DataFrame"""
    experiments = results.keys()
    return pd.DataFrame({
        #Experiment and Environment
        'runid': [results[k]['EXPERIMENT-RUNID'] for k in experiments],
        'version': [results[k]['EXPERIMENT-VERSION'] for k in experiments],
        'GPU': [gpu(results[k]) for k in experiments],
        #Data
        'dataset': [results[k]['DATA-DATASET_FN'] for k in experiments],
        'rows_to_load': [rows_to_load(results[k]) for k in experiments],
        'training_set_size': [training_set_size(results[k]) for k in experiments],
        'test_set_size': [test_set_size(results[k]) for k in experiments],
        'max_seq_len': [max_seq_len(results[k]) for k in experiments],
        'output_dim': [results[k]['DATA-OUTPUT_DIM'] for k in experiments],
        #Model and training
        'lm': [lm(results[k]) for k in experiments],
        'batch_size': [results[k]['TRAIN-BATCH_SIZE'] for k in experiments],
        'learning_rate': [learning_rate(results[k]) for k in experiments],
        #Results
        'training_time': [training_time(results[k]) for k in experiments],
        'bac': [results[k]['RESULTS-BALANCED_ACCURACY'] for k in experiments],
        'min_loss': [min_loss(results[k]) for k in experiments],
        'last_loss': [last_loss(results[k]) for k in experiments],
        'total_epochs': [epochs(results[k]) for k in experiments],
        'best_epoch': [best_epoch(results[k]) for k in experiments],
        'val_loss': [val_loss(results[k]) for k in experiments],
        'test_bac': [test_bac(results[k]) for k in experiments]
    })

def warn_if_experiments_differ(data, must_be_unique):
    """
    Sends a warning if a field which is expected to have one unique value
    has more. Use to inspect the dataframe
    """
    for field in must_be_unique:
        if len(data[field].unique()) != 1:
            print(f"WARNING: There are experiments with different {field}:")
            print(data[field].unique())

def extract_type_from_nnlm(data):
    """
    Extract the version for the NNLM collection of Language Models
    https://tfhub.dev/google/collections/nnlm/1
    """
    return data['lm'].str.extract(r'nnlm-(e.-.*)-w.*')


def average_list_metric(
        data,
        metric_field,
        dimension_field,
        ignore_trailing=True):
    """
    Averages "dimension_field" values of list field (metric_field).
    This is used on experiments results that are a list, for instance,
    val_loss or accuracy. This can be used to plot learning curves.

    The argument 'ignore_trailing' will stop averaging values on the list
    with the least arguments. For example,

    if ignore_trailing is True:  [1, 1, 3] [1, 2] will produce [1, 1,5]
    if ignore_trailing is False: [1, 1, 3] [1, 2] will produce [1, 1,5, 3]
    """

    def val_or_none(list_, index):
        if index < len(list_):
            return list_[index]
        return None

    if not ignore_trailing:
        print("WARNING: ignore_trailing set to False. "
              "Last epochs for some dimensions may be misleading.")

    values = collections.defaultdict(list)
    for _, row in data.iterrows():
        values[row[dimension_field]].append(row[metric_field])
    aggregated = collections.defaultdict(list)
    for dim, metrics in values.items():
        aggregated[dimension_field].append(dim)
        vals = []
        if ignore_trailing:
            epochs = min([len(x) for x in metrics])
        else:
            epochs = max([len(x) for x in metrics])
        for index in range(epochs):
            if not ignore_trailing:
                m = [val_or_none(x, index) for x in metrics]
                m = list(filter(lambda x: x is not None, m))
                val = sum(m) / len(m)
            else:
                val = sum([x[index] for x in metrics]) / len(metrics)
            vals.append(val)
        aggregated[metric_field].append(vals)
    return pd.DataFrame(aggregated)


#Graph utils
#from https://stackoverflow.com/questions/1855884/determine-font-color-based-on-background-color
def contrast_color(color, blackish='black', whiteish='whitesmoke'):
    """Selects white(ish) or black(ish) for text to contrast over some RGB"""
    luminance = (0.299 * color[0] + 0.587 * color[1] + 0.114 * color[2])
    if luminance > 0.6:
        return blackish
    return whiteish

def colors_by_value(values, color_space='muted', return_legend_handles=False):
    """
    Creates a list of colors based on the unique categorical values
    """
    categories = sorted(values.unique())
    pal = sns.color_palette(color_space, len(categories))
    col_dict = dict(zip(categories, pal))
    #colors = list(values.map(col_dict))
    colors = [col_dict[val] for val in values]
    if return_legend_handles:
        handles = []
        for k, v in col_dict.items():
            handles.append(mpl.patches.Patch(color=v, label=k))
        return (colors, handles)
    return colors

#TODO: Revisar
#https://matplotlib.org/3.1.3/gallery/statistics/barchart_demo.html
#para mejor manejo de la posición del texto.
# Hacer el blog post despues con la solucion
##http://eduardofv.com/wp-admin/post.php?post=517&action=edit
#https://colab.research.google.com/drive/1kwKuOwim7ngYmFSRjkVYMi5_K6WA9vmD
def annotate_barh(ax, fontsize='x-small'):
    """Adds value labels inside the bars of a barh plot"""
    plt.draw()
    for patch in ax.patches:
        label = f"{patch.get_width():1.4f}"
        p_x = patch.get_width()
        p_y = patch.get_y()
        #Put an invisible text to measure it's dimensions
        txt = plt.text(p_x, p_y, label, fontsize=fontsize, alpha=0.0)
        bbox = txt.get_window_extent().transformed(ax.transData.inverted())
        t_w = bbox.width * 1.1
        t_h = bbox.height
        p_y += (patch.get_height() - t_h)/1.5
        if t_w > 0.9 * patch.get_width():
            plt.text(p_x, p_y, label, fontsize=fontsize)
        else:
            p_x -= t_w
            col = contrast_color(patch.get_facecolor())
            plt.text(p_x, p_y, label, fontsize=fontsize, color=col)

def plot_learning_curve(data, curve_field, dimension_field):
    """
    Plots learning curves contained as lists in in the *curve_field* of the
    DataFrame. Dimension field will be used for the labels of each sample.
    """
    #fig = plt.figure()
    ax = plt.axes()
    df1 = data.sort_values(dimension_field)
    df1['color'] = colors_by_value(df1[dimension_field])
    for _, row in df1.iterrows():
        plt.plot(row[curve_field],
                 label=row[dimension_field],
                 color=row['color'])
    ax.legend()

#Config object field parsing
# These methods convert the values from the metadata to standard values that
# can be used in the analysis. Fills non-existent values, select correct fields
# set default, etc
#pylint: disable=C0116
def learning_rate(metadata):
    val = None
    try:
        val = metadata['MODEL-OPTIMIZER_FULL_CONFIG']['learning_rate']
    except KeyError:
        print("WARNING: Actual learning_rate not found")
    return val

def training_time(metadata):
    val = None
    if 'EXPERIMENT-TRAINING_TOOK' in metadata:
        val = metadata['EXPERIMENT-TRAINING_TOOK']
    return val

def lm(metadata):
    lm_val = None
    if 'TFHUB-EMB_MODEL' in metadata:
        lm_val = metadata['TFHUB-EMB_MODEL']
        match = re.search("https://tfhub.dev/google/([^/]+)/.$", lm_val)
        if match is not None:
            lm_val = match.group(1)
        else:
            print(f"WARNING: LM could not be parsed from {lm_val}")
            lm_val = "LM Not Found"
    elif 'HUG-EMB_MODEL' in metadata:
        lm_val = metadata['HUG-EMB_MODEL']
    return lm_val

def gpu(metadata):
    if metadata['EXPERIMENT-ENVIRONMENT'][4] == 'GPU: available':
        return metadata['EXPERIMENT-ENVIRONMENT'][5].split('(')[0].split(":")[1].strip()
    return "GPU: Not available"

def max_seq_len(metadata):
    max_seq_len_val = 'Full'
    if 'MODEL-HUG_MAX_SEQ_LEN' in metadata:
        max_seq_len_val = metadata['MODEL-HUG_MAX_SEQ_LEN']
    elif 'MODEL-BERT_MAX_SEQ_LEN' in metadata:
        max_seq_len_val = metadata['MODEL-BERT_MAX_SEQ_LEN']
    return max_seq_len_val

def rows_to_load(metadata):
    rows_to_load_val = "All"
    if 'DATA-ROWS_TO_LOAD' in metadata:
        rows_to_load_val = metadata['DATA-ROWS_TO_LOAD']
    return rows_to_load_val

def min_loss(metadata):
    val = None
    if 'RESULTS-HISTORIES' in metadata and metadata['RESULTS-HISTORIES']:
        val = min(metadata['RESULTS-HISTORIES'][0]['val_loss'])
    return val

def last_loss(metadata):
    val = None
    if 'RESULTS-HISTORIES' in metadata and metadata['RESULTS-HISTORIES']:
        val = metadata['RESULTS-HISTORIES'][0]['val_loss'][-1]
    return val

def epochs(metadata):
    val = 'NA'
    if 'RESULTS-HISTORIES' in metadata and metadata['RESULTS-HISTORIES']:
        val = len(metadata['RESULTS-HISTORIES'][0]['loss'])
    return val

def best_epoch(metadata):
    val = 'NA'
    if 'RESULTS-HISTORIES' in metadata and metadata['RESULTS-HISTORIES']:
        if 'val_loss' in metadata['RESULTS-HISTORIES'][0]:
            v_loss = metadata['RESULTS-HISTORIES'][0]['val_loss']
            val = np.argmin(v_loss) + 1
    return val

def training_set_size(metadata):
    return metadata['DATA-TRAINING_SET_SIZE']

def test_set_size(metadata):
    return metadata['DATA-TEST_SET_SIZE']

def test_bac(metadata):
    val = None
    try:
        val = metadata['RESULTS-HISTORIES'][0]['test_bac']
    except KeyError:
        print(f"WARNING: test_bac not found in Histories")
    return val

def val_loss(metadata):
    val = None
    try:
        val = metadata['RESULTS-HISTORIES'][0]['val_loss']
    except KeyError:
        print(f"WARNING: val_loss not found in Histories")
    return val
