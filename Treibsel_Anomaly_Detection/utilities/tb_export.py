#!/usr/bin/env python
#######################################################
### tb_export.py: What is the task of the code. ( in Short )
# General Text and Notice
#######################################################

"""
NumberList holds a sequence of numbers, and defines several statistical
operations (mean, stdev, etc.) FrequencyDistribution
holds a mapping from items (not necessarily numbers)
to counts, and defines operations such as Shannon
entropy and frequency normalization.
"""

__date__ = "18.03.2024"
__author__ = "Sven Nivera & Tjark Ziehm"
__copyright__ = "Copyright 2024, BalticMaterials"
__credits__ = ["Sven Nivera"]
__license__ = "MIT"
__version__ = "1.0.1"
__maintainer__ = "Tjark Ziehm"
__email__ = "kontakt@balticmaterials.de"
__status__ = "Development"
import os
import pandas as pd
from tensorflow.python.summary.summary_iterator import summary_iterator

def convert_tb_data(root_dir, sort_by=None):
    """Convert local TensorBoard data into Pandas DataFrame.
    
    Function takes the root directory path and recursively parses
    all events data.    
    If the `sort_by` value is provided then it will use that column
    to sort values; typically `wall_time` or `step`.
    
    *Note* that the whole data is converted into a DataFrame.
    Depending on the data size this might take a while. If it takes
    too long then narrow it to some sub-directories.
    
    Paramters:
        root_dir: (str) path to root dir with tensorboard data.
        sort_by: (optional str) column name to sort by.
    
    Returns:
        pandas.DataFrame with [wall_time, name, step, value] columns.
    
    """
    

    def convert_tfevent(filepath):
        return pd.DataFrame([
            parse_tfevent(e) for e in summary_iterator(filepath) if len(e.summary.value)
        ])

    def parse_tfevent(tfevent):
        return dict(
            wall_time=tfevent.wall_time,
            name=tfevent.summary.value[0].tag,
            step=tfevent.step,
            value=float(tfevent.summary.value[0].simple_value),
        )
    
    columns_order = ['wall_time', 'name', 'step', 'value']
    
    out = []
    for (root, _, filenames) in os.walk(root_dir):
        for filename in filenames:
            if "events.out.tfevents" not in filename:
                continue
            file_full_path = os.path.join(root, filename)
            out.append(convert_tfevent(file_full_path))

    # Concatenate (and sort) all partial individual dataframes
    all_df = pd.concat(out)[columns_order]
    if sort_by is not None:
        all_df = all_df.sort_values(sort_by)
        
    return all_df.reset_index(drop=True)

def save_as_csv(ex: tuple(str, pd.DataFrame)) -> None:
    path = "./eval/"
    ex[1].to_csv(path + ex[0].replace(":", "_") + ".csv")

if __name__ == "__main__":
    dir_path = "./runs/"
    exp_name = "Apr25_12-37-09_balticmaterials-MS-7B98"
    df = convert_tb_data(f"{dir_path}/{exp_name}")
    df.drop(columns="wall_time", inplace=True)
    df = df.reindex(columns=['step', 'value', 'name'])

    metrics = ["dice", "f1", "jaccard", "loss", "precision", "recall"]
    exp = []
    for fn in range(5):
        for batch in range(5):
            for lr in range(3):
                df_new = pd.DataFrame()
                name = df.iloc[0]
                name = name["name"]
                # take 50 values per metric and add them as a column to a new frame
                # order: dice, f1, jaccard, loss, precision, recall
                
                for met in range(6):
                    fity = df.iloc[:50]
                    assert fity["step"].to_numpy()[0] == 0
                    df_new[metrics[met]] = fity["value"].to_numpy()
                    df = df.iloc[50:]
                exp.append((name, df_new))

    for ex in exp:
        save_as_csv(ex)