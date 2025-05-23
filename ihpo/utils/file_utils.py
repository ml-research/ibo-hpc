import pandas as pd
import os

def load_task_files(files):
    """
        Load files for each task and concatenate
    """
    file_dfs = []
    for f in files:
        df = pd.read_csv(f, index_col=0)
        file_dfs.append(df)

    return file_dfs

def create_dir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)