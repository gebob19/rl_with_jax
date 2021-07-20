import tensorflow as tf 
import traceback
import numpy as np 
import matplotlib.pyplot as plt 

import pandas as pd
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

# Extraction function
def tflog2pandas(path):
    DEFAULT_SIZE_GUIDANCE = {
        "compressedHistograms": 1,
        "images": 1,
        "scalars": 0,  # 0 means load all
        "histograms": 1,
    }
    runlog_data = pd.DataFrame({"metric": [], "value": [], "step": []})
    try:
        event_acc = EventAccumulator(path, DEFAULT_SIZE_GUIDANCE)
        event_acc.Reload()
        tags = event_acc.Tags()["scalars"]
        for tag in tags:
            event_list = event_acc.Scalars(tag)
            values = list(map(lambda x: x.value, event_list))
            step = list(map(lambda x: x.step, event_list))
            r = {"metric": [tag] * len(step), "value": values, "step": step}
            r = pd.DataFrame(r)
            runlog_data = pd.concat([runlog_data, r])
    # Dirty catch of DataLossError
    except Exception:
        print("Event file possibly corrupt: {}".format(path))
        traceback.print_exc()
    return runlog_data

def plot_metric(bpath, metric_name, plot_every=1):
    dfs = []
    for p in bpath.iterdir():
        p = str(list(p.iterdir())[0])
        df = tflog2pandas(p)
        dfs.append(df)

    print(f'comparing {len(dfs)} different runs...')

    m = metric_name
    min_epis = min([len(df[df['metric'] == m]) for df in dfs]) # min number of episodes
    values = np.array([df[df['metric'] == m][:min_epis].value.values for df in dfs])

    # plot every `plot_every` steps 
    step_idxs = np.arange(0, values.shape[1], plot_every)
    values = values[:, step_idxs]

    median = np.mean(values, 0)
    std = np.std(values, 0)
    x = np.arange(len(median))
    y = median 
    ci = std 

    _, ax = plt.subplots()
    ax.plot(x, y)
    ax.fill_between(x, (y-ci), (y+ci), color='b', alpha=.1)
    ax.set_title(m)
    
    return ax 
