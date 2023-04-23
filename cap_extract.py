#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import dhedfreader
import glob
import os
from mne.io import concatenate_raws, read_raw_edf
import codecs
from datetime import datetime, timedelta
import math
from scipy.signal import resample


# In[2]:


DATA_DIR = '/share/wequ0318/CAP/physionet.org/files/capslpdb/1.0.0'
OUTPUT_DIR = '/share/wequ0318/cap_clean/cap'
channels = {'P4-O2': '_P4_O2', 'O2':'_O2', 'O2-A1': '_O2_A1', 'O2A1': '_O2_A1'}


# In[3]:


ann2label = {
    "MCAP-A1": 0,
    "MCAP-A2": 1,
    "MCAP-A3": 2
}


# In[4]:


def if_duplicate(time_duration_dict, epoch_time, duration):
    for t, d in time_duration_dict.items():
        t_t = datetime.strptime(t, "%H:%M:%S")
        e_t = datetime.strptime(epoch_time, "%H:%M:%S")
        print(t, epoch_time)
        delta = e_t - t_t
        delta_sec = delta.total_seconds() + 86400 if delta.total_seconds() < -(12*3600) else delta.total_seconds()
        if int(delta_sec) < d: return True
        if delta_sec < 0: break
    return False


# In[ ]:


max_duration_sec = 0
subject_id = 0
for f in glob.glob('{}/*.edf'.format(DATA_DIR)):
    subject_id += 1
    ann = f.replace('.edf', '.txt')
    df_ann = pd.read_csv(ann, skiprows=20 if 'n16.edf' in f else 21, delimiter='\t')
    raw = read_raw_edf(f, preload=True, stim_channel=None)
    sampling_rate = raw.info['sfreq']
    print("file {} contains channels {}".format(f, raw.info['ch_names']))
    select_ch = None
    suffix = None
    for ch in raw.info['ch_names']:
        if ch in list(channels.keys()):
            select_ch = ch
            suffix = channels[select_ch]
            break
    print("selected channel {} and suffix is {}".format(select_ch, suffix))
    out_f = os.path.join(OUTPUT_DIR, os.path.basename(f.replace('.edf', suffix+'.npz')))
    if os.path.exists(out_f):
        continue
    print('output to', out_f)  
    raw_ch_df = raw.to_data_frame(scaling_time=100.0)[select_ch]
    raw_ch_df = raw_ch_df.to_frame()
    raw_ch_df.set_index(np.arange(len(raw_ch_df)))

    # Get raw header
    file = codecs.open(f, 'r', encoding="utf-8")
    reader_raw = dhedfreader.BaseEDFReader(file)
    reader_raw.read_header()
    h_raw = reader_raw.header
    file.close()
    raw_start_dt = datetime.strptime(h_raw['date_time'], "%Y-%m-%d %H:%M:%S")
    df_ann['Event'].fillna('UNKNOWN', inplace=True)
    df_ann_cap = df_ann[df_ann['Event'].str.startswith('MCAP-')]
    # annotation start time
    start_time = h_raw['date_time'].split(' ')[0] + ' ' + list(df_ann_cap.head(1).to_dict()['Time [hh:mm:ss]'].values())[0]
    start_time = start_time.replace('.', ':')
    print('recording start time {}, annotation start time {}'.format(raw_start_dt, start_time))
    t1 = datetime.strptime(start_time, "%Y-%m-%d %H:%M:%S")
    # normally, annotation starts later than eeg.
    delta = t1 - raw_start_dt
    print(f"Time difference is {delta.total_seconds()} seconds")
    
    cnt = 0
    if delta.total_seconds() < 0 and delta.total_seconds() > -3000:
        print('handle the situation that annotation starts earlier than eeg')
        for _, row in df_ann_cap.iterrows():
            cnt += 1
            if row['Event'] not in list(ann2label.keys()):
                continue
            this_epoch_time = start_time.split(' ')[0] + ' ' + row['Time [hh:mm:ss]'].replace('.', ':')
            this_t = datetime.strptime(this_epoch_time, "%Y-%m-%d %H:%M:%S")
            delta = (this_t - raw_start_dt)
            print(this_t, raw_start_dt, delta, delta.total_seconds() > 0)
            if delta.total_seconds() > 0:
                start_time = this_epoch_time
                print('adjusted start time', start_time)
                break
    
    labels = []
    label_idx = []
    last_epoch_time = start_time.split(' ')[-1]
    # start from where we have labels
    start_idx = delta.total_seconds() * sampling_rate if delta.total_seconds() >= 0 else (delta.total_seconds() + 86400) * sampling_rate

    duration_sec = 0
    
    time_duration = {}
    
    for _, row in df_ann_cap.iterrows():
        cnt -= 1
        if cnt > 0:
            continue
            
        if row['Event'] not in list(ann2label.keys()):
            continue

        this_duratioin_sec = int(row['Duration[s]']) if 'n16.edf' not in f else int(row['Duration [s]'])
        this_epoch_time = row['Time [hh:mm:ss]'].replace('.', ':')
        if if_duplicate(time_duration, this_epoch_time, this_duratioin_sec):
            continue
        else:
            time_duration[this_epoch_time] = this_duratioin_sec
            
        label = ann2label[row['Event']]
        last_t = datetime.strptime(last_epoch_time, "%H:%M:%S")
        this_t = datetime.strptime(this_epoch_time, "%H:%M:%S")
        
        # may skip some epochs
        offset = (this_t - last_t).total_seconds()
        print("cnt:", cnt, "last_epoch_time:", last_epoch_time, "this_epoch_time:", this_epoch_time, "offset:",
              offset, "start_idx:", start_idx, "duration_sec:", duration_sec)

        if offset < 0:
            offset += 86400
        offset -= duration_sec
        start_idx += offset * sampling_rate
        
        duration_sec = this_duratioin_sec
        max_duration_sec = duration_sec if duration_sec > max_duration_sec else max_duration_sec
        print('Current max duration:', max_duration_sec)
        
        time_duration[this_epoch_time] = duration_sec
        
        duration_epoch = duration_sec // 1
        # set indices for this epoch
        idx = int(start_idx) + np.arange(duration_sec * sampling_rate, dtype=np.int)
        start_idx += int(duration_sec * sampling_rate)
        if start_idx > len(raw_ch_df):
            break
        label_epoch = np.ones(duration_epoch, dtype=np.int) * label
        print('label_epoch:', label_epoch, len(label_epoch), 'idx:', idx, len(idx))
        
        assert len(label_epoch) == len(idx)//int(sampling_rate)
        
        labels.append(label_epoch)
        label_idx.append(idx)

        print("last_epoch_time:", last_epoch_time, "this_epoch_time:", this_epoch_time, "offset:", offset,
              "len(idx)", len(idx), "duration_sec:", duration_sec)
        
        last_epoch_time = this_epoch_time

    labels = np.hstack(labels)
    select_idx = np.arange(len(raw_ch_df))
    print("before intersect label: {}".format(select_idx.shape))
    label_idx = np.hstack(label_idx)
    print('labels.shape:', labels.shape, 'label_idx.shape:', label_idx.shape)
    select_idx = np.intersect1d(select_idx, label_idx)
    print("after intersect label: {}".format(select_idx.shape))

    # Remove movement and unknown stages if any
    raw_ch = raw_ch_df.values[select_idx]

    if len(raw_ch) % sampling_rate != 0:
        raise Exception("Something wrong")
    n_epochs = len(raw_ch) / sampling_rate
    print('n_epochs:', n_epochs)
    # Get epochs and their corresponding labels
    x = np.asarray(np.split(raw_ch, n_epochs)).astype(np.float32)
    y = labels.astype(np.int32)

    print('len(x):', len(x), 'len(y):', len(y))
    assert len(x) == len(y)

    # Select on sleep periods
    print("Data after selection: {}, {}".format(x.shape, y.shape))
    x = resample(x, 512*30, axis=1) if sampling_rate != 512 else x
    print('Before save {}, {}'.format(x.shape, y.shape))

    save_dict = {
        "x": x, 
        "y": y
    }
    np.savez(out_f, **save_dict)
    

