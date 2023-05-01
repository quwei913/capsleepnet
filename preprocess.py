#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import glob
import os
from datetime import datetime, timedelta
import math
from scipy.signal import resample


# In[2]:


STG_DIR = '/share/wequ0318/cap_clean/staging'
CAP_DIR = '/share/wequ0318/cap_clean/cap'
OUTPUT_DIR = '/share/wequ0318/cap_clean/processed'
if not os.path.exists(OUTPUT_DIR):
    os.mkdir(OUTPUT_DIR)


# In[3]:


def extend_time(start_time):
    start_time = start_time.replace('.', ':')
    s_t = datetime.strptime(start_time, "%H:%M:%S")
    ext_t = [(s_t + timedelta(seconds=d)).strftime("%H:%M:%S") for d in range(30)]
    return np.array(ext_t)


# In[5]:


for cap_f in glob.glob('{}/*.npz'.format(CAP_DIR)):
    f_name = os.path.basename(cap_f)
#     if f_name != 'n6_O2_A1.npz': continue
    s_f = glob.glob('{}/*{}'.format(STG_DIR, f_name))
    assert len(s_f) == 1
    stg_f = s_f[0]
    out_f = os.path.join(OUTPUT_DIR, os.path.basename(stg_f))
    print(out_f)
    with np.load(cap_f) as c_f:
        c_x = c_f['x']
        c_y = c_f['y']
        c_t = c_f['t']
        c_fs = c_f['fs']
        print(c_x.shape, c_y.shape, c_t.shape)
        with np.load(stg_f) as s_f:
            s_x = s_f['x']
            s_y = s_f['y']
            s_t = s_f['t']
            s_fs = s_f['fs']
            print(s_x.shape, s_y.shape, s_t.shape)
            assert c_fs == s_fs
            s_t = np.array([extend_time(xi) for xi in s_t]).reshape(-1,)
            inter = np.intersect1d(s_t, c_t, return_indices=True)
            s_x_common = np.take(s_x.reshape(-1, int(s_fs)), inter[1], axis=0)
            c_x_common = np.take(c_x.reshape(-1, int(c_fs)), inter[2], axis=0)
            assert np.array_equal(s_x_common, c_x_common)
            cap_labels = np.ones(s_t.shape, dtype=int) * 3
            np.put(cap_labels, inter[1], np.take(c_y.reshape(-1,), inter[2], axis=0))
            s_x = resample(s_x, 512*30, axis=1) if int(s_fs) != 512 else s_x
            print(s_x.shape, s_y.shape, cap_labels.shape)
            save_dict = {
                "x": s_x, 
                "c_y": cap_labels,
                "s_y": s_y
            }
            np.savez(out_f, **save_dict)
    


# In[ ]:




