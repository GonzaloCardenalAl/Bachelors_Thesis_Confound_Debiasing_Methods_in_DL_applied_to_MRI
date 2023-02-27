#!/usr/bin/env python
# coding: utf-8


import os, sys
from glob import glob
from os.path import join

import pandas as pd
import numpy as np
import json
from tqdm import tqdm


from ukbb2020_dataloader import UKBB2020


dataset = UKBB2020()

for i, split in enumerate(['train', 'holdout']):
    df = dataset.get_metadata(predefined=[], cols=["1558-2.0", # Lifestyle - Freq
                                               "1628-2.0","2664-2.0","3859-2.0", # Lifestyle - prev. drinkers
                                               "20414-0.0","20403-0.0","20416-0.0",  # AUDIT
                                               '31-0.0', '21003-2.0' # demographic
                                              ], split=split, rename_cols=False)

    df = df.rename(columns= {'1558-2.0': 'freq',
                         '1628-2.0': 'Alcohol intake versus 10 years previously',
                         '2664-2.0': 'Reason for reducing amount of alcohol drunk',
                         '3859-2.0': 'Reason former drinker stopped drinking alcohol',
                         '20414-0.0': 'freqaudit',
                         '20403-0.0': 'amtaudit',
                         '20416-0.0': 'bingeaudit',
                         '31-0.0'   : 'Sex',
                         '21003-2.0': 'Age'})

    alc_cols = ['freq', 'freqaudit', 'amtaudit', 'bingeaudit']

    # change the dtype to float
    for c in alc_cols:
        for cat in df[c].cat.categories:
            # give invalid categories the same value (-1.0)
            if float(cat) < 0.0:
                df[c]= df[c].replace(cat, np.NaN)

        df[c] = df[c].astype(float)


    # apply the AUDIT skip rules to 'amtaudit' and 'bingeaudit'
    df.loc[df['freqaudit']==0, 'amtaudit'] = 1.0
    df.loc[df['freqaudit']==0, 'bingeaudit'] = 1.0

    def get_alc_lvl(row):
        # row = row.fillna(0)
        if (row['freq']<=3 and (row['amtaudit']+row['bingeaudit'])>=4):
            return 2 #True
        elif row[['freq', 'freqaudit']].isna().any() or ( # also exclude very high value in any of the variables
            row['amtaudit']>3 or row['bingeaudit']>3 or row['freq']<=1): # (row['freq']<=3 and (row['amtaudit']+row['bingeaudit'])<3) or
            return 1 #np.NaN
        else:
            return 0 #False

    df['highalc'] = df.apply(lambda row: get_alc_lvl(row), axis=1)

    dataset = UKBB2020()


    dataset.add_var_to_h5(df, 'highalc', typ='lbl', binarize=True, class0=0, class1=2, viz=False)
    dataset.add_var_to_h5(df, 'bingeaudit', typ='lbl', binarize=True, class0=1, class1=3, viz=False)
    dataset.add_var_to_h5(df, 'freq', y_colname='alcfreq', typ='lbl', norm=True, viz=False)
    dataset.add_var_to_h5(df, 'Sex', typ='conf', viz=False)
    dataset.add_var_to_h5(df, 'Age', typ='conf', viz=False)
    
    print(split, len(dataset.df_h5.dropna()), dataset.all_labels, dataset.all_confs)
    
    dataset.prepare_X(mri_col='path_T1_MNI')
        
    prefix = 'holdout' if split=='holdout' else ''

    dataset.save_h5(prefix, mri_kwargs={'z_factor':0.525})
