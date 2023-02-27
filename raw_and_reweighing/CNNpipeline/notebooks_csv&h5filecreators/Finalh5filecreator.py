#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Final H5file creator
import os, sys
from glob import glob
from os.path import join

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(color_codes=False)
import json
from tqdm.notebook import tqdm
import h5py
from feature_engine.discretisation import DecisionTreeDiscretiser
import re

sys.path.insert(0, "../dataloaders/UKBB/")
from ukbb2020_dataloader import UKBB2020

sys.path.insert(0, "../helper/")
from plotGraphs import *


# In[2]:


dataset = UKBB2020()


# In[3]:


df = dataset.get_metadata(predefined=[], cols = ["31-0.0", #sex
                                                '54-2.0', #assestment center
                                                 '21003-2.0', #age
                                                 '1707-0.0',#handedness
                                                 '21000-0.0',#ethenic background binary 
                                                 '6142-2.0', #current employment status
                                                  '738-2.0',#avarage household total income after tax
                                                 "709-2.0", #Household size 
                                                 '2178-2.0', #Overall health rating
                                                 '26521-2.0', #total brain volume
                                                 "1558-2.0",#Alc int freq
                                                 "26414-0.0", #Education Score
                                                 "20016-2.0", #Fluid Intelligence
                                                 "1239-2.0", #Actual Tobacco smoking
                                                 "6350-2.0", #Duration to complete alphanumeric path trial 1
                                                 "6348-2.0", #Duration to complete alphanumeric path trial 2
                                                 "5832-2.0", #Which eye has hypermetrophia
                                                 "20414-0.0","20403-0.0","20416-0.0",  # AUDIT
                                                 "6148-2.0", #Eye problem/disorders
                                                ],split='train', rename_cols=True)  


# In[4]:


dataset.plot_metadata(df)


# In[5]:


df = dataset.get_metadata(predefined=[], cols = ["31-0.0", #sex
                                                '54-2.0', #assestment center
                                                 '21003-2.0', #age
                                                 '1707-0.0',#handedness
                                                 '21000-0.0',#ethenic background binary 
                                                 '6142-2.0', #current employment status
                                                '738-2.0',#avarage household total income after tax
                                                 "709-2.0", #Household size 
                                                 '2178-2.0', #Overall health rating
                                                 '26521-2.0', #total brain volume
                                                 "1558-2.0",#Alc int freq
                                                 "26414-0.0", #Education Score
                                                 "20016-2.0", #Fluid Intelligence
                                                 "1239-2.0", #Actual Tobacco smoking
                                                 "6350-2.0", #Duration to complete alphanumeric path trial 1
                                                 "6348-2.0", #Duration to complete alphanumeric path trial 2
                                                 "5832-2.0", #Which eye has hypermetrophia
                                                 "20414-0.0","20403-0.0","20416-0.0",  # AUDIT
                                                 "6148-2.0", #Eye problem/disorders
                                                ],split='train', rename_cols=False) 


# In[6]:


#Adding central nervous system diseases as a column
dficd = dataset.get_metadata(predefined=['icd'], cols=[], print_cols=False, split='train', rename_cols=False)
df['disease_of_central_nervous_system'] =  dficd.apply(lambda row: row.astype(str).str.contains('G(?!50|51|52|53|54|55|56|57|70|71|72|73)[0-9][0-9]').any(), axis=1)
df['disease_of_central_nervous_system'] = df['disease_of_central_nervous_system'].astype(float)


# In[7]:


#Current employment status binary to retired or not
df['6142-2.0'] = df['6142-2.0'].replace(['-3.0','-1.0'],np.nan)
df['6142-2.0'] = df['6142-2.0'].apply(lambda x: 1.0 if x == '2.0' else 0.0) #Retired is 1.0

#Household size "709-2.0" eliminate outliers
df = df[df["709-2.0"] < 7.0]
df = df[df["709-2.0"] > 0.0]

#Index of multiple depravation .astype(float)
#df['26410-0.0'] = df['26410-0.0'].apply(lambda x: 1.0 if x > 42 else 0.0)

#Overall health rating turn missing values to np.nan 1	Excellent,2	Good,3	Fair,4	Poor
df['2178-2.0'] = df['2178-2.0'].replace(['-3.0','-1.0'],np.nan)

#Avarage income turn missing values to np.nan
df['738-2.0'] = df['738-2.0'].replace(['-3.0','-1.0'],np.nan)

#ethnic background binary British or not british
df['21000-0.0'] = df['21000-0.0'].replace('-3.0',np.nan)
df['21000-0.0'] = df['21000-0.0'].apply(lambda x: 1.0 if x == '1001.0' else 0.0) #british is 1.0
  
 #handedness turn: no available to np.nan and turn binary to right handed or not right handed
df['1707-0.0']= df['1707-0.0'].replace('-3.0',np.nan)
df['1707-0.0'] = df['1707-0.0'].apply(lambda x: 1.0 if x == '1.0' else 0.0)

  #eyeproblems #Yes or no
df['6148-2.0']= df['6148-2.0'].replace(['1.0','2.0','3.0','4.0','5.0','6.0'],'1.0')
df['6148-2.0']= df['6148-2.0'].replace(['-7.0'],'0.0')
df['6148-2.0']= df['6148-2.0'].replace(['-3.0','-1.0'],np.nan)
 
  #hypermetrophia
df['5832-2.0']= df['5832-2.0'].map({np.nan:'0.0',
                                   '1.0' :'1.0',
                                    '2.0':'1.0',
                                    '3.0':'1.0'})
df['5832-2.0'] = pd.Categorical(df['5832-2.0'])
df['5832-2.0']= df['5832-2.0'].cat.add_categories('0.0') #so you need to run this with one error to get the fill na working, no clue why
df['5832-2.0']= df['5832-2.0'].fillna('0.0') #no hypermetrophia

#Current_tobacco_smoking
df['1239-2.0']= df['1239-2.0'].replace(['1.0','2.0'],'1.0') #smokers
df['1239-2.0']= df['1239-2.0'].replace(['-3.0'],np.nan)

#Trail making avg duration
df["trail_making_avg_duration"]=(df["6350-2.0"] + df["6348-2.0"])/2
df = df[df["trail_making_avg_duration"] < 1200.0] #to remove outliers
df = df[df["trail_making_avg_duration"] > 1.0]


# In[9]:


df = df.rename(columns={'31-0.0' : 'Sex',
                        '21003-2.0' : 'Age',
                        "1558-2.0" :'Alc_int_freq' ,
                        '26521-2.0': 'Total_brain_volume',
                        '54-2.0': 'Site',
                        "26414-0.0":'Education_Score',
                        "20016-2.0" :'Fluid_Intelligence',
                        "6350-2.0" : 'Duration to complete alphanumeric path trial 2',
                        "6348-2.0" : 'Duration to complete alphanumeric path trial 1',
                        "5832-2.0" : 'Hypermetrophia',
                        "6148-2.0" : 'Eye_disorders',
                        '20414-0.0': 'freqaudit', 
                         '20403-0.0': 'amtaudit', 
                         '20416-0.0': 'bingeaudit',
                        '1707-0.0' : 'handedness',
                         '21000-0.0' : 'ethenic_background_binary' ,
                         '6142-2.0' : 'employment_status',
                          '738-2.0' : 'avarage_household_income',
                         "709-2.0" : 'number_of_household_integrants' ,
                         '2178-2.0' : 'Overall_health_rating',
                        "1239-2.0" : 'Current_tobacco_smoking',
                       })


# In[10]:


df = df.dropna()


# In[11]:


df


# In[12]:


#discretasing age
discage = DecisionTreeDiscretiser(cv=5, regression=True, variables = ['Age'])
discage.fit(df, df['Age'])
df_transformedage= discage.transform(df)

#discretasing brain volume
discbrainvol = DecisionTreeDiscretiser(cv=5, regression=True, variables = ['Total_brain_volume'])
discbrainvol.fit(df, df['Total_brain_volume'])
df_transformedbrain= discbrainvol.transform(df)

#discretasing trail making
disctrail = DecisionTreeDiscretiser(cv=5, regression=True, variables = ["trail_making_avg_duration"])
disctrail.fit(df, df["trail_making_avg_duration"])
df_transformedtrail= disctrail.transform(df)

df_transformedage['Age'].value_counts()


# In[13]:


df


# In[14]:


#adding discretized variables
df['Age_binned'] = df_transformedage['Age']
df['Total_brain_volume_binned'] = df_transformedbrain['Total_brain_volume']
df["trail_making_avg_duration_binned"] = df_transformedtrail["trail_making_avg_duration"]


# In[15]:


# change the dtype to float
for c in df.columns:
    if df[c].dtype.name == 'category':
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
    if (row['Alc_int_freq']<=3 and (row['amtaudit']+row['bingeaudit'])>=4):
        return 2 #True
    elif row[['Alc_int_freq', 'freqaudit']].isna().any() or ( # also exclude very high value in any of the variables
        row['amtaudit']>3 or row['bingeaudit']>3 or row['Alc_int_freq']<=1): # (row['freq']<=3 and (row['amtaudit']+row['bingeaudit'])<3) or 
        return 1 #np.NaN
    else:
        return 0 #False

    #get highalc class
df['Highalc'] = df.apply(lambda row: get_alc_lvl(row), axis=1)


# In[16]:


dataset.plot_metadata(df)


# In[17]:


dataset.add_var_to_h5(df, 'Sex', typ='lbl', viz=False)
dataset.add_var_to_h5(df, 'Age', typ='lbl', viz=False)
dataset.add_var_to_h5(df, 'Alc_int_freq', typ='lbl', viz=False)
dataset.add_var_to_h5(df, 'Total_brain_volume', typ='lbl', viz=False)
dataset.add_var_to_h5(df, 'Site', typ='lbl', viz=False)
dataset.add_var_to_h5(df, 'Education_Score', typ='lbl', viz=False)
dataset.add_var_to_h5(df, 'Fluid_Intelligence', typ='lbl', viz=False)
dataset.add_var_to_h5(df, 'Current_tobacco_smoking', typ='lbl', viz=False)
dataset.add_var_to_h5(df, 'Hypermetrophia', typ='lbl', viz=False)
dataset.add_var_to_h5(df, 'Eye_disorders', typ='lbl', viz=False)
dataset.add_var_to_h5(df, 'trail_making_avg_duration', typ='lbl', viz=False)
dataset.add_var_to_h5(df, 'trail_making_avg_duration_binned', typ='lbl', viz=False)
dataset.add_var_to_h5(df, 'Age_binned', typ='lbl', viz=False)
dataset.add_var_to_h5(df, 'Total_brain_volume_binned', typ='lbl', viz=False)
dataset.add_var_to_h5(df, 'Highalc', typ='lbl', binarize=True, class0=0, class1=2, viz=False)
dataset.add_var_to_h5(df, 'handedness', typ='lbl', viz=False)
dataset.add_var_to_h5(df, 'ethenic_background_binary', typ='lbl', viz=False)
dataset.add_var_to_h5(df, 'employment_status', typ='lbl', viz=False)
dataset.add_var_to_h5(df, 'avarage_household_income', typ='lbl', viz=False)
dataset.add_var_to_h5(df, 'number_of_household_integrants', typ='lbl', viz=False)
dataset.add_var_to_h5(df, 'Overall_health_rating', typ='lbl', viz=False)
dataset.add_var_to_h5(df, 'disease_of_central_nervous_system', typ='lbl', viz=False)


# In[18]:


dataset.prepare_X(mri_col='path_T1_MNI')


# In[19]:


dataset.save_h5(filename_prefix="confounds_tasks_8364k_train", mri_kwargs={'z_factor':(0.525)})


# In[28]:


#We create holdout in the same file because we need the decision trees for discretizating for the holdout
dataset = UKBB2020()


# In[29]:


df = dataset.get_metadata(predefined=[], cols = ["31-0.0", #sex
                                                '54-2.0', #assestment center
                                                 '21003-2.0', #age
                                                 '1707-0.0',#handedness
                                                 '21000-0.0',#ethenic background binary 
                                                 '6142-2.0', #current employment status
                                                '738-2.0',#avarage household total income after tax
                                                 "709-2.0", #Household size 
                                                 '2178-2.0', #Overall health rating
                                                 '26521-2.0', #total brain volume
                                                 "1558-2.0",#Alc int freq
                                                 "26414-0.0", #Education Score
                                                 "20016-2.0", #Fluid Intelligence
                                                 "1239-2.0", #Actual Tobacco smoking
                                                 "6350-2.0", #Duration to complete alphanumeric path trial 1
                                                 "6348-2.0", #Duration to complete alphanumeric path trial 2
                                                 "5832-2.0", #Which eye has hypermetrophia
                                                 "20414-0.0","20403-0.0","20416-0.0",  # AUDIT
                                                 "6148-2.0", #Eye problem/disorders
                                                ],split='holdout', rename_cols=False) 


# In[30]:


#Adding central nervous system diseases as a column
dficd = dataset.get_metadata(predefined=['icd'], cols=[], print_cols=False, split='holdout', rename_cols=False)
df['disease_of_central_nervous_system'] =  dficd.apply(lambda row: row.astype(str).str.contains('G(?!50|51|52|53|54|55|56|57|70|71|72|73)[0-9][0-9]').any(), axis=1)
df['disease_of_central_nervous_system'] = df['disease_of_central_nervous_system'].astype(float)


# In[31]:


#Current employment status binary to retired or not
df['6142-2.0'] = df['6142-2.0'].replace(['-3.0','-1.0'],np.nan)
df['6142-2.0'] = df['6142-2.0'].apply(lambda x: 1.0 if x == '2.0' else 0.0) #Retired is 1.0

#Household size "709-2.0" eliminate outliers
df = df[df["709-2.0"] < 7.0]
df = df[df["709-2.0"] > 0.0]

#Index of multiple depravation .astype(float)
#df['26410-0.0'] = df['26410-0.0'].apply(lambda x: 1.0 if x > 42 else 0.0)

#Overall health rating turn missing values to np.nan 1	Excellent,2	Good,3	Fair,4	Poor
df['2178-2.0'] = df['2178-2.0'].replace(['-3.0','-1.0'],np.nan)

#Avarage income turn missing values to np.nan
df['738-2.0'] = df['738-2.0'].replace(['-3.0','-1.0'],np.nan)

#ethnic background binary British or not british
df['21000-0.0'] = df['21000-0.0'].replace('-3.0',np.nan)
df['21000-0.0'] = df['21000-0.0'].apply(lambda x: 1.0 if x == '1001.0' else 0.0) #british is 1.0
  
 #handedness turn: no available to np.nan and turn binary to right handed or not right handed
df['1707-0.0']= df['1707-0.0'].replace('-3.0',np.nan)
df['1707-0.0'] = df['1707-0.0'].apply(lambda x: 1.0 if x == '1.0' else 0.0)

  #eyeproblems #Yes or no
df['6148-2.0']= df['6148-2.0'].replace(['1.0','2.0','3.0','4.0','5.0','6.0'],'1.0')
df['6148-2.0']= df['6148-2.0'].replace(['-7.0'],'0.0')
df['6148-2.0']= df['6148-2.0'].replace(['-3.0','-1.0'],np.nan)
 
  #hypermetrophia
df['5832-2.0']= df['5832-2.0'].map({np.nan:'0.0',
                                   '1.0' :'1.0',
                                    '2.0':'1.0',
                                    '3.0':'1.0'})
df['5832-2.0'] = pd.Categorical(df['5832-2.0'])
df['5832-2.0']= df['5832-2.0'].cat.add_categories('0.0') #so you need to run this with one error to get the fill na working, no clue why
df['5832-2.0']= df['5832-2.0'].fillna('0.0') #no hypermetrophia

#Current_tobacco_smoking
df['1239-2.0']= df['1239-2.0'].replace(['1.0','2.0'],'1.0') #smokers
df['1239-2.0']= df['1239-2.0'].replace(['-3.0'],np.nan)

#Trail making avg duration
df["trail_making_avg_duration"]=(df["6350-2.0"] + df["6348-2.0"])/2
df = df[df["trail_making_avg_duration"] < 1200.0] #to remove outliers
df = df[df["trail_making_avg_duration"] > 1.0]


# In[32]:


df = df.rename(columns={'31-0.0' : 'Sex',
                        '21003-2.0' : 'Age',
                        "1558-2.0" :'Alc_int_freq' ,
                        '26521-2.0': 'Total_brain_volume',
                        '54-2.0': 'Site',
                        "26414-0.0":'Education_Score',
                        "20016-2.0" :'Fluid_Intelligence',
                        "6350-2.0" : 'Duration to complete alphanumeric path trial 2',
                        "6348-2.0" : 'Duration to complete alphanumeric path trial 1',
                        "5832-2.0" : 'Hypermetrophia',
                        "6148-2.0" : 'Eye_disorders',
                        '20414-0.0': 'freqaudit', 
                         '20403-0.0': 'amtaudit', 
                         '20416-0.0': 'bingeaudit',
                        '1707-0.0' : 'handedness',
                         '21000-0.0' : 'ethenic_background_binary' ,
                         '6142-2.0' : 'employment_status',
                          '738-2.0' : 'avarage_household_income',
                         "709-2.0" : 'number_of_household_integrants' ,
                         '2178-2.0' : 'Overall_health_rating',
                         '26410-0.0': 'index_of_multiple_depravation',
                        "1239-2.0" : 'Current_tobacco_smoking',
                       })


# In[33]:


df = df.dropna()


# In[34]:


#discretasing age
df_transformedage= discage.transform(df)

#discretasing brain volume
df_transformedbrain= discbrainvol.transform(df)

#discretasing trail making
df_transformedtrail= disctrail.transform(df)


# In[35]:


#adding discretized variables
df['Age_binned'] = df_transformedage['Age']
df['Total_brain_volume_binned'] = df_transformedbrain['Total_brain_volume']
df["trail_making_avg_duration_binned"] = df_transformedtrail["trail_making_avg_duration"]


# In[36]:


# change the dtype to float
for c in df.columns:
    if df[c].dtype.name == 'category':
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
    if (row['Alc_int_freq']<=3 and (row['amtaudit']+row['bingeaudit'])>=4):
        return 2 #True
    elif row[['Alc_int_freq', 'freqaudit']].isna().any() or ( # also exclude very high value in any of the variables
        row['amtaudit']>3 or row['bingeaudit']>3 or row['Alc_int_freq']<=1): # (row['freq']<=3 and (row['amtaudit']+row['bingeaudit'])<3) or 
        return 1 #np.NaN
    else:
        return 0 #False

    #get highalc class
df['Highalc'] = df.apply(lambda row: get_alc_lvl(row), axis=1)


# In[39]:


dataset.add_var_to_h5(df, 'Sex', typ='lbl', viz=False)
dataset.add_var_to_h5(df, 'Age', typ='lbl', viz=False)
dataset.add_var_to_h5(df, 'Alc_int_freq', typ='lbl', viz=False)
dataset.add_var_to_h5(df, 'Total_brain_volume', typ='lbl', viz=False)
dataset.add_var_to_h5(df, 'Site', typ='lbl', viz=False)
dataset.add_var_to_h5(df, 'Education_Score', typ='lbl', viz=False)
dataset.add_var_to_h5(df, 'Fluid_Intelligence', typ='lbl', viz=False)
dataset.add_var_to_h5(df, 'Current_tobacco_smoking', typ='lbl', viz=False)
dataset.add_var_to_h5(df, 'Hypermetrophia', typ='lbl', viz=False)
dataset.add_var_to_h5(df, 'Eye_disorders', typ='lbl', viz=False)
dataset.add_var_to_h5(df, 'trail_making_avg_duration', typ='lbl', viz=False)
dataset.add_var_to_h5(df, 'trail_making_avg_duration_binned', typ='lbl', viz=False)
dataset.add_var_to_h5(df, 'Age_binned', typ='lbl', viz=False)
dataset.add_var_to_h5(df, 'Total_brain_volume_binned', typ='lbl', viz=False)
dataset.add_var_to_h5(df, 'Highalc', typ='lbl', binarize=True, class0=0, class1=2, viz=False)
dataset.add_var_to_h5(df, 'handedness', typ='lbl', viz=False)
dataset.add_var_to_h5(df, 'ethenic_background_binary', typ='lbl', viz=False)
dataset.add_var_to_h5(df, 'employment_status', typ='lbl', viz=False)
dataset.add_var_to_h5(df, 'avarage_household_income', typ='lbl', viz=False)
dataset.add_var_to_h5(df, 'number_of_household_integrants', typ='lbl', viz=False)
dataset.add_var_to_h5(df, 'Overall_health_rating', typ='lbl', viz=False)
dataset.add_var_to_h5(df, 'disease_of_central_nervous_system', typ='lbl', viz=False)


# In[40]:


dataset.prepare_X(mri_col='path_T1_MNI')


# In[ ]:


dataset.save_h5(filename_prefix="confounds_tasks_3460k_holdout", mri_kwargs={'z_factor':(0.525)})






