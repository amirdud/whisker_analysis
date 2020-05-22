import numpy as np
import glob
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.io as sio

import whisker_funs as wf
import mouse_funs as mf
import vid_funs as vf
import util_funs as uf

work_dir = 'C:\\Users\\amird\\OneDrive\\Documents\\Berkeley\\Amir\\' # global variable
mouse = '6994_210_000'
mouse_short = mouse[0:4]

# whisker names
whisker_names = mf.get_whisker_names(mouse)
dic_mapping = mf.get_whisker_stim_mapping(mouse,names=whisker_names)
xlabels = [dic_mapping[key] for key in dic_mapping]

# load table
file_name_load = work_dir + mouse_short + '\\' + mouse + '_exp_AnalysisInfo.csv'
df = pd.read_csv(file_name_load)

# =========== Analysis ===============
# number of trials
df_counts = df['StimID'].value_counts().sort_index()
g = sns.countplot(x='StimID', data=df,color=(22/255,67/255,140/255))
g.set_xticklabels(xlabels,rotation=90)
g.set_ylabel('# Trials')
g.set_xlabel('')
g.set_title('n = ' + str(df_counts.sum()))
sns.despine()
plt.show()


# number of contact frames
w = 'C1'
ind = whisker_names.index(w)
column = 'num_touches_' + str(ind+1)
colors = uf.get_colors(5)
color_1 = uf.colors_255_to_1(colors)[ind]

stimIDs_dic = wf.get_stimIDs_for_whisker(dic_mapping,w)
stimIDs = [int(key) for key in stimIDs_dic]
df_spec_w = df[df['StimID'].isin(stimIDs)]

g = sns.catplot(x="StimID", y=column,kind="point",
            ci=68, data=df_spec_w,color=color_1) # 68: s.e.m

xlabels = [stimIDs_dic[key] for key in stimIDs_dic]
g.set_xticklabels(xlabels,rotation=90)
g.set(xlabel='', ylabel='# Contact Frames')
plt.show()

# mean angle for whisker
columns = ['mean_angle_1','mean_angle_2',
           'mean_angle_3','mean_angle_4','mean_angle_5']

# df_Tracing_contacts = df_Tracing.loc[:,columns]

i = 5
column = 'mean_angle_' + str(i)
color_1 = uf.colors_255_to_1(colors)[i-1]

g = sns.catplot(x="StimID", y=column,kind="point",
            ci=68, data=df,color=color_1) # 68: s.e.m
xlabels = [dic_mapping[key] for key in dic_mapping]
g.set_xticklabels(xlabels,rotation=90)
g.set(xlabel='', ylabel=column)
plt.show()

# range for whisker
# df_Tracing_contacts = df_Tracing.loc[:,columns]
w = 'C1'
ind = whisker_names.index(w)
column = 'rng_angle_' + str(ind+1)
colors = uf.get_colors(5)
color_1 = uf.colors_255_to_1(colors)[ind]

stimIDs_dic = wf.get_stimIDs_for_whisker(dic_mapping,w)
stimIDs = [int(key) for key in stimIDs_dic]
df_spec_w = df[df['StimID'].isin(stimIDs)]

g = sns.catplot(x="StimID", y=column,
            ci=68, data=df_spec_w,color=color_1)
xlabels = [stimIDs_dic[key] for key in stimIDs_dic]
g.set_xticklabels(xlabels,rotation=90)
g.set(xlabel='', ylabel=column)
plt.show()

# angle
stimIDs_dic = wf.get_stimIDs_for_whisker(dic_mapping,w)
stimIDs = [int(key) for key in stimIDs_dic]
df_spec_w = df[df['StimID'].isin(stimIDs)]

g = sns.catplot(x="StimID", y=column,kind='point',ci=68, data=df_spec_w,
                color=(22/255,67/255,140/255))

xlabels = [stimIDs_dic[key] for key in stimIDs_dic]
g.set_xticklabels(xlabels,rotation=90)
g.set(xlabel='', ylabel='Mean Angle (degree)')
plt.show()

# range angle
columns = ['rng_angle_1','rng_angle_2',
           'rng_angle_3','rng_angle_4','rng_angle_5']

# df_Tracing_contacts = df_Tracing.loc[:,columns]

w = 'gamma'
ind = whisker_names.index(w)
column = 'rng_angle_' + str(ind+1)
colors = uf.get_colors(5)
color_1 = uf.colors_255_to_1(colors)[ind]

stimIDs_dic = wf.get_stimIDs_for_whisker(dic_mapping,w)
stimIDs = [int(key) for key in stimIDs_dic]
df_spec_w = df[df['StimID'].isin(stimIDs)]

g = sns.catplot(x="StimID", y=column,
            ci=68, data=df_spec_w,color=color_1)
xlabels = [stimIDs_dic[key] for key in stimIDs_dic]
g.set_xticklabels(xlabels,rotation=90)
g.set(xlabel='', ylabel=column)
plt.show()


# curvature
w = 'D1'
ind = whisker_names.index(w)
column = 'mean_deltaKappa_' + str(ind+1)
colors = uf.get_colors(5)
color_1 = uf.colors_255_to_1(colors)[ind]

stimIDs_dic = wf.get_stimIDs_for_whisker(dic_mapping,w)
stimIDs = [int(key) for key in stimIDs_dic]
df_spec_w = df[df['StimID'].isin(stimIDs)]

g = sns.catplot(x="StimID", y=column,
            ci=68, data=df_spec_w,color=color_1)

xlabels = [stimIDs_dic[key] for key in stimIDs_dic]
g.set_xticklabels(xlabels,rotation=90)
g.set(xlabel='', ylabel=column)
plt.show()


# find all trials in specific condition
stimID = [1] # choose one stimID
whisker_comb = dic_mapping[str(stimID[0])]
df_spec_w = df[df['StimID'].isin(stimID)]
trials = list(df_spec_w['TrialIndex'])

# find specific trials according to condition
w = 'C1'
column = "mean_deltaKappa_1"
stimID = [19] # choose one stimID
whisker_comb = dic_mapping[str(stimID[0])]
df_spec_w = df[df['StimID'].isin(stimID)]
condition = df_spec_w[column] >0.05
trials = list(df_spec_w['TrialIndex'].loc[condition])

