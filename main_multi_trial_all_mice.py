import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from collections import Counter
from itertools import combinations

import seaborn as sns
from matplotlib import rc
import numpy as np

import util_funs as uf
import mouse_funs as mf
import whisker_funs as wf
# matplotlib.use('Qt5Agg')

work_dir = 'C:\\Users\\amird\\OneDrive\\Documents\\Berkeley\\Amir\\'

# ============ combine dataframes from all 3 mice ===============
mouse_1 = '7737_291_000'
mouse_short_1 = mouse_1[0:4]
csv_name_load_1 = work_dir + mouse_short_1 + '\\' + mouse_1 + '_exp_AnalysisInfo_14.csv'
df_1 = pd.read_csv(csv_name_load_1)
df_1['dataframe'] = mouse_short_1

# add contact duration parameter
msec_in_sec = 1000
stim_frames = 300-35
columns_for_contact = ['num_contacts_1','num_contacts_2','num_contacts_3','num_contacts_4','num_contacts_5']
columns_for_contact_duration = ['contact_duration_' + name[-1] for name in columns_for_contact]
for i,column in enumerate(columns_for_contact):
    df_1[columns_for_contact_duration[i]] =df_1[column]/stim_frames*msec_in_sec

# add which whiskers where stimulated
new_cols = ['1','2','3','4','5']
df_1 = mf.add_columns_df(df_1, new_cols)
piston_comb = mf.get_piston_combinations(mouse_1)

for key in piston_comb:
    for piston in piston_comb[key]:
        df_1.loc[df_1.loc[df_1['StimID']==int(key)].index,str(piston)] = True

# unify df_s:
# check all whisker names are the same and in the same order
# whisker_names_1 = mf.get_whisker_names(mouse_1)
# print(whisker_names_1)
# mouse_2 = '6994_210_000'
# whisker_names_2 = mf.get_whisker_names(mouse_2)
# print(whisker_names_2)
# mouse_3 = '7736_300_000'
# whisker_names_3 = mf.get_whisker_names(mouse_3)
# print(whisker_names_3)

# if not:
# 1. change problematic whisker name if needed: (refer to D2 as D1 in 7737)
whisker_names_1 = mf.get_whisker_names(mouse_1)
whisker_names_1[2]='D1'

# 2. match stimID to the others: (df_1 is matched to the others)
mouse_2 = '6994_210_000'
whisker_names_2 = mf.get_whisker_names(mouse_2)
dic_mapping_1 = mf.get_whisker_stim_mapping(mouse_1,names=whisker_names_1)
dic_mapping_2 = mf.get_whisker_stim_mapping(mouse_2,names=whisker_names_2)

# find all stims in df_1
for stim_1 in dic_mapping_1:

    # find matching stimIDs
    if Counter(dic_mapping_1[stim_1]) == Counter(dic_mapping_2[stim_1]):
        df_1.loc[df_1['StimID'] == int(stim_1), 'new_StimID'] = int(stim_1)

    # find unmatching stimIDs
    else:
        # find the true stimID
        for stim_2 in dic_mapping_2:
            if Counter(dic_mapping_1[stim_1]) == Counter(dic_mapping_2[stim_2]):
                new_stim = int(stim_2)
                df_1.loc[df_1['StimID']==int(stim_1),'new_StimID'] = new_stim
                break


df_1['StimID'] = df_1['new_StimID'].copy()
df_1['StimID'] = df_1['StimID'].astype(np.int64)
df_1 = df_1.drop(['new_StimID'],axis=1)

# change column names:
start_col = -84
col_names = list(df_1.keys())[start_col:]
col_names_new = []
for i,col in enumerate(col_names):
    last_char = col[-1] #numbers 1-5
    if last_char.isdigit():
        number = int(last_char)
        index = number - 1
        new_col = col[:-1] + whisker_names_1[number-1] # indexing 0-4
        col_names_new.append(new_col)
        df_1.rename(columns={col: new_col}, inplace=True)

mouse_2 = '6994_210_000'
mouse_short_2 = mouse_2[0:4]
csv_name_load_2 = work_dir + mouse_short_2 + '\\' + mouse_2 + '_exp_AnalysisInfo_14.csv'
df_2 = pd.read_csv(csv_name_load_2)
df_2['dataframe'] = mouse_short_2

whisker_names = mf.get_whisker_names(mouse_2)

# add contact duration parameter
msec_in_sec = 1000
stim_frames = 300-35
columns_for_contact = ['num_contacts_1','num_contacts_2','num_contacts_3','num_contacts_4','num_contacts_5']
columns_for_contact_duration = ['contact_duration_' + name[-1] for name in columns_for_contact]
for i,column in enumerate(columns_for_contact):
    df_2[columns_for_contact_duration[i]] =df_2[column]/stim_frames*msec_in_sec

piston_comb = mf.get_piston_combinations(mouse_2)

for key in piston_comb:
    for piston in piston_comb[key]:
        df_2.loc[df_2.loc[df_2['StimID']==int(key)].index,str(piston)] = True

# change column names:
start_col = -84
col_names = list(df_2.keys())[start_col:]
col_names_new = []
for i,col in enumerate(col_names):
    last_char = col[-1] #numbers 1-5
    if last_char.isdigit():
        number = int(last_char)
        index = number - 1
        new_col = col[:-1] + whisker_names[number-1] # indexing 0-4
        col_names_new.append(new_col)
        df_2.rename(columns={col: new_col}, inplace=True)

mouse_3 = '7736_300_000'
mouse_short_3 = mouse_3[0:4]
csv_name_load_3 = work_dir + mouse_short_3 + '\\' + mouse_3 + '_exp_AnalysisInfo_14.csv'
df_3 = pd.read_csv(csv_name_load_3)
df_3['dataframe'] = mouse_short_3

whisker_names = mf.get_whisker_names(mouse_3)

# add contact duration parameter
msec_in_sec = 1000
stim_frames = 300-35
columns_for_contact = ['num_contacts_1','num_contacts_2','num_contacts_3','num_contacts_4','num_contacts_5']
columns_for_contact_duration = ['contact_duration_' + name[-1] for name in columns_for_contact]
for i,column in enumerate(columns_for_contact):
    df_3[columns_for_contact_duration[i]] =df_3[column]/stim_frames*msec_in_sec

piston_comb = mf.get_piston_combinations(mouse_3)

for key in piston_comb:
    for piston in piston_comb[key]:
        df_3.loc[df_3.loc[df_3['StimID']==int(key)].index,str(piston)] = True

# change column names:
start_col = -84
col_names = list(df_3.keys())[start_col:]
col_names_new = []
for i,col in enumerate(col_names):
    last_char = col[-1] #numbers 1-5
    if last_char.isdigit():
        number = int(last_char)
        index = number - 1
        new_col = col[:-1] + whisker_names[number-1] # indexing 0-4
        col_names_new.append(new_col)
        df_3.rename(columns={col: new_col}, inplace=True)

# combine all:
# dfs = [df_1,df_3]
dfs = [df_1,df_2,df_3]
df = pd.concat(dfs, keys=[mouse_short_1, mouse_short_2,mouse_short_3],sort=False,ignore_index=True)
df = df.loc[df['evan_trials_inliers']==True]

whisker_names = mf.get_whisker_names(mouse_2)
dic_mapping = mf.get_whisker_stim_mapping(mouse_2,names=whisker_names)

# make sure there is no nan where it shouldn't be:
# sum(pd.notna(df.loc[df['StimID']==3,'num_contacts_B1']))


# =========== Analyze =============
# group: zscore by mouse

cols_dfg = ['StimID'] + list(df.columns[-84:])
df_reduced = df[cols_dfg].copy()
df_reduced_z = df_reduced.copy()

remove_z_cols = ['StimID','C1','C2','D1','B1','gamma','dataframe','evan_trials_inliers', 'evan_trials','num_whiskers']
cols_run_for_z = [x for x in cols_dfg if x not in remove_z_cols]

mice_names = [mouse_short_1, mouse_short_2, mouse_short_3]
for col in cols_run_for_z:
    for name in mice_names:
        df_reduced_z.loc[df_reduced['dataframe']==name,col] = (df_reduced.loc[df_reduced['dataframe']==name,col]
                                                               - df_reduced.loc[df_reduced['dataframe']==name,col].mean())/df_reduced.loc[df_reduced['dataframe']==name,col].std(ddof=0)
df_reduced_z['dataframe'] = df_reduced['dataframe'].copy()

# split
groups_dfg = df_reduced.groupby(['StimID', 'dataframe'], as_index=False)
groups_dfg.groups

# apply
dfg = groups_dfg.agg(np.mean)

# =========== scatterplot comparing mix vs single =============
# type,ii = 'mean_angle',0
# type,ii = 'rng_angle',1
# type,ii = 'contact_duration',2
# type,ii = 'num_touches',3
type,ii = 'mean_deltaKappa_part1_external',4


whisker_names = mf.get_whisker_names(mouse_2,asdic = True)

whisker_numbers_to_use = [1,2,3,4,5]  # change this to whisker names to use! B1,D1 can switch places
whiskers_to_use = {key:whisker_names[key] for key in whisker_numbers_to_use}

# how many points for each mouse:
n_combs = 0
for i in range(2,len(whisker_numbers_to_use)+1):
    comb = list(combinations(whisker_numbers_to_use, i))
    n_combs = n_combs + len(comb)

n_mice = len(mice_names)
par_array = np.empty((n_combs,2,n_mice))

# generate stimulation pairs
pairs_list = mf.get_stim_pairs(dic_mapping)
pairs_list_to_use = []
for pair in pairs_list:
    single_stims_names = dic_mapping[str(pair[0])]
    single_stims_numbers = pair[1]
    if set(single_stims_numbers).issubset(whisker_numbers_to_use):
        pairs_list_to_use.append(pair)

# get mean parameter values from each mouse and each stimuli pairs
for i,name in enumerate(mice_names):
    for j,pair in enumerate(pairs_list_to_use):
        single_stims_names = dic_mapping[str(pair[0])]
        single_stims_numbers = pair[1]

        if set(single_stims_numbers).issubset(whisker_numbers_to_use):
            # columns to use
            columns = []
            for stim_name in single_stims_names:
                columns.append(type + '_' + stim_name)

            vals_single = []
            vals_mix = []

            # get values from relevant columns
            for k,col in enumerate(columns):
                single_stim_num_k = single_stims_numbers[k]
                val_single_i = np.array(dfg.loc[(dfg['StimID'] == single_stim_num_k) & (dfg['dataframe'] == name), col])[0]
                val_mix_i = np.array(dfg.loc[(dfg['StimID'] == pair[0]) & (dfg['dataframe'] == name), col])[0]

                vals_single.append(val_single_i)
                vals_mix.append(val_mix_i)

            vals_single = np.array(vals_single)
            vals_mix = np.array(vals_mix)

            mean_vals_single = np.mean(vals_single)
            mean_vals_mix = np.mean(vals_mix)

            point = np.array([mean_vals_single,mean_vals_mix])
            par_array[j, :, i] = point

xy = np.concatenate( par_array, axis=1 ).T

import scipy.stats as stats
# p_vals_trel = np.zeros(5)

trel = stats.ttest_rel(xy[:,0],xy[:,1])
trel.pvalue


fig,ax = plt.subplots()
for i in [1,2,0]:
    x = par_array[:,0,i] # single
    y = par_array[:,1,i] # mix
    ax.plot(x,y,'.',markersize = 20,label = mice_names[i],color='k')

# add line
min_lim = np.min(par_array)
max_lim = np.max(par_array)
lims = [min_lim - (max_lim-min_lim)/20,max_lim + (max_lim-min_lim)/20]
ax.plot(lims,lims,'k')
ax.set_xlim(lims)
ax.set_ylim(lims)
ax.set_xlabel('Mean Single',fontsize=20)
ax.set_ylabel('Mean Mix',fontsize=20)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.xaxis.set_ticks_position('bottom')
ax.yaxis.set_ticks_position('left')
ax.tick_params(labelsize=20)
# ax.xaxis.set_ticks([-0.25,0,0.25,0.5])
# ax.yaxis.set_ticks([-0.25,0,0.25,0.5])

plt.title(type,fontsize=20)

fig.show()
#
# fig_name = work_dir + 'Figs\\' + 'SupraLinearity_scatterplot_black_' + type + '.pdf'
# fig.savefig(fig_name, bbox_inches='tight')


# =========== average across conditions =============
whisker_names = mf.get_whisker_names(mouse_2)
type = 'raw'
w = 'C1'
columns = ['num_touches_' + w, 'mean_angle_' + w, 'mean_deltaKappa_part1_external_' + w, 'rng_angle_' + w, 'contact_duration_' + w]

for column in columns:
    ind = whisker_names.index(w)
    colors = uf.get_colors(5)
    color_1 = uf.colors_255_to_1(colors)[ind]

    stimIDs_dic = wf.get_stimIDs_for_whisker(dic_mapping, w)
    stimIDs = [int(key) for key in stimIDs_dic]
    df_spec_w = dfg[dfg['StimID'].isin(stimIDs)]
    df_spec_w = df_spec_w[['StimID','dataframe'] + columns]


    xlabels = [stimIDs_dic[key] for key in stimIDs_dic]

    f, ax = plt.subplots()
    sns.pointplot(x="StimID", y=column,
                  data=df_spec_w, ax=ax,ci='sem',color='k')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    ax.set(xlabel='', ylabel=column)
    plt.show()

    # fig_name = work_dir + 'Figs\\' + 'plot_average_black_sem_' + type + '_' + column + '.pdf'
    # f.savefig(fig_name, bbox_inches='tight')

# ================= statistics ==================
# A. ANOVA test for each variable (average across conditions) ####
import scipy.stats as stats
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.formula.api import ols,OLS
p_vals = np.zeros(5)

w = 'C1'
stimIDs_dic = wf.get_stimIDs_for_whisker(dic_mapping, w)
stimIDs = [int(key) for key in stimIDs_dic]
dfg_w= dfg[dfg['StimID'].isin(stimIDs)]

column,i = 'mean_angle_C1',0
# column,i = 'rng_angle_C1',1
# column,i = 'contact_duration_C1',2
# column,i = 'num_touches_C1',3
# column,i = 'mean_deltaKappa_part1_external_C1',4
dfg_wc = dfg_w[['StimID',column]]
# dfg_wc[column].describe() # all data
# dfg_wc.groupby('StimID')[column].describe() # grouped

x = dfg_wc.loc[dfg_wc['StimID'] == 1,column]
x = dfg_wc[column][dfg_wc['StimID'] == 1] # same as above
vars = [dfg_wc[dfg_wc['StimID'] == s][column] for s in stimIDs]

oneway_res = stats.f_oneway(*vars) # 1st way
p_vals[i]=oneway_res.pvalue

l_model = ols(column + ' ~ C(StimID)', data=dfg_wc).fit()
l_model.summary()

anova_tbl = sm.stats.anova_lm(l_model, typ=2)

def update_anova_table(anova_tbl):
    SS = anova_tbl[:]['sum_sq']
    SS_T = sum(anova_tbl['sum_sq'])
    df = anova_tbl[:]['df']

    anova_tbl['mean_sq'] =  SS / df
    anova_tbl['R_sq'] = SS[:-1] / SS_T
    anova_tbl['omega_sq'] = (anova_tbl[:-1]['sum_sq'] - (anova_tbl[:-1]['df'] * anova_tbl['mean_sq'][-1])) / (sum(anova_tbl['sum_sq']) + anova_tbl['mean_sq'][-1])

    cols = ['sum_sq', 'df', 'mean_sq', 'F', 'PR(>F)', 'R_sq', 'omega_sq']
    anova_tbl = anova_tbl[cols]
    return anova_tbl

anova_tbl = update_anova_table(anova_tbl)

# checking assumptions
# 1: All
l_model.diagn

# 2: Assumption: Homogeneity of Variance (Levine)
levine = stats.levene(*vars)
shapiro = stats.shapiro(l_model.resid)

# ANOVA alternative: Kruskal-Wallis
KW = stats.kruskal(*vars)

# checking for multiple comparisons across 5 graphs
# ==================================================
from statsmodels.stats import multitest
reject= multitest.multipletests(p_vals, alpha=0.05, method='fdr_bh', is_sorted=False, returnsorted=False)
# p_vals_sorted = np.sort(p_vals)
# i = np.arange(1, 5+1) # the 1-based i index of the p values, as in p(i)
# plt.plot(i, p_vals_sorted, '.')
# q = 0.05
# plt.plot(i, p_vals_sorted, 'b.', label='$p(i)$')
# plt.plot(i, q * i / 5, 'r', label='$q i / N$')
# plt.show()

# B. t-test for slope
# ====================
p_vals = np.zeros(5)

# get data
xy = np.concatenate( par_array, axis=1 ).T
df_xy = pd.DataFrame(xy,columns=['x', 'y'])

# build linear model
formula = 'y ~ x'
results = smf.ols(formula,data=df_xy).fit()

# Also: examine the ci of the slope coefficient
# if 1 is not included it is statistically significant
alpha = 0.05
cis = results.conf_int(alpha=alpha).iloc[1:]

hypotheses = '(x = 1)'
t = results.t_test(hypotheses)
params = (t.tvalue[0][0],t.pvalue.flatten()[0])
p_vals[ii] = t.pvalue.flatten()[0]
params

f, ax = plt.subplots()

min_lim = np.min(par_array)
max_lim = np.max(par_array)
lims = [min_lim - (max_lim-min_lim)/20,max_lim + (max_lim-min_lim)/20]
ax.plot(lims,lims,'k')
ax.set_xlim(lims)
ax.set_ylim(lims)
sns.regplot(x=xy[:,0], y=xy[:,1],ci=100-alpha,ax=ax)

ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

ax.set(xlabel=column, ylabel=column)

plt.show()

# B. paired t-test alternative for slope
# =======================================
p_vals_trel = np.zeros(5)

# get data
xy = np.concatenate( par_array, axis=1 ).T

trel = stats.ttest_rel(xy[:,0],xy[:,1])
p_vals_trel[ii] = trel.pvalue
