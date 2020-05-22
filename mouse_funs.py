import numpy as np
import scipy.io as sio
import h5py
import glob
import pandas as pd
import time
import csv

import trial_funs as trf
import util_funs as uf

work_dir = 'C:\\Users\\amird\\OneDrive\\Documents\\Berkeley\\Amir\\' # global variable

def get_whisker_stim_mapping(mouse,names=None):
    mouse_short = mouse[0:4]
    file_name_exp = work_dir + mouse_short + '\\' + mouse + '.exp'
    exp_obj = h5py.File(file_name_exp, 'r')

    # get which pistons are involved
    experiment = exp_obj.get('Experiment')
    experiment_stim = experiment.get('stim')
    logical_stim = experiment_stim.get('stim')
    logical_stim = np.array(logical_stim).T
    stim_whiskers,_ = np.where(logical_stim)

    dic_mapping = {}

    # run over all rows
    for i,v in enumerate(logical_stim):
        whisker_numbers = list(np.where(v)[0] + 1)

        if names is not None:
            whisker_stims = [names[i-1] for i in whisker_numbers]
        else:
            whisker_stims = whisker_numbers

        dic_mapping[str(i)] = whisker_stims

    return dic_mapping

def get_stim_pairs(dic_mapping):
    pairs_list = []
    # find stims - single pistons
    for key in dic_mapping:
        n_pistons = len(dic_mapping[key])

        # if mix
        if n_pistons >= 2:
            i_list = []
            for p in dic_mapping[key]:
                p = [p]
                for i,j in dic_mapping.items():
                    if j == p:
                        i_list.append(int(i))
                        break

            key_int = int(key)
            pair = (key_int,i_list)
            pairs_list.append(pair)
    return pairs_list

def get_whisker_names(mouse,asdic=False):
    mouse_short = mouse[0:4]
    file_whisker_names = work_dir + mouse_short + '\\' + mouse + '_exp_whisker_names.mat'
    whisker_names = sio.loadmat(file_whisker_names)
    whisker_names = whisker_names['whisker_names']

    n_whisker_names = len(whisker_names[0])

    if asdic == True:
        whisker_names_dic = {}
        for i in range(n_whisker_names):
            key = i+1
            whisker_name_i = whisker_names[0][i][0]
            whisker_names_dic[key] = whisker_name_i
        return whisker_names_dic
    else:
        whisker_names_ls = []
        for i in range(n_whisker_names):
            whisker_name_i = whisker_names[0][i][0]
            whisker_names_ls.append(whisker_name_i)
        return whisker_names_ls

def get_trials_by_stimID(df,stimID):
    trials = np.array(df['TrialIndex'].loc[df['StimID'] == stimID])
    return trials

def get_piston_combinations(mouse):
    mouse_short = mouse[0:4]
    file_name_exp = work_dir + mouse_short + '\\' + mouse + '.exp'
    exp_obj = h5py.File(file_name_exp, 'r')

    # get which pistons are involved
    experiment = exp_obj.get('Experiment')
    experiment_stim = experiment.get('stim')
    experiment_piston_combinations = experiment_stim.get('pistonCombinations')
    n_stims = experiment_piston_combinations.shape[1]

    piston_comb = {}
    # create piston combinations dictionary:
    for i in range(n_stims):
        pistons = np.array(exp_obj[experiment_piston_combinations[0][i]]).flatten().astype(int)

        # handle None stimulus
        if ~pistons.all():
            pistons = np.array([])
        pistons = pistons.tolist()

        piston_comb_i = {str(i): pistons}
        piston_comb = {**piston_comb,**piston_comb_i}

    return piston_comb


def get_internal_kappa_from_all_trials(mouse, trials_to_take,whisker_order=[3,2,4,1,5],
                                      pixels_after_to_fit={3: 5, 2: 5, 4: 5, 1: 5, 5: 20},
                                      date_DLC_files='Feb19',iters_DLC_files=250000,n_points_fit=500):
    mouse_short = mouse[0:4]

    path = work_dir + mouse_short + '\\DLCoutput\\*.h5'
    files = glob.iglob(path)

    kappa_intenal_part1_list = []
    kappa_intenal_part2_list = []
    kappa_intenal_half_list = []

    kappa_intenal_part1_mean = {1:0,2:0,3:0,4:0,5:0}
    kappa_intenal_part2_mean = {1:0,2:0,3:0,4:0,5:0}
    kappa_intenal_half_mean = {1:0,2:0,3:0,4:0,5:0}

    err_trials = []

    for file in files:
        uf.tic()
        trial = file[77:80]  # trial number in file name

        if int(trial) in trials_to_take:

            frames, data, probs = trf.get_DLC_data(mouse, trial, whisker_order=whisker_order,
                                                   convert_to_int=False,date=date_DLC_files,
                                                   iters=iters_DLC_files)
            frames_trace, data_trace, score_dic = trf.fit_DLC_data(data, type='ransac',
                                                                   min_samples_for_reg=4,
                                                                   residual_threshold=None,
                                                                   pixels_after_to_fit=pixels_after_to_fit,
                                                                   th_score_high=0.2,
                                                                   th_score_low=-0.1,
                                                                   n_points=n_points_fit)

            # linear curve for angle calculation
            try:
                kappa = trf.calc_data_curvature(data_trace, frames_trace, s_start=0, s_end=0.3, type='kappa',alg = 'derivative')
                arcs = trf.calc_arc_length(data_trace)
                angles = trf.calc_angle(data_trace)

            except np.lib.polynomial.RankWarning:
                err_trials.append(int(trial))
                continue
            except RuntimeWarning:
                err_trials.append(int(trial))
                continue


            frames_trace_f, data_trace_f, (kappa_f, arcs_f, angles_f) = trf.filter_data_by_parameter(frames_trace,
                                                                                                      data_trace,
                                                                                                      kappa,
                                                                                                      arcs,angles)

            frames_interp, data_interp = trf.fill_DLC_data(frames_trace_f, data_trace_f)

            start_frame_internal = 0
            stop_frame_internal = 30
            kappa_intenal_part1 = trf.calc_data_curvature(data_interp, frames_interp, s_start=0, s_end=0.3,
                                                          start_frame_internal=start_frame_internal,
                                                          stop_frame_internal = stop_frame_internal,
                                                          type='kappa_int')
            kappa_intenal_part1_list.append(kappa_intenal_part1)

            start_frame_internal = 40
            stop_frame_internal = 300
            kappa_intenal_part2 = trf.calc_data_curvature(data_interp, frames_interp, s_start=0, s_end=0.3,
                                                          start_frame_internal=start_frame_internal,
                                                          stop_frame_internal = stop_frame_internal,
                                                          type='kappa_int')
            kappa_intenal_part2_list.append(kappa_intenal_part2)

            start_frame_internal = 0
            stop_frame_internal = 150
            kappa_intenal_half= trf.calc_data_curvature(data_interp, frames_interp, s_start=0, s_end=0.3,
                                                        start_frame_internal=start_frame_internal,
                                                        stop_frame_internal = stop_frame_internal,
                                                        type='kappa_int')
            kappa_intenal_half_list.append(kappa_intenal_half)
        print('Finished trial: ' + trial)
        uf.toc()

    # average
    # part 1:
    for dic in kappa_intenal_part1_list:
        for key in dic:
            kappa_intenal_part1_mean[key] = kappa_intenal_part1_mean[key] + dic[key]

    for key in kappa_intenal_part1_mean:
        kappa_intenal_part1_mean[key] = kappa_intenal_part1_mean[key]/len(kappa_intenal_part1_list)

    # part 2:
    for dic in kappa_intenal_part2_list:
        for key in dic:
            kappa_intenal_part2_mean[key] = kappa_intenal_part2_mean[key] + dic[key]

    for key in kappa_intenal_part2_mean:
        kappa_intenal_part2_mean[key] = kappa_intenal_part2_mean[key] / len(kappa_intenal_part2_list)

    # half:
    for dic in kappa_intenal_half_list:
        for key in dic:
            kappa_intenal_half_mean[key] = kappa_intenal_half_mean[key] + dic[key]

    for key in kappa_intenal_half_mean:
        kappa_intenal_half_mean[key] = kappa_intenal_half_mean[key] / len(kappa_intenal_half_list)


    return kappa_intenal_part1_mean, kappa_intenal_part2_mean, kappa_intenal_half_mean, err_trials

# create list of dics for a parameter
def create_dic_parameters_of_listdics(mouse,all_5_piston_locs,
                                      all_5_piston_subclass,
                                      types_list, trials_to_take,
                                      whisker_order=[3,2,4,1,5],
                                      pixels_after_to_fit={3: 5, 2: 5, 4: 5, 1: 5, 5: 20},
                                      s_contact_start={3: 1, 2: 1, 4: 1, 1: 1, 5: 1},
                                      s_contact_end={3: 1, 2: 1, 4: 1, 1: 1, 5: 1},
                                      date_DLC_files='Feb19',iters_DLC_files=250000,n_points_fit=500,
                                      external_kappa_internal_list = None):
    '''
    mouse:
    type_args:  'mean_deltaKappa'
                'mean_angle'
                'rng_angle'
                'num_contacts'
                'num_contacts_above'
                'num_contacts_below'

    '''

    mouse_short = mouse[0:4]

    path = work_dir + mouse_short + '\\DLCoutput\\*.h5'
    files = glob.iglob(path)

    parameters_dic = {}
    # add blank lists to dic for appending later
    for key in types_list:
        parameters_dic[key] = []

    trials = []
    err_trials = []


    for file in files:
        uf.tic()
        trial = file[77:80]  # trial number in file name

        if int(trial) in trials_to_take:

            frames, data, probs = trf.get_DLC_data(mouse, trial, whisker_order=whisker_order,
                                                   convert_to_int=False,date=date_DLC_files,
                                                   iters=iters_DLC_files)
            frames_trace, data_trace, score_dic = trf.fit_DLC_data(data, type='ransac',
                                                                   min_samples_for_reg=4,
                                                                   residual_threshold=None,
                                                                   pixels_after_to_fit=pixels_after_to_fit,
                                                                   th_score_high=0.2,
                                                                   th_score_low=-0.1,
                                                                   n_points=n_points_fit)

            # linear curve for angle calculation
            try:
                kappa = trf.calc_data_curvature(data_trace, frames_trace, s_start=0, s_end=0.3, type='kappa',alg = 'derivative')
                arcs = trf.calc_arc_length(data_trace)
                angles = trf.calc_angle(data_trace)

            except np.lib.polynomial.RankWarning:
                err_trials.append(int(trial))
                continue
            except RuntimeWarning:
                err_trials.append(int(trial))
                continue


            frames_trace_f, data_trace_f, (kappa_f, arcs_f, angles_f) = trf.filter_data_by_parameter(frames_trace,
                                                                                                      data_trace,
                                                                                                      kappa,
                                                                                                      arcs,angles)

            frames_interp, data_interp = trf.fill_DLC_data(frames_trace_f, data_trace_f)

            kappa_interp  = trf.calc_data_curvature(data_interp, frames_interp, s_start=0, s_end=0.3, type='kappa')
            arcs_interp = trf.calc_arc_length(data_interp)
            angles_interp = trf.calc_angle(data_interp)

            c = 0
            for type in types_list:

                if type == 'mean_deltaKappa_part1':
                    deltaKappa_interp = trf.calc_data_curvature(data_interp, frames_interp, s_start=0, s_end=0.3,
                                                                start_frame_internal=0,
                                                                stop_frame_internal=30,
                                                                external_kappa_internal=None,
                                                                type='delta_kappa')

                    statistic_parameter = trf.get_statistic_parameter(deltaKappa_interp, 'mean',start=34)

                elif type == 'mean_deltaKappa_part2':
                    deltaKappa_interp = trf.calc_data_curvature(data_interp, frames_interp, s_start=0, s_end=0.3,
                                                                start_frame_internal=40,
                                                                stop_frame_internal=300,
                                                                external_kappa_internal=None,
                                                                type='delta_kappa')

                    statistic_parameter = trf.get_statistic_parameter(deltaKappa_interp, 'mean',start=34)

                elif type == 'mean_deltaKappa_half':
                    deltaKappa_interp = trf.calc_data_curvature(data_interp, frames_interp, s_start=0, s_end=0.3,
                                                                start_frame_internal=0,
                                                                stop_frame_internal=150,
                                                                external_kappa_internal=None,
                                                                type='delta_kappa')

                    statistic_parameter = trf.get_statistic_parameter(deltaKappa_interp, 'mean',start=34)

                elif type == 'mean_deltaKappa_part1_external':
                    external_kappa_internal_part1 = external_kappa_internal_list[0]
                    deltaKappa_interp = trf.calc_data_curvature(data_interp, frames_interp, s_start=0, s_end=0.3,
                                                                start_frame_internal=None,
                                                                stop_frame_internal=None,
                                                                external_kappa_internal=external_kappa_internal_part1,
                                                                type='delta_kappa')

                    statistic_parameter = trf.get_statistic_parameter(deltaKappa_interp, 'mean',start=34)

                elif type == 'mean_deltaKappa_part2_external':
                    external_kappa_internal_part2 = external_kappa_internal_list[1]
                    deltaKappa_interp = trf.calc_data_curvature(data_interp, frames_interp, s_start=0, s_end=0.3,
                                                                start_frame_internal=None,
                                                                stop_frame_internal=None,
                                                                external_kappa_internal=external_kappa_internal_part2,
                                                                type='delta_kappa')

                    statistic_parameter = trf.get_statistic_parameter(deltaKappa_interp, 'mean',start=34)

                elif type == 'mean_deltaKappa_half_external':
                    external_kappa_internal_half = external_kappa_internal_list[2]
                    deltaKappa_interp = trf.calc_data_curvature(data_interp, frames_interp, s_start=0, s_end=0.3,
                                                                start_frame_internal=None,
                                                                stop_frame_internal=None,
                                                                external_kappa_internal=external_kappa_internal_half,
                                                                type='delta_kappa')

                    statistic_parameter = trf.get_statistic_parameter(deltaKappa_interp, 'mean',start=34)

                elif type == 'mean_kappa':
                    statistic_parameter = trf.get_statistic_parameter(kappa_interp, 'mean',start=34)

                elif type == 'mean_angle':
                    statistic_parameter = trf.get_statistic_parameter(angles_interp, 'mean',start=34)

                elif type == 'rng_angle':
                    statistic_parameter = trf.get_statistic_parameter(angles_interp, 'range',start=34)

                elif type == 'num_contacts' or type == 'num_contacts_above' or type == 'num_contacts_below' or type == 'num_touches':

                    if c==0:
                        pistons_in_trial = trf.get_which_pistons_in_trial(mouse, trial)
                        pistons_in_trial_locs, pistons_in_trial_subclass = trf.get_trial_piston_locs(all_5_piston_locs,
                                                                                             all_5_piston_subclass,
                                                                                             pistons_in_trial)
                        pxls_for_contact = trf.get_pxls_for_contact(data_interp, pistons_in_trial,
                                                                    s_contact_start,s_contact_end)

                        # tips = trf.get_data_tip(data_interp, get_tips_to=pistons_in_trial)

                        contacts, contacts_above, contacts_below = trf.get_contacts(pxls_for_contact, frames_interp, pistons_in_trial_locs,
                                                                                pistons_in_trial_subclass)
                        c = c+1

                    if type == 'num_contacts':
                        statistic_parameter = trf.calc_num_contacts(contacts)

                    elif type == 'num_contacts_above':
                        statistic_parameter = trf.calc_num_contacts(contacts_above)

                    elif type == 'num_contacts_below':
                        statistic_parameter = trf.calc_num_contacts(contacts_below)

                    elif type == 'num_touches':
                        touches = trf.calc_touches(contacts)
                        statistic_parameter = trf.calc_num_contacts(touches)

                else:
                    raise ValueError('type is not recognized:{}'.format(type))

                parameters_dic[type].append(statistic_parameter)

            print('finished trial: ' + trial)
            uf.toc()


    return parameters_dic,err_trials

def insert_listdic_to_csv(csv_name,trials,parameter,column_names_list,iter=None):
    if iter is None:
        file_name_load = csv_name
        iter = 0
    else:
        file_name_load = csv_name + '_' + str(iter) + '.csv'
    df = pd.read_csv(file_name_load)

    df = add_columns_df(df, column_names_list)
    df = add_diclist_to_df(df, trials, parameter)

    file_name_save = csv_name + '_' + str(iter+1) + '.csv'

    df.to_csv(file_name_save, index=False)

def change_listdic_keys_to_column_names(listdic,column_names):
    listdic_new_names = []
    n_dics = len(listdic)

    for i in range(n_dics):
        dic_new_names = {}
        for key in listdic[i]:
            dic_new_names[column_names[key]] = listdic[i][key]

        listdic_new_names.append(dic_new_names)

    return listdic_new_names

def add_dic_to_csv(dic,file_name ):

    my_dict = {"test": 1, "testing": 2}

    with open('mycsvfile.csv', 'w') as f:
        w = csv.DictWriter(f, my_dict.keys())
        w.writeheader()
        w.writerow(my_dict)


def add_diclist_to_df(df, trials, list_dics):
    df_new = df.copy()

    for i, tr in enumerate(trials):
        rowIndex = df_new.index[df_new['TrialIndex'] == int(tr)]

        num_contacts_i = list_dics[i]
        for key in num_contacts_i:
            df_new.loc[rowIndex, key] = num_contacts_i[key]

    return df_new

def add_columns_df(df, new_columns):
    df_new = df.copy()
    for c in new_columns:
        df_new[c] = np.nan

    return df_new

def insert_bool_trial_conditioned_to_csv(csv_name,column_name,trials):
    file_name_load = csv_name + '.csv'
    df = pd.read_csv(file_name_load)
    df[column_name] = df['TrialIndex'].isin(trials)

    file_name_save = csv_name + '_1.csv'
    df.to_csv(file_name_save, index=False)

def insert_vector_to_csv(csv_name_load,column_name,vector):

    df = pd.read_csv(csv_name_load)
    df[column_name] = vector

    file_name_save = csv_name_load[:-4] + '_1.csv'
    df.to_csv(file_name_save, index=False)
    return file_name_save

def get_evan_included_trials(mouse):
    mouse_short = mouse[0:4]
    file_name = work_dir + mouse_short + '\\' + mouse

    data_obj = h5py.File(file_name + '.exp', 'r')
    n_trials = data_obj['TrialIndex'].size
    trials = list(np.array(data_obj['TrialIndex']).reshape(n_trials).astype(int))

    return trials


def get_whisker_tracing(mouse,all_5_piston_locs,all_5_pistons_subclass,types_list,
                       trials_to_take,n_points_fit=500,
                       whisker_order=[3, 2, 4, 1, 5],
                       pixels_after_to_fit={3: 5, 2: 5, 4: 5, 1: 5, 5: 20},
                       s_contact_start= {3: 1, 2: 1, 4: 1, 1: 1, 5: 1},
                       s_contact_end={3: 1, 2: 1, 4: 1, 1: 1, 5: 1},
                       date_DLC_files='Feb19',
                       iters_DLC_files=250000,
                       external_kappa_internal=None):

    mouse_short = mouse[0:4]

    path = work_dir + mouse_short + '\\DLCoutput\\*.h5'
    files = glob.iglob(path)

    trials = []
    err_trials = []
    tracings = {}

    for key in types_list:
        tracings[key] = []

    for file in files:
        uf.tic()
        trial = file[77:80]  # trial number in file name

        if int(trial) in trials_to_take:

            frames, data, probs = trf.get_DLC_data(mouse, trial, whisker_order=whisker_order,
                                                   convert_to_int=False,date=date_DLC_files,
                                                   iters=iters_DLC_files)
            frames_trace, data_trace, score_dic = trf.fit_DLC_data(data, type='ransac',
                                                                   min_samples_for_reg=4,
                                                                   residual_threshold=None,
                                                                   pixels_after_to_fit=pixels_after_to_fit,
                                                                   th_score_high=0.2,
                                                                   th_score_low=-0.1,
                                                                   n_points=n_points_fit)
            # linear curve for angle calculation
            try:
                kappa = trf.calc_data_curvature(data_trace, frames_trace, s_start=0, s_end=0.3, type='kappa',alg = 'derivative')
                arcs = trf.calc_arc_length(data_trace)
                angles = trf.calc_angle(data_trace)

            except np.lib.polynomial.RankWarning:
                err_trials.append(int(trial))
                continue
            except RuntimeWarning:
                err_trials.append(int(trial))
                continue

            frames_trace_f, data_trace_f, (kappa_f, arcs_f, angles_f) = trf.filter_data_by_parameter(frames_trace,
                                                                                                      data_trace,
                                                                                                      kappa,
                                                                                                      arcs,angles)

            frames_interp, data_interp = trf.fill_DLC_data(frames_trace_f, data_trace_f)

            kappa_interp  = trf.calc_data_curvature(data_interp, frames_interp, s_start=0, s_end=0.3, type='kappa')
            arcs_interp = trf.calc_arc_length(data_interp)
            angles_interp = trf.calc_angle(data_interp)

            c = 0
            for type in types_list:

                if type == 'deltaKappa':
                    deltaKappa_interp = trf.calc_data_curvature(data_interp, frames_interp, s_start=0, s_end=0.3,
                                                                start_frame_internal=0,
                                                                stop_frame_internal=30,
                                                                external_kappa_internal=None,
                                                                type='delta_kappa')
                    tracings[type].append(deltaKappa_interp)

                elif type == 'deltaKappa_external':
                    deltaKappa_interp_ext = trf.calc_data_curvature(data_interp, frames_interp, s_start=0, s_end=0.3,
                                                                    start_frame_internal=None,
                                                                    stop_frame_internal=None,
                                                                    external_kappa_internal=external_kappa_internal,
                                                                    type='delta_kappa')
                    tracings[type].append(deltaKappa_interp_ext)

                elif type == 'kappa':
                    tracings[type].append(kappa_interp)

                elif type == 'angle':
                    tracings[type].append(angles_interp)

                elif type == 'contact' or type == 'contact_above' or type == 'contact_below' or type == 'touch':

                    if c==0:
                        pistons_in_trial = trf.get_which_pistons_in_trial(mouse, trial)
                        pistons_in_trial_locs, pistons_in_trial_subclass = trf.get_trial_piston_locs(all_5_piston_locs,
                                                                                             all_5_pistons_subclass,
                                                                                             pistons_in_trial)
                        pxls_for_contact = trf.get_pxls_for_contact(data_interp, pistons_in_trial,
                                                                    s_contact_start,s_contact_end)

                        contacts, contacts_above, contacts_below = trf.get_contacts(pxls_for_contact, frames_interp, pistons_in_trial_locs,
                                                                                pistons_in_trial_subclass)
                        c = c+1

                    if type == 'contact':
                        tracings[type].append(contacts)

                    elif type == 'contact_above':
                        tracings[type].append(contacts_above)

                    elif type == 'contact_below':
                        tracings[type].append(contacts_below)

                    elif type == 'touch':
                        touches = trf.calc_touches(contacts)
                        tracings[type].append(touches)

                else:
                    raise ValueError('type is not recognized:{}'.format(type))

            print('finished trial: ' + trial)
            trials.append(int(trial))
            uf.toc()

    return tracings,trials,err_trials