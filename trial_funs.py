import numpy as np
import pandas as pd
import h5py
from scipy import stats
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import statsmodels.api as sm
import itertools
from sklearn import preprocessing as pp
from sklearn import linear_model as lm

import mouse_funs as mf
import whisker_funs as wf
import util_funs as uf


work_dir = 'C:\\Users\\amird\\OneDrive\\Documents\\Berkeley\\Amir\\' # global variable

def get_which_pistons_in_trial(mouse,trial):
    mouse_short = mouse[0:4]
    file_name_analysis_info = work_dir + mouse_short + '\\' + mouse + '_exp_AnalysisInfo.csv'
    analysis_info = pd.read_csv(file_name_analysis_info)

    # get trial numbers
    trial_num = int(trial)
    trial_indices = analysis_info['StimID'].index + 1 # trial number 1 is index 0

    # compare trial number with input trial and get StimID
    trial_stimID = str(int((np.array(analysis_info['StimID'].loc[trial_indices == trial_num]))))

    dic_mapping = mf.get_whisker_stim_mapping(mouse)
    pistons_in_trial = dic_mapping[trial_stimID]

    return pistons_in_trial

def get_DLC_data(mouse,trial,date = 'Feb19',iters = 250000,
                 order = True,convert_to_int = True,
                 n_whiskers = 5,n_points_whisker = 6,
                 whisker_order = [1,2,3,4,5], filter_outliers = True):
    '''return DeepLabCut data:
        1. as numpay array of points (order = False): each point is (frame, 45 values)
        2. as dictionary (order = True):
            key is whisker number
            value is list of frames and in each entry
            n x 2 matrix (rows: fol, ... ,tip; columns: x_pxl, y_pxl)
            (if order is True)

        whisker_order: list
        define ordering of whisker numbers.
        e.g., if whisker 1 in tracing DLC data should be whisker 4
        in the dictionary, enter 4 as the first number in the list.
    '''

    mouse_short = mouse[0:4]
    file_name = work_dir + mouse_short + '\\DLCoutput\\' + mouse + '_0' + trial + 'DeepCut_resnet50_whisking' + date + 'shuffle1_' + str(iters)
    data_obj = h5py.File(file_name + '.h5', 'r')
    data_obj = data_obj.get('df_with_missing')
    # ind= list(pts_obj['_i_table']['index'].keys())

    data_DLC_tuple = np.array(data_obj['table'])
    frames_DLC_tuple = len(data_DLC_tuple)

    # arrange as data in a whisker dictionary
    if order:
        len_data_DLC_tuple = len(data_DLC_tuple)

        frames = {}
        data = {}
        probs = {}
        for i,w in enumerate(whisker_order):
            data_dic_values = []
            frames_dic_values = []
            probs_dic_values = []
            for fr in range(len_data_DLC_tuple):
                frames_i = data_DLC_tuple[fr][0]
                data_DLC_i = data_DLC_tuple[fr][1]

                probs_inds = slice(2, None, 3)
                probs_i = data_DLC_i[probs_inds]
                data_i = np.delete(data_DLC_i, probs_inds)  # get rid of probabilities
                data_i = data_i.reshape((-1, 2))

                # convert to integer
                if convert_to_int:
                    data_i = data_i.astype(np.int32)

                # insert to dictionary
                data_dic_values_i = data_i[i * n_points_whisker:(i + 1) * n_points_whisker]
                probs_dic_values_i = probs_i[i * n_points_whisker:(i + 1) * n_points_whisker]

                data_dic_values.append(data_dic_values_i)
                probs_dic_values.append(probs_dic_values_i)
                frames_dic_values.append(frames_i)

            data[w] = data_dic_values
            probs[w] = probs_dic_values
            frames[w] = frames_dic_values

    else:
        data = data_DLC_tuple
        frames = frames_DLC_tuple
        probs = [] # dummy

    return frames,data,probs

def fit_DLC_data(data,type = 'ransac',min_samples_for_reg=2,residual_threshold=20,
                 n_points = 500,pixels_after_to_fit = {1:0,2:0,3:0,4:0,5:0},
                 th_score_high = 0.4,th_score_low = -0.3,fill_vals = False):
    ''' th_score_high: increase for more strict analysis
        th_score_low: increase for more strict analysis
    '''

    frames = {}
    data_interp = {}
    scores_dic = {}
    filled_dic = {}
    for key in data:
        w_pts = data[key]
        n_frames = len(w_pts)

        frames_list = []
        pxls_list = []
        score_list = []
        filled_list = []
        for i in range(n_frames):
            x, y = w_pts[i][:, 0], w_pts[i][:, 1]

            poly = pp.PolynomialFeatures(degree=2)
            x_trans = poly.fit_transform(x[:, np.newaxis])

            if type =='ransac':
                reg = lm.RANSACRegressor(min_samples=min_samples_for_reg,residual_threshold=residual_threshold, max_trials=10000)
            elif type =='lr':
                reg = lm.LinearRegression()
            elif type == 'ridge':
                reg = lm.Ridge(alpha=1000)

            reg.fit(x_trans, y)
            y_pred = reg.predict(x_trans)

            x_new = np.linspace(np.min(x),np.max(x)+pixels_after_to_fit[key],n_points).astype(np.int32)
            y_new = reg.predict(poly.fit_transform(x_new[:,np.newaxis])).astype(np.int32)

            score = reg.score(x_trans,y)

            # plt.scatter(x, y)
            # plt.plot(x_new,y_new)
            # plt.show()
            # print(score)

            # save only values that stand criterion
            if score < th_score_high and score > th_score_low :
                score_list.append(score)
            else:
                pxls = np.vstack((x_new, y_new)).astype(np.int32).T

                frames_list.append(i)
                pxls_list.append(pxls)
                score_list.append(score)

        frames[key] = frames_list
        data_interp[key] = pxls_list
        scores_dic[key] = score_list

    return frames,data_interp,scores_dic


def filter_data_by_parameter(frames_trace,data_trace,*args):

    # find bad inds from each parameter for each whisker
    high_z_inds = {}
    for key in data_trace:
        high_z_inds_list = []
        for parameter in args:
            parameter_i = parameter[key]
            z = stats.zscore(parameter_i)
            high_z_inds_i = list(np.where(abs(z) > 3)[0])
            high_z_inds_list.append(high_z_inds_i)

        high_z_inds_list_flat = [item for sublist in high_z_inds_list for item in sublist]
        high_z_inds[key] = high_z_inds_list_flat

    # run over all whiskers and delete bad inds
    frames_trace_f = {}
    data_trace_f = {}
    for key in data_trace:
        high_z_inds_i = high_z_inds[key]
        frames_trace_i = frames_trace[key]
        data_trace_i = data_trace[key]

        data_trace_i_f = [pxls for i, pxls in enumerate(data_trace_i) if i not in high_z_inds_i]
        frames_trace_i_f = [fr for i, fr in enumerate(frames_trace_i) if i not in high_z_inds_i]

        data_trace_f[key] = data_trace_i_f
        frames_trace_f[key] = frames_trace_i_f

    args_f = ()
    for parameter in args:
        parameter_f = {}

        for key in data_trace:
            high_z_inds_i = high_z_inds[key]

            parameter_i = parameter[key]
            parameter_i_f = np.delete(parameter_i, high_z_inds_i , None)
            parameter_f[key] = parameter_i_f

        args_f = args_f + (parameter_f,)

    return frames_trace_f,data_trace_f,args_f

def filter_outliers(badframes,frames,data):
    frames_filt,ind_shared = uf.get_unequal_vals_in_2_dics(frames, badframes)

    data_filt = {}
    for key in data:
        data_filt[key] = [data[key][i] for i in ind_shared[key]]

    return frames_filt,data_filt

def fit_curve_DLC_data(data):
    ss_interp = {}
    data_interp = {}
    for key in data:
        list_pts_array = data[key]

        ss_list = []
        pxls_list = []
        for pts_array in list_pts_array:
            pxls_x,pxls_y,ss = uf.fit_curve(pts_array)
            pxls = np.vstack((pxls_x,pxls_y)).astype(np.int32).T

            pxls_list.append(pxls)
            ss_list.append(ss)

        data_interp[key] = pxls_list
        ss_interp[key] = ss_list

    return data_interp,ss_interp

def calc_data_curvature(data,frames,s_start=0 ,s_end=0.3,start_frame_internal=0,stop_frame_internal = 30,
                        external_kappa_internal=None,type='delta_kappa',alg ='derivative'):

    kappa_int = {}
    kappa = {}
    delta_kappa = {}
    for key in data:
        pxls_list_whisker = data[key]
        frames_whisker = frames[key]

        kappa[key] = wf.calc_whisker_kappa(pxls_list_whisker, s_start, s_end, alg=alg)

        if (type=='delta_kappa') or (type=='kappa_int'):
            if external_kappa_internal is None:
                kappa_int[key] = wf.calc_whisker_internal_kappa(pxls_list_whisker,frames_whisker,s_start, s_end,
                                                                start_frame_internal = start_frame_internal,
                                                                stop_frame_internal = stop_frame_internal,
                                                                alg=alg)
            elif (external_kappa_internal is not None) and (type=='delta_kappa'):
                kappa_int = external_kappa_internal.copy()

            if type=='delta_kappa':
                delta_kappa[key] = wf.calc_whisker_delta_kappa(kappa[key], kappa_int[key])

    if type=='delta_kappa':
        return delta_kappa

    elif type=='kappa':
        return kappa

    elif type=='kappa_int':
        return kappa_int


def calc_arc_length(data):
    arcs = {}
    for key in data:
        pxls_list_whisker = data[key]

        arcs_list = []
        for pxls in pxls_list_whisker:
            arc_i = uf.arc_length(pxls[:,0],pxls[:,1])
            arcs_list.append(arc_i)
        arcs[key] = arcs_list
    return arcs

def calc_angle(data):
    angles = {}

    for key in data:
        pxls_list_whisker = data[key]
        n_points_for_angle = round(len(pxls_list_whisker)/10)

        angles_list = []
        for pxls in pxls_list_whisker:
            angle_i = wf.calc_whisker_angle(pxls,n_points=n_points_for_angle)
            angles_list.append(angle_i)
        angles_np = np.array(angles_list)

        # angles[key] = angles_list
        angles[key] = angles_np
    return angles

def get_data_tip(data,get_tips_to = None):
    tips = {}

    # reduce data
    data_reduced = {key: data[key] for key in get_tips_to}

    # get tips from each whisker and each frame
    for key in data_reduced:
        whisker_pxls_list = data_reduced[key]
        tips_list = []

        for pxls in whisker_pxls_list:
            tip = pxls[-1].tolist()
            tips_list.append(tip)

        tips[key] = np.array(tips_list)

    return tips

def get_trial_piston_locs(all_5_piston_locs,all_5_pistons_subclass,pistons_in_trial):
    '''get piston locations (pxls) in a specific trial'''

    trial_piston_locs = {key: all_5_piston_locs[key] for key in pistons_in_trial}
    if all_5_pistons_subclass is not None:
        trial_piston_subclass = {key: all_5_pistons_subclass[key] for key in pistons_in_trial}
    else:
        trial_piston_subclass = None
    return trial_piston_locs,trial_piston_subclass

def get_contacts(pxls_for_contact,frames,piston_locs,piston_subclass):
    '''returns frames in which there is a contact between whiskers and pistons subclassified
    according to above/below piston'''
    contacts = {}

    if piston_subclass is not None:
        contacts_above = {}
        contacts_below = {}
    else:
        contacts_above = None
        contacts_below = None

    for key in pxls_for_contact:
        piston_locs_list = piston_locs[key]
        piston_subclass_list = piston_subclass[key]
        pxls_for_contact_list = pxls_for_contact[key]
        frames_list = frames[key]

        contacts_list,contacts_above_list,contacts_below_list = wf.get_whisker_contacts(pxls_for_contact_list,frames_list,piston_locs_list,piston_subclass_list)

        contacts[key] = np.array(contacts_list)
        contacts_above[key] = np.array(contacts_above_list)
        contacts_below[key] = np.array(contacts_below_list)

    return contacts,contacts_above,contacts_below

def calc_num_contacts(contacts):
    num_contacts = {}
    for key in contacts:
        num_contacts[key] = contacts[key].size

    return num_contacts

def fill_DLC_data(frames,data,n_frames_total = 300):
    '''fill values in data and frames for whisker tracing
    '''

    frames_filled = {}
    data_filled = {}
    for key in data:
        frames_filled_list = []
        data_filled_list = []

        data_i = data[key]

        # add 3rd dimension to pixels
        data_i_reshaped = [pxls.reshape(-1,2,1) for pxls in data_i]
        data_i_time = np.concatenate(data_i_reshaped,axis=2) # slice data_i by time
        n_points_whisker = data_i_time.shape[0]

        frames_i = np.array(frames[key])
        x_interp = np.arange(0,n_frames_total,1)

        data_i_interp_time_list = []

        # check how each whisker pixel is changed over time
        for i in np.arange(n_points_whisker):
            y_0 = data_i_time[i, 0, :]
            y_1 = data_i_time[i, 1, :]
            f_0 = interp1d(frames_i, y_0,kind='linear',bounds_error = False,fill_value=(y_0[0],y_0[-1]))
            f_1 = interp1d(frames_i, y_1,kind='linear',bounds_error = False,fill_value=(y_1[0],y_1[-1]))

            y_0_interp = f_0(x_interp).astype(np.int32)
            y_1_interp = f_1(x_interp).astype(np.int32)

            data_i_interp_time_list.append(np.vstack((y_0_interp,y_1_interp)).reshape(1,2,-1))
        data_i_interp = np.concatenate(data_i_interp_time_list, axis=0)

        # reorder as list
        data_i_interp_list = [data_i_interp[:,:,i].reshape(-1,2) for i in np.arange(n_frames_total)]
        data_filled[key] = data_i_interp_list
        frames_filled[key] = x_interp

    return frames_filled, data_filled


def get_statistic_parameter(parameter,type = 'mean',start = 0):
    '''

    parameter:
    type: 'mean','median','std','range' (range of 95% of histogram)

    '''
    statistic = {}
    for key in parameter:
        parameter_vec = np.array(parameter[key])

        if type == 'mean':
            statistic[key] = np.mean(parameter_vec[start:])
        elif type == 'median':
            statistic[key] = np.median(parameter_vec[start:])
        elif type == 'std':
            statistic[key] = np.std(parameter_vec[start:])
        elif type == 'range':
            rng_min = np.percentile(parameter_vec[start:], 2.5)
            rng_max = np.percentile(parameter_vec[start:], 97.5)
            rng = rng_max - rng_min
            statistic[key] = rng

    return statistic

def get_pxls_for_contact(data,get_pixels_to,s_contact_start={1:1,2:1,3:1,4:1,5:1},s_contact_end={1:1,2:1,3:1,4:1,5:1}):

    pxls_for_contact= {}
    # reduce data
    data_reduced = {key: data[key] for key in get_pixels_to}

    # get tips from each whisker and each frame
    for key in data_reduced:
        whisker_pxls_list = data_reduced[key]
        s_start = s_contact_start[key]
        s_end = s_contact_end[key]

        pxls_for_contact_list = []
        for pxls in whisker_pxls_list:
            whisker_len = len(pxls)

            # take tip if only last value is wanted
            if s_start == s_end and s_start == 1:
                s_start_ind = whisker_len-1
                s_end_ind = whisker_len-1
            else:
                s_start_ind = round(whisker_len * s_start)
                s_end_ind = round(whisker_len * s_end)

            pxls_for_contact_one_frame = pxls[s_start_ind:s_end_ind+1,:].tolist()
            pxls_for_contact_list.append(pxls_for_contact_one_frame)

        pxls_for_contact[key] = np.array(pxls_for_contact_list)

    return pxls_for_contact


def calc_touches(contacts,tolerance=3):
    touches = {}
    for key in contacts:
        countacts_i = contacts[key]
        diffs = np.diff(countacts_i)
        if len(diffs) > 0:
            is_touch = np.hstack((True,diffs > tolerance))
            touches[key] = countacts_i[is_touch]
        else:
            touches[key] = countacts_i

    return touches