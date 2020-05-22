import numpy as np
import pandas as pd
import h5py
import matplotlib.pyplot as plt

import trial_funs as trf
import util_funs as uf

work_dir = 'C:\\Users\\amird\\OneDrive\\Documents\\Berkeley\\Amir\\' # global variable

# get whisker summary dataframe from csv file
def get_whisk_summary(mouse,trial):
    mouse_short = mouse[0:4]
    file_name_whisk_summary = work_dir + mouse_short + '\\Traces_vars' + '\\' + mouse + '_0' + trial + '_summary.csv'

    whisk_summary = pd.read_csv(file_name_whisk_summary)

    return whisk_summary

def replace_df_labels_to_real(df,new_labels):
    # replace label 1 in whisk_summary to 4, 2 is 2, etc.
    # to know the "real" numbering (according to experiment order)

    # temp dummy replacing
    for i in (np.arange(5)+1):
        df['label'].replace(i, str(i), inplace=True)

    for i in (np.arange(5)+1):
        df['label'].replace(str(i), new_labels[i-1], inplace=True)
    return  df

def get_object_all_whiskers_pixels(mouse,trial):
    mouse_short = mouse[0:4]
    file_name_pixels = work_dir + mouse_short + '\\Traces' + '\\' + mouse + '_0' + trial + '.whisk'

    hdf5_obj_all_whiskers = h5py.File(file_name_pixels, 'r')
    return hdf5_obj_all_whiskers


def clear_duplicate_frames(frames, data, pixelen):
    # after getting the relevant frames for a single whisker
    # this function makes sure there are no duplicates
    is_np = isinstance(data,np.ndarray)

    un_frames, counts = np.unique(frames, return_counts=True)

    # check for duplicate values
    if np.any(counts > 1):
        ind_un_dup = np.where(counts > 1)[0]

        nested_list_to_remove = []
        # run over each duplicate value
        for i in ind_un_dup:
            dup_value = un_frames[i]

            # find the duplicate indices in the frames array
            ind_frames = np.where(frames == dup_value)[0].tolist()
            pixelen_to_compare = np.take(pixelen,ind_frames)
            max_index = np.argmax(pixelen_to_compare)

            # remove from list all values that are not
            ind_frames_to_remove_i=ind_frames.copy()
            ind_frames_to_remove_i.pop(max_index)  # now ind_frames consists of indices that should be removed
            nested_list_to_remove.append(ind_frames_to_remove_i)

        # make a flat list of indices to remove
        flat_list_to_remove = [item for sublist in nested_list_to_remove for item in sublist]

        # remove
        data = [i for j, i in enumerate(data) if j not in flat_list_to_remove]

        # if the original data type was numpy array, change it back to numpy array
        if is_np:
            data = np.array(data)

        frames = [i for j, i in enumerate(frames) if j not in flat_list_to_remove]
        frames = np.array(frames)

    return frames, data

def get_whisker_contacts(pxls_for_contact_list,frames_list,piston_locs_list,piston_subclass_list):
    # tip_pxls should be a list of 2 x n numpy array
    # calculate contacts by pistons proximity to tip (and curv change)
    stim_show_frame = 34

    contacts_list = []
    contacts_above_list = []
    contacts_below_list = []

    for i,pxls_for_contact_i in enumerate(pxls_for_contact_list):
        if i >= stim_show_frame:
            indices = uf.ismember(pxls_for_contact_i, piston_locs_list).tolist()

            is_contact = any([len(indices)])

            # check if empty
            if is_contact:
                contacts_list.append(frames_list[i])

            if piston_subclass_list is not None:
                indices_above = uf.ismember(pxls_for_contact_i, piston_locs_list[piston_subclass_list]).tolist()
                indices_below = uf.ismember(pxls_for_contact_i, piston_locs_list[~piston_subclass_list]).tolist()

                is_contact_above = any([len(indices_above)])
                is_contact_below = any([len(indices_below)])

                if is_contact_above:
                    contacts_above_list.append(frames_list[i])

                elif is_contact_below:
                    contacts_below_list.append(frames_list[i])

            else:
                contacts_above_list = None
                contacts_below_list = None

    return contacts_list,contacts_above_list ,contacts_below_list

def get_all_whiskers_contacts(all_frames_whisker_tips,all_whisker_tips,all_piston_locs):
    # tip_pxls should be a list of 2 x n numpy array

    all_frames_contacts = {}
    for key in all_piston_locs:
        frames_contacts = get_whisker_contacts(all_frames_whisker_tips[key],
                                               all_whisker_tips[key],all_piston_locs[key])
        all_frames_contacts[key] = frames_contacts

    return all_frames_contacts


def plot_num_contacts(num_contacts,colors):
    # colors = np.array(colors)

    whiskers = np.array(list(num_contacts.keys()))
    num_contacts_array = np.array(list(num_contacts.values()))

    plt.bar(whiskers,num_contacts_array,color=colors,alpha=0.5)
    plt.ylabel('# Contact Frames',fontsize = 18)
    plt.show()

def get_stimIDs_for_whisker(dic_mapping,w):
    stimIDs = {}
    for key in dic_mapping:
        if w in dic_mapping[key]:
            stimIDs[key] = dic_mapping[key]

    return stimIDs


def get_DLC_data_whisker(listdics,w_label):
    '''returns pixels for single whisker assuming each point has data in each frame'''

    list_3_pts = []
    for i,dic in enumerate(listdics):
        list_3_pts.append(dic[w_label])

    return list_3_pts

def calc_whisker_kappa(pxls_list,s_start=0,s_end=0.3,alg='derivative'):
    '''returns the curvature of a whisker at a desired point
        x_list: pixels of whisker in the x axis of the image (list according to frames)
        y_list: pixels of whisker in the y axis of the image (list according to frames)
        s_start: relative start location on the whisker
        to calculate curvature where 0 = folicle and 1 = tip
        s_end: relative end location on the whisker

        '''

    # if folicle was chosen for start and end
    if s_start == s_end and s_end == 0:
        s_end = 0.1
    elif s_start == s_end and s_start == 1:
        s_start = 0.9

    s_mid = s_start + (s_end - s_start)/2

    whisker_len = len(pxls_list[0][:,0])
    s_start_ind = round(whisker_len * s_start)
    s_mid_ind = round(whisker_len * s_mid)
    if s_end >= 1:
        s_end_ind = -1 # last index
    else:
        s_end_ind = round(whisker_len * s_end)

    kappa_list = []

    for pxls in pxls_list:
        x = pxls[:,0]
        y = pxls[:,1]
        p1 = (x[s_start_ind],y[s_start_ind])
        p2 = (x[s_mid_ind]  ,y[s_mid_ind])
        p3 = (x[s_end_ind]  ,y[s_end_ind])

        kappa_i = uf.define_curvature(p1,p2,p3,alg = alg)
        kappa_list.append(kappa_i)

    return np.array(kappa_list)

def calc_whisker_internal_kappa(pxls_list_whisker,frames_whisker,s_start=0,s_end=0.3,
                                start_frame_internal = 0,stop_frame_internal = 30,alg='derivative'):
    '''returns whisker internal (baseline) curvature'''

    # find stop frame in frames
    _,idx_start_frame = uf.find_nearest(frames_whisker,start_frame_internal,type = 'above')
    _,idx_stop_frame = uf.find_nearest(frames_whisker,stop_frame_internal)

    pxls_use_list = pxls_list_whisker[start_frame_internal:idx_stop_frame+1]

    kappa = calc_whisker_kappa(pxls_use_list ,s_start, s_end,alg=alg)

    kappa_int = np.array(kappa).mean()

    return kappa_int

def calc_whisker_delta_kappa(kappa_list,kappa_int):
    '''return delta kappa for single whisker '''

    deltaKappa = []
    for kappa in kappa_list:
        deltaKappa_i = kappa - kappa_int
        deltaKappa.append(deltaKappa_i)

    return np.array(deltaKappa)

def calc_whisker_angle(pxls,n_points = 10,reference = 90):
    '''

    pxls:
    n_points: number of points in whisker for calculating linear fit
    reference: degrees to add to all calculations
               0 degrees -> relative to the horizontal axis
               90 degrees -> relative to the vertical axis

    '''

    x = np.array(pxls[0:n_points, 0])
    y = np.array(pxls[0:n_points, 1])

    # fit line to those points
    p = np.polyfit(x, y, deg=1)
    slope = p[0]
    angle = np.rad2deg(np.arctan(slope))*(-1) + reference # -1 due to inverted y axis

    return angle