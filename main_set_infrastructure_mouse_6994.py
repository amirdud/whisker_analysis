import numpy as np
import glob
import pandas as pd
from datetime import datetime

import vid_funs as vf
import mouse_funs as mf
import util_funs as uf

# run over all mouse files, calculate parameters in each file
# save them in a AnalysisInfo csv file:
# e.g., contact_1, contact_2, contact_3, etc.

work_dir = 'C:\\Users\\amird\\OneDrive\\Documents\\Berkeley\\Amir\\'

mouse = '6994_210_000'
mouse_short = mouse[0:4]

single_w_labels = [1,2,3,4,5]
single_w_trials = ['023','030','027','010','008']

# ========== get mean internal curvature =======
trials_to_take = mf.get_evan_included_trials(mouse)
pixels_after_to_fit = {3:5, 2: 5, 4: 15, 1: 10, 5:10}
whisker_order=[3,2,1,4,5]
date_DLC_files='Mar13'
iters_DLC_files=750000
n_points_fit=500

kappa_intenal_part1_mean, kappa_intenal_part2_mean, kappa_intenal_half_mean, err_trials =\
                mf.get_internal_kappa_from_all_trials(mouse, trials_to_take,whisker_order=whisker_order,
                                                      pixels_after_to_fit=pixels_after_to_fit,
                                                      date_DLC_files=date_DLC_files,iters_DLC_files=iters_DLC_files,
                                                      n_points_fit=n_points_fit)

# 1st time save
# path = work_dir + mouse_short + '\\'
# variable_name = 'kappa_intenal_part1_mean_' + mouse_short + datetime.now().strftime("%Y%m%d_%H_%M_%S")
# uf.save_obj(kappa_intenal_part1_mean,variable_name ,path)
#
# variable_name = 'kappa_intenal_part2_mean_' + mouse_short + datetime.now().strftime("%Y%m%d_%H_%M_%S")
# uf.save_obj(kappa_intenal_part2_mean,variable_name ,path)
#
# variable_name = 'kappa_intenal_half_mean_' + mouse_short + datetime.now().strftime("%Y%m%d_%H_%M_%S")
# uf.save_obj(kappa_intenal_half_mean,variable_name ,path)


# load
# path = work_dir + mouse_short + '\\'
# kappa_intenal_part1_mean = uf.load_obj('kappa_intenal_part1_mean_699420190314_03_27_42',path)
# kappa_intenal_part2_mean = uf.load_obj('kappa_intenal_part2_mean_699420190314_03_27_42',path)
# kappa_intenal_half_mean = uf.load_obj('kappa_intenal_half_mean_699420190314_03_27_42',path)


# ============= run over all files and extract parameters ================
kernel_size = [(5,5),(3,3),(3,3),(3,3),(3,3)]
n_iter = [2,2,2,2,2]
canny_low_th=90
canny_high_th=140
cut_below_x = [None,None,None,None,400]
cut_below_y = [200,250,300,320,None]
cut_above_x = [None,None,None,350,None]
cut_above_y = [None,None,None,None,None]

# piston points
a_points = [(339, 0),(116,2),(138,0),(564,0),(639,135)]
b_points = [(302, 138),(206,257),(175,143),(277,413),(527,311)]
all_5_piston_locs,all_5_pistons_subclass = vf.get_all_pistons_pixels(mouse,single_w_trials,single_w_labels,kernel_size=kernel_size,n_iter=n_iter,
                                                                     canny_low_th=canny_low_th,canny_high_th=canny_high_th,
                                                                     cut_below_x=cut_below_x,cut_below_y=cut_below_y,
                                                                     cut_above_x=cut_above_x,cut_above_y=cut_above_y,
                                                                     a_points=a_points,b_points =b_points ,subclassify=True,
                                                                     show=False)

# type_list= ['mean_angle','rng_angle','mean_kappa','num_contacts',
#             'num_contacts_above','num_contacts_below','num_touches',
#             'mean_deltaKappa_part1','mean_deltaKappa_part2',
#             'mean_deltaKappa_half','mean_deltaKappa_part1_external',
#             'mean_deltaKappa_part2_external','mean_deltaKappa_half_external']
type_list= ['num_contacts','num_contacts_above','num_contacts_below','num_touches']
s_contact_start = {3: 0.5, 2: 0.5,4: 0.5,1:0.5 , 5:0.5}
s_contact_end =  {3:1, 2:1, 4:1, 1:1 ,5:1}


external_kappa_internal_list = [kappa_intenal_part1_mean,kappa_intenal_part2_mean,kappa_intenal_half_mean]
parameters_dic,err_trials = mf.create_dic_parameters_of_listdics(mouse,all_5_piston_locs,
                                                                 all_5_pistons_subclass,type_list,
                                                                  trials_to_take,
                                                                  n_points_fit=n_points_fit,
                                                                  whisker_order=whisker_order,
                                                                  pixels_after_to_fit=pixels_after_to_fit,
                                                                  s_contact_start=s_contact_start,
                                                                  s_contact_end=s_contact_end,
                                                                  date_DLC_files=date_DLC_files,
                                                                  iters_DLC_files=iters_DLC_files,
                                                                  external_kappa_internal_list=external_kappa_internal_list)

path = work_dir + mouse_short + '\\'
variable_name = datetime.now().strftime("%Y%m%d_%H_%M_%S")

# 1st time save
# uf.save_obj(parameters_dic, variable_name,path)

# load
# parameters_dic = uf.load_obj(variable_name,path)

# =========== save to csv ============
# save listdics in csv
csv_name = work_dir + mouse_short + '\\' + mouse + '_exp_AnalysisInfo'
for i,type in enumerate(type_list):
    column_names = {i+1: type + '_' + str(i+1) for i in range(5)}
    listdic = mf.change_listdic_keys_to_column_names(parameters_dic[type],column_names)
    column_names_list = list(column_names.values())
    mf.insert_listdic_to_csv(csv_name,trials_to_take,listdic,column_names_list,iter=i)
