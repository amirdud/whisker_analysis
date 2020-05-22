import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy import interpolate
from scipy import stats

import vid_funs as vf
import util_funs as uf
import whisker_funs as wf
import trial_funs as trf
import mouse_funs as mf

matplotlib.use('Qt5Agg') # for interactive

# ========== pre-process ==============
mouse = '6994_210_000'
mouse_short = mouse[0:4]

trial = '083'

# map whiskers to pistons
single_w_labels = [1,2,3,4,5]
single_w_trials = ['023','030','027','010','008']

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

# ========== get tracking info ==============
wt_labels = np.array([3,2,1,4,5])
colors = uf.get_colors(5)
wt_color = colors
whisker_order = [3,2,1,4,5]

frames,data,probs = trf.get_DLC_data(mouse,trial,date = 'Mar13',iters = 750000,whisker_order=whisker_order,
                                     convert_to_int=False)

pixels_after_to_fit = {3:5, 2: 5, 4: 15, 1: 10, 5:10}

frames_trace,data_trace,score_dic = trf.fit_DLC_data(data,type = 'ransac',
                                                     min_samples_for_reg=4,
                                                     residual_threshold=None,
                                                     pixels_after_to_fit=pixels_after_to_fit,
                                                     th_score_high=0.2,
                                                     th_score_low=-0.1)

# for tracing improvement:
# 1. retraining on specific frames (gamma is out of range) or get rid of those using running threshold
# 2. correct in wf the angle function where no slope is detected
# 3. too curved whisker problems in trial 595
# 4. correct straight part red whisker touches problems in trial 595
# 5. correct "whisker is out" trials such as 673

# ========== extract parameters from tracing ==============
# angle, curvature and arc
s_start, s_end = 0, 0.3
start_frame_internal = 0
stop_frame_internal = 300
alg = 'derivative'

deltaKappa = trf.calc_data_curvature(data_trace,frames_trace,s_start=s_start ,s_end=s_end,
                                     start_frame_internal=start_frame_internal,
                                     stop_frame_internal= stop_frame_internal,
                                     external_kappa_internal=None,
                                     type='delta_kappa',alg=alg)

kappa = trf.calc_data_curvature(data_trace, frames_trace, s_start=s_start, s_end=s_end,type='kappa',alg=alg)
arcs = trf.calc_arc_length(data_trace)
angles = trf.calc_angle(data_trace)

frames_trace_f,data_trace_f,(deltaKappa_f,kappa_f,arcs_f,angles_f) = trf.filter_data_by_parameter(frames_trace,data_trace,deltaKappa,kappa,arcs,angles)
frames_interp,data_interp = trf.fill_DLC_data(frames_trace_f,data_trace_f)

start_frame_internal = 0
stop_frame_internal = 30
deltaKappa_interp = trf.calc_data_curvature(data_interp,frames_interp,s_start ,s_end,
                                            start_frame_internal=start_frame_internal,
                                            stop_frame_internal= stop_frame_internal,
                                            external_kappa_internal=None,
                                            type='delta_kappa',alg=alg)
kappa_interp = trf.calc_data_curvature(data_interp,frames_interp,s_start ,s_end,type='kappa',alg=alg)
arcs_interp = trf.calc_arc_length(data_interp)
angles_interp = trf.calc_angle(data_interp)

# w = 5
# plt.plot(frames_interp[w],deltaKappa_interp[w]);plt.show()
# plt.plot(frames_interp[w],kappa_interp[w]);plt.show()
# plt.plot(frames_interp[w],arcs_interp[w]);plt.show()
# plt.plot(frames_interp[w],angles_interp[w]);plt.show()

# contacts & touches
s_contact_start = {3: 0.5, 2: 0.5,4: 0.5,1:0.5 , 5:0.5}
s_contact_end =  {3:1, 2:1, 4:1, 1:1 ,5:1}

pistons_in_trial = trf.get_which_pistons_in_trial(mouse,trial)
pistons_in_trial_locs,pistons_in_trial_subclass = trf.get_trial_piston_locs(all_5_piston_locs,all_5_pistons_subclass,pistons_in_trial)
pxls_for_contact = trf.get_pxls_for_contact(data_interp,pistons_in_trial,s_contact_start,s_contact_end)
tips = trf.get_data_tip(data_interp,get_tips_to = pistons_in_trial)

contacts,contacts_above,contacts_below = trf.get_contacts(pxls_for_contact,frames_interp,pistons_in_trial_locs,pistons_in_trial_subclass)
num_contacts = trf.calc_num_contacts(contacts)
num_contacts_above = trf.calc_num_contacts(contacts_above)
num_contacts_below = trf.calc_num_contacts(contacts_below)

# calculate first touches
touches = trf.calc_touches(contacts,tolerance=3)
num_touches = trf.calc_num_contacts(touches)

# w = 1
# ind = w-1
# plt.eventplot(touches[w])
# plt.show()


# show barplot of contacts
whiskers = np.array(list(num_contacts.keys()))
num_contacts_array = np.array(list(num_contacts.values())).T

colors_1 = uf.colors_255_to_1(colors)
pistons_in_trial_inds = np.array(pistons_in_trial) -1
colors_1_reduced = [ colors_1[i] for i in pistons_in_trial_inds]

# fig, ax = plt.subplots()
# ax.bar(whiskers, num_contacts_array, color=colors_1_reduced, alpha=0.5)
# plt.ylabel('# Contact Frames', fontsize=20)
# plt.ylim([0,300])
# uf.show_plus(ax)

record_name_input = 'trial_' + trial + 'for_figure'
vf.show_video_plus(mouse,trial,frame_rate = 30,
                   wt_labels = wt_labels, wt_frames = frames_interp,
                   wt_data = data_interp,wt_ref_frames=None,wt_ref = None,
                   wt_mark_frames = contacts ,wt_color = wt_color,p_whisker = w,
                   p_name = 'angle',p_data = angles_interp[w],
                   record = True,record_name_input = record_name_input)

# work_dir = 'C:\\Users\\amird\\OneDrive\\Documents\\Berkeley\\Amir\\' # global variable
# file_name = work_dir + mouse_short + '\\Videos' + '\\' + mouse + '_0' + trial + '.avi'
# frame_number = 60
# vf.save_frame_from_video(file_name,frame_number)

# ========== extract tracing statistics ========
bins = range(90,180,5)

names = mf.get_whisker_names(mouse)
fig,ax = plt.subplots()
for i,ind in enumerate(pistons_in_trial_inds):
    plt.hist(angles[ind+1],bins,label=names[ind] ,color = colors_1_reduced[i],alpha=0.5)

plt.xlabel('Angle (degree)',fontsize = 20)
plt.ylabel('Counts',fontsize = 20)
uf.show_plus(ax)

# parameters_to_add_to_whisker_table
mean_angle = trf.get_statistic_parameter(angles_interp,'mean')
rng_angle = trf.get_statistic_parameter(angles_interp,'range')
mean_deltaKappa = trf.get_statistic_parameter(deltaKappa_interp,'mean')
