import cv2
import matplotlib.pyplot as plt
import numpy as np
import time
import whisker_funs as wf
import util_funs as uf
from datetime import datetime

work_dir = 'C:\\Users\\amird\\OneDrive\\Documents\\Berkeley\\Amir\\' # global variable

def show_video(mouse,trial,frame_rate = 50,
               record=False,record_name_input=None):
    mouse_short = mouse[0:4]
    file_name = work_dir + mouse_short + '\\Videos' + '\\' + mouse + '_0' + trial + '.avi'
    cap = cv2.VideoCapture(file_name)

    if record:
        if record_name_input is None:
            str_time = datetime.now().strftime("%Y%m%d_%H_%M_%S")
            record_name = mouse + '_0' + trial + '_' + str_time
        else:
            record_name = record_name_input

        record_full_name = record_name + '.avi'
        output_vid = cv2.VideoWriter(record_full_name, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), frame_rate, (640, 512))

    # Read until video is completed
    while (cap.isOpened()):

        # Capture frame-by-frame
        ret, frame = cap.read()

        if ret == True:
            # Display frame
            cv2.imshow('Frame', frame)
            time.sleep(1/frame_rate)

            if record:
                output_vid.write(frame)

            # Press Q on keyboard to  exit
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break

        # Break the loop
        else:
            break

    # release the video capture object
    cap.release()
    if record:
        output_vid.release()

    # Closes all the frames
    cv2.destroyAllWindows()


def show_video_plus(mouse,trial,frame_rate = 50,
                    wt_labels=None,wt_frames=None,wt_data=None,wt_ref=None,
                    wt_ref_frames = None,wt_mark_frames=None,
                    wt_color=(214, 182, 69),wt_thick=2,p_whisker=None,p_name=None,
                    p_data=None,general_mark_frames=None,record=False,
                    record_name_input = None):

    mouse_short = mouse[0:4]
    file_name = work_dir + mouse_short + '\\Videos' + '\\' + mouse + '_0' + trial + '.avi'
    cap = cv2.VideoCapture(file_name)

    if record:
        if record_name_input is None:
            str_time = datetime.now().strftime("%Y%m%d_%H_%M_%S")
            record_name = mouse + '_0' + trial + '_' + str_time
        else:
            record_name = record_name_input

        record_full_name = record_name + '.avi'
        output_vid = cv2.VideoWriter(record_full_name, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), frame_rate, (640, 512))

    count_frame = 0
    count_wt_frame = np.zeros(len(wt_labels)).astype(int)
    count_wt_ref_frame = np.zeros(len(wt_labels)).astype(int)

    # Read until video is completed
    while (cap.isOpened()):

        # Capture frame-by-frame
        ret, img = cap.read() # frame is a 2D numpy array

        if ret == True:
            # get img shape in the first image
            if count_frame == 0:
                h, w, _ = img.shape

                if p_whisker is not None:
                    # define max & min pixels values
                    max_new = 0  # upper pixel
                    min_new = h / 4

            # run over whiskers
            for key in wt_labels:
                ind = key-1

                # get whisker tracing info
                if wt_data is not None:
                    wt_data_i = wt_data[key]
                    wt_frames_i = wt_frames[key]

                if wt_ref is not None:
                    wt_ref_frames_i = wt_ref_frames[key]
                    wt_ref_i = wt_ref[key]

                wt_color_i = wt_color[ind]

                if wt_mark_frames is not None:
                    if key in wt_mark_frames:
                        wt_mark_frames_i = wt_mark_frames[key]

                # start with plotting the parameter curve
                if p_whisker == key:
                    # calculate parameter location
                    if p_name == 'angle':
                        min_old = p_data.min()
                        max_old = p_data.max()
                    elif p_name == 'curvature':
                        min_old = p_data.min() #-0.002
                        max_old = p_data.max() #0.002

                    p_data_sc = np.interp(p_data, (min_old, max_old), (min_new, max_new))
                    # p_data_sc = uf.scale(p_data,min_old,max_old, min_new,max_new)
                    p_range = np.arange(len(wt_frames_i))
                    p_frames_sc = np.arange(w - p_range.size, w, 1)
                    p_pts = np.vstack((p_frames_sc, p_data_sc)).astype(np.int32).T

                    # Plot parameter
                    cv2.polylines(img, [p_pts], False, wt_color_i)

                    # plot Progress line of parameter if exists
                    if p_whisker == key:
                        mov_line_pt1 = (p_frames_sc[count_wt_frame[ind]].astype(np.int32), np.min(p_data_sc).astype(np.int32))
                        mov_line_pt2 = (p_frames_sc[count_wt_frame[ind]].astype(np.int32), np.max(p_data_sc).astype(np.int32))
                        cv2.line(img, mov_line_pt1,mov_line_pt2, color=(255, 255, 255))


                # plot whisker tracing data if exists
                if wt_data is not None:
                    if count_frame in wt_frames_i:
                        # Trace whisker location
                        wt_data_mat = wt_data_i[count_wt_frame[ind]]
                        cv2.polylines(img, [wt_data_mat.astype(np.int32)], False, wt_color_i,thickness=wt_thick)
                        count_wt_frame[ind] = count_wt_frame[ind] + 1

                if wt_ref is not None:
                    if count_frame in wt_ref_frames_i:
                        # whisker ref location
                        wt_ref_mat = wt_ref_i[count_wt_ref_frame[ind]]
                        for ref_pt in wt_ref_mat:
                            cv2.circle(img, tuple(ref_pt.astype(np.int32)), 5,wt_color_i, -1)

                        count_wt_ref_frame[ind] = count_wt_ref_frame[ind] + 1


                # plot whisker marking frames if exists
                if wt_mark_frames is not None:
                    if key in wt_mark_frames:
                        if count_frame in wt_mark_frames_i:
                            cv2.circle(img, (30*key,30) , 12, wt_color_i,-1)

                # mark frames
                if general_mark_frames is not None:
                    if count_frame in general_mark_frames:
                        cv2.circle(img, (500, 500), 12, (255, 255, 255), -1)

                # add frame number
                cv2.putText(img, str(count_frame), (10, 500), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

            # Display the resulting frame
            cv2.imshow('Frame', img)

            if frame_rate != -1:
                time.sleep(1/frame_rate)

            count_frame = count_frame + 1

            if record:
                output_vid.write(img)

            # Press Q on keyboard to  exit
            # skip frames 1 by 1:
            if frame_rate == -1:
                if cv2.waitKey(-1) & 0xFF == ord('q'):
                    break
            else:
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        # Break the loop
        else:
            break

    # release the video capture object
    cap.release()
    if record:
        output_vid.release()

    # Closes all the frames
    cv2.destroyAllWindows()

def z_proj_movie(mouse,trial,statistic_name = 'avg',show = False):
    # statistic_name: 'avg','med'
    mouse_short = mouse[0:4]
    file_name = work_dir + mouse_short + '\\Videos' + '\\' + mouse + '_0' + trial + '.avi'

    # loading the video
    cap = cv2.VideoCapture(file_name)
    ret, frame = cap.read()
    h, w, _ = frame.shape

    # establish a while loop for reading all the video frames
    count_frames = 0

    # accumulator in double precision
    if statistic_name=='avg':
        avg = np.zeros((h, w), dtype=np.float64)
    elif statistic_name=='med':
        medstack = []

    while (cap.isOpened()):
        ret, frame = cap.read()
        if ret == True:

            gray_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            if statistic_name=='avg':
                avg = np.add(avg, gray_img)
            elif statistic_name == 'med':
                medstack.append(gray_img)

            count_frames += 1

        # Break the loop
        else:
            break

    # release the video capture object
    cap.release()

    if statistic_name == 'avg':
        avg = np.divide(avg, count_frames)
        z_proj = avg

    elif statistic_name == 'med':
        med = np.median(medstack, axis=0)
        z_proj = med

    # change type:
    z_proj = z_proj.astype(np.uint8)

    # show projection
    if show:
        plt.imshow(z_proj)
        plt.show()

    return z_proj

# apply threshold to image
def apply_thrshold_img(img,th = 40,max_pixel = 255,th_type = 'binary_inv',show = False):
    if th_type == 'binary_inv':
        cv2_th_type = cv2.THRESH_BINARY_INV
    elif th_type == 'trunc':
        cv2_th_type = cv2.THRESH_TRUNC

    _, th_img = cv2.threshold(img, th, max_pixel, cv2_th_type )

    if show:
        plt.imshow(th_img)
        plt.show()

    return th_img

def apply_dilation(bin_img,kernel_size = (2,2),method = 'close',n_iter=2, show = False):
    # input must be binary (add check)
    kernel = np.ones(kernel_size,np.uint8)
    if method=='open':
        cv2_method = cv2.MORPH_OPEN
    elif method =='close':
        cv2_method = cv2.MORPH_CLOSE

    bin_method = cv2.morphologyEx(bin_img, cv2_method, kernel, iterations=n_iter)
    dil_img = cv2.dilate(bin_method,kernel,iterations=n_iter)

    if show:
        plt.imshow(dil_img)
        plt.show()

    return dil_img

def apply_dilation_difference(th_img,kernel_size = (2,2),method='close',n_iter=5, show = False):
    kernel = np.ones(kernel_size, np.uint8)
    if method=='open':
        cv2_method = cv2.MORPH_TOPHAT
    elif method =='close':
        cv2_method = cv2.MORPH_BLACKHAT

    bin_img = cv2.morphologyEx(th_img, cv2_method, kernel, iterations=n_iter)

    if show:
        plt.imshow(bin_img)
        plt.show()

    return bin_img

def show_img_plus_mask(img,mask):
    # mask should be binary (add check)
    new_img = cv2.bitwise_and(img, img, mask=mask)
    plt.imshow(new_img)
    plt.show()


def get_one_piston_pixels(mouse,trial,whisker_label,kernel_size=(5,5),n_iter=5,
                          canny_low_th=0,canny_high_th=40,cut_below_x = None,
                          cut_below_y = None,cut_above_x = None,cut_above_y = None,
                          dilate = True, show=False):
    # input:
    #   mouse: which mouse are you referring to
    #   trial: a list of trial numbers (in strings) in which single
    #       piston appears. e.g., [02,04]
    #   whisker_label: a list of whisker_label that are stimulated by the
    #       piston (respectively)
    #   kernel_size: how sensitive you want the contact detection to be.
    #       larger kernel -> less sensitive
    #   n_iter: another parameter for contact sensitivity detection.
    #       larger number -> less sensitive

    # To get the relevant trials and whisker_label:
    # 1. find the single whisker trials
    # 2. run each one of them in the video (slowly), with one whisker at a time
    #   and understand which whisker is relevant for each whisker label.
    # 3. enter those lists respectively

    med_img = z_proj_movie(mouse, trial, statistic_name='med', show=False)

    # use Canny edge detection algorithm:
    # - Noise reduction (gaussion blur).
    # - Edge detection: Sobel kernel to find edge gradients.
    # - Values that were considered edges and do not serve as local
    #   maximum are suppressed.
    # - Min and Max values are used for detecting edges. Values
    #   in between are determined according to their connectivity with
    #   sure or non edges

    edges = cv2.Canny(med_img,canny_low_th,canny_high_th)
    if dilate:
        dil_edges = apply_dilation(edges, kernel_size=kernel_size, method='close', n_iter=n_iter,show=False)
    else:
        dil_edges = edges.copy()

        # condition on x and y
    if cut_below_x and cut_below_y is not None:
        dil_edges[cut_below_y:-1, 0:cut_below_x] = 0
    if cut_below_x and cut_above_y is not None:
        dil_edges[0:cut_above_y, 0:cut_below_x] = 0
    if cut_above_x and cut_below_y is not None:
        dil_edges[cut_below_y:-1, cut_above_x:-1] = 0
    if cut_above_x and cut_above_y is not None:
        dil_edges[0:cut_above_y, cut_above_x:-1] = 0

    # condition only on x or y
    if cut_below_x and not (cut_above_y or cut_below_y)  is not None:
        dil_edges[:,0:cut_below_x]=0
    if cut_below_y and not (cut_above_x or cut_below_x) is not None:
        dil_edges[cut_below_y:-1,:]=0
    if cut_above_x and not (cut_above_y or cut_below_y) is not None:
            dil_edges[:, cut_above_x:-1] = 0
    if cut_above_y and not (cut_above_x or cut_below_x) is not None:
            dil_edges[0:cut_above_y,:] = 0

    msk = dil_edges.copy()
    msk_pixels_y, msk_pixels_x = np.where(msk)
    msk_pixels = np.vstack((msk_pixels_x ,msk_pixels_y )).T

    if show:
        show_img_plus_mask(med_img, msk)
        # plt.imshow(msk)
        plt.show()

    return msk_pixels


def get_all_pistons_pixels(mouse,trials,wt_labels,
                           kernel_size=[(5,5),(5,5),(5,5),(5,5),(5,5)],
                           n_iter=[2,2,2,2,2],
                           canny_low_th=0,
                           canny_high_th=40,
                           cut_below_x = [None,None,None,None,None],
                           cut_below_y = [None,None,None,None,None],
                           cut_above_x = [None,None,None,None,None],
                           cut_above_y = [None,None,None,None,None],
                           a_points = None,
                           b_points = None,
                           subclassify = False,
                           dilate = True,
                           show=False):
    '''
    a_points: list of tuples
              first point of each line defining a separator between above/below piston
    b_points: list of tuples
              last point of each line defining a separator between above/below piston
    '''

    pistons_locs = {}
    pistons_subclass = {}

    for i,tr in enumerate(trials):
        msk_pixels = get_one_piston_pixels(mouse,tr,wt_labels[i],kernel_size=kernel_size[i],n_iter=n_iter[i],
                                           canny_low_th=canny_low_th,canny_high_th=canny_high_th,
                                           cut_below_x = cut_below_x[i],cut_below_y = cut_below_y[i],
                                           cut_above_x=cut_above_x[i], cut_above_y=cut_above_y[i],
                                           dilate = dilate, show=show)

        if subclassify:
            # subclassify pixels to above/below
            # msk_pixels_subclassified = subclass_pixels(msk_pixels)

            a = np.array(a_points[i])
            b = np.array(b_points[i])

            c = subclass_pixels(msk_pixels, a, b)

            pistons_subclass[wt_labels[i]] = c

        pistons_locs[wt_labels[i]] = msk_pixels

    if show:
        # fig, ax = plt.subplots(111)
        for key in pistons_locs:
            plt.scatter(pistons_locs[key][:,0],pistons_locs[key][:,1],label=str(key))

        plt.legend(fontsize=18,loc='lower right')
        plt.gca().invert_yaxis()
        plt.show()

        if subclassify:
            for key in pistons_locs:
                plt.scatter(pistons_locs[key][:, 0], pistons_locs[key][:, 1], c = pistons_subclass[wt_labels[key-1]])

            plt.legend(fontsize=18, loc='lower right')
            plt.gca().invert_yaxis()
            plt.show()

    return pistons_locs,pistons_subclass

def subclass_pixels(msk_pixels,a,b,show=False):

    c = uf.isabove(msk_pixels,a,b)

    if show:
        plt.scatter(msk_pixels[:, 0], msk_pixels[:, 1], c=c, cmap="bwr", vmin=0, vmax=1)
        plt.show()

    return c
