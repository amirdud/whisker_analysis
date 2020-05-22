import numpy as np
from matplotlib import cm
import matplotlib.pyplot as plt
from scipy import interpolate
from numpy.linalg import norm
import time
import pickle
from datetime import datetime

def scale(v, min_old, max_old, min_new, max_new):
    # scale values to new range
    v_new = (max_new - min_new) / (max_old - min_old) * ( v - max_old ) + max_new
    return v_new

def ismember(x,y,type=True):
    is_np_x = isinstance(x, np.ndarray)
    is_np_y = isinstance(y, np.ndarray)

    is_list_x = isinstance(x, list)
    is_list_y = isinstance(y, list)

    is_range_x = isinstance(x,range)
    is_range_y = isinstance(y,range)

    is_int_x = isinstance(x, int)
    is_int_y = isinstance(y, int)

    if all([is_np_x,is_np_y]):
        xl = x.tolist()
        yl = y.tolist()
    elif is_range_x:
        xl = list(x)
        yl = y
    elif is_range_y:
        yl = list(y)
        xl = x
    elif all([is_list_x,is_list_y]):
        xl = x
        yl = y
    elif all([is_int_x, is_np_y]):
        xl = [x]
        yl = y.tolist()
    elif all([is_np_x, is_int_y]):
        xl = x.tolist()
        yl = [y]

    # ismember
    if type:
        inds_in_x = [i for i, x in enumerate(xl) if x in yl]

    # is not member
    elif ~type:
        inds_in_x = [i for i, x in enumerate(xl) if x not in yl]

    if all([is_np_x,is_np_y]):
        inds_in_x = np.array(inds_in_x)

    return inds_in_x


def colors_255_to_1(colors_255):
    num_colors = len(colors_255)

    colors_1 = []
    for i in range(num_colors):
        colors_1_i = tuple(np.flip(np.array(colors_255[i])[0:3] / 255))
        colors_1.append(colors_1_i)

    return colors_1

def colors_1_to_255(colors_1):
    num_colors = len(colors_1)

    colors_255 = []
    for i in range(num_colors):
        colors_255_i = tuple(np.flip(np.array(cm.tab10(i))[0:3]*255))
        colors_255.append(colors_255_i)

    return colors_255

def get_colors(num_colors = 5,type = 'lines'):
    colors = []
    for i in range(num_colors):
        if type == 'gradual':
            color = np.array([0, 0, 0])+i/num_colors
            rgb = tuple(color* 255)
            colors.append(rgb)
        elif type == 'lines':
            # convert to RGB
            rgb = tuple(np.flip(np.array(cm.tab10(i))[0:3]*255))
            colors.append(rgb)

    return colors

def show_plus(ax):
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    ax.tick_params(labelsize=20)
    ax.legend(fontsize=20)
    plt.show()

def define_curvature(p1,p2,p3,scaling_factor=60,alg = 'derivative'):
    '''returns curvature from 3 points
            p1,p2,p3: tuples
            scaling_factor: pxls/mm
                the number of pixels in 1 mm
            kappa:    int               '''

    if alg == 'radius':

        p_cen,r = define_circle(p1, p2, p3)
        # uf.draw_line_from_points(p1,p2,p3)

        if p_cen is not None:
            circ_direction = isabove(np.array(p_cen), np.array(p1), np.array(p3))

            if circ_direction:
                r = r * (-1)

        kappa = 1/r

    elif alg == 'derivative':
        pts = np.vstack((np.array(p1),np.array(p2),np.array(p3)))
        dx_dt = np.gradient(pts[:,0])
        dy_dt = np.gradient(pts[:,1])
        d2x_dt2 = np.gradient(dx_dt)
        d2y_dt2 = np.gradient(dy_dt)

        kappa = ((d2x_dt2 * dy_dt - dx_dt * d2y_dt2) / (dx_dt * dx_dt + dy_dt * dy_dt) ** 1.5)[1]

    kappa = kappa*scaling_factor

    return kappa


def define_circle(p1, p2, p3):
    """
    Returns the center and radius of the circle passing the given 3 points.
    In case the 3 points form a line, returns (None, infinity).

    """
    temp = p2[0] * p2[0] + p2[1] * p2[1]
    bc = (p1[0] * p1[0] + p1[1] * p1[1] - temp) / 2
    cd = (temp - p3[0] * p3[0] - p3[1] * p3[1]) / 2
    det = (p1[0] - p2[0]) * (p2[1] - p3[1]) - (p2[0] - p3[0]) * (p1[1] - p2[1])

    if abs(det) < 1.0e-12:
        return (None, np.inf)

    # Center of circle
    cx = (bc*(p2[1] - p3[1]) - cd*(p1[1] - p2[1])) / det
    cy = ((p1[0] - p2[0]) * cd - (p2[0] - p3[0]) * bc) / det

    radius = np.sqrt((cx - p1[0])**2 + (cy - p1[1])**2) # in pixels

    return ((cx, cy), radius)


def fit_curve(pts_array,n_points = 100):
    x = pts_array[:,0]
    y = pts_array[:,1]
    p,ss,_,_,_ = np.polyfit(x, y, deg=2,full=True)

    x_new = np.linspace(x.min(),x.max(),num = n_points,endpoint=True)
    y_new = np.polyval(p,x_new)
    return x_new,y_new,ss

def arc_length(x, y):
    npts = len(x)
    arc = np.sqrt((x[1] - x[0])**2 + (y[1] - y[0])**2)
    for k in range(1, npts):
        arc = arc + np.sqrt((x[k] - x[k-1])**2 + (y[k] - y[k-1])**2)

    return arc

def tri_area(a, b, c):
    area = 0.5 * norm( np.cross( b-a, c-a ) )
    return area

def minpass(mymarks, mypass):
    large_number = 1e10
    min_value = large_number # something bigger than your maximum value
    for x in mymarks:
        if x < min_value and x >= mypass:
            min_value = x

    if min_value == large_number:
        min_value = None

    return min_value

def concat_vals_once_in_2_dics(a,b):
    '''a & b must have the same keys'''
    c={}
    for key in a:
        a_vals,b_vals = a[key], b[key]
        c_vals = list(set(a_vals + b_vals))
        c[key] = sorted(c_vals)

    return c

def get_unequal_vals_in_2_dics(a,b):
    '''a & b must have the same keys
        a: many values
        b: subvalues that appear in a'''
    c = {}
    ind_shared = {}
    for key in a:
        a_vals,b_vals = a[key], b[key]
        c_vals = list(set(a_vals).difference(set(b_vals)))
        c[key] = c_vals

        ind_shared[key] = ismember(a_vals,b_vals,False)

    return c,ind_shared

def find_nearest(x,target,type='below'):
    '''find the closest value and its index above/below
        a certain number'''
    x = np.array(x)

    if type == 'above':
        inidices = np.where(x >= target)[0]
        val = np.min(x[inidices])
        idx = np.argmin(x[inidices])
    elif type == 'below':
        inidices = np.where(x <= target)[0]
        val = np.max(x[inidices])
        idx = np.argmax(x[inidices])

    return val,idx

def isabove(p,a,b):
    ''' a,b: points that define 2 edges of a line
        p:   point

    # see also:
    https://stackoverflow.com/questions/45766534/finding-cross-product-to-find-points-above-below-a-line-in-matplotlib
    https://math.stackexchange.com/questions/274712/calculate-on-which-side-of-a-straight-line-is-a-given-point-located
'   '''

    isabove_bool = np.cross(p-a, b-a) < 0
    return isabove_bool

def draw_line_from_points(*args):
    x = []
    y = []
    for p in args:
        x.append(p[0])
        y.append(p[1])

    plt.plot(x,y,'*-')
    plt.show()

def save_obj(variable, variable_name,path):
    with open(path + variable_name + '.pkl', 'wb') as f:
        pickle.dump(variable, f, pickle.HIGHEST_PROTOCOL)

def load_obj(variable_name,path):
    with open(path + variable_name + '.pkl', 'rb') as f:
        return pickle.load(f)