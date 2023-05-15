# Karsten McCollam, UW-Madison, 2021
# Jens von der Linden canonical helicity project on MST

from sys import version_info
# May not work for versions before 2.7...
#if version_info.major >= 3:
#    from importlib import reload
import matplotlib.pyplot as mp
import numpy as np
import numpy.linalg as nl
import MDSplus as mds


def get_data(name='probe3_01', shot=1210719051, tree='mst', comp='dave',
    #tmin=-0.015, tmax=0.125):
    tmin=None, tmax=None):
    server = comp + '.physics.wisc.edu'
    signalname = '\\' + name
    conn = mds.Connection(server)
    conn.openTree(tree, shot)
    node = conn.get(signalname)
    y = node.data()
    x = conn.get('DIM_OF(' + signalname + ')')
    if tmin is not None:
        wh = np.where(x >= tmin)[0]
        y = y[wh]
        x = x[wh]
    if tmax is not None:
        wh = np.where(x < tmax)[0]
        y = y[wh]
        x = x[wh]
    return x, y


def smooth(x, n, axis=None, periodic=False, stype='mean'):
    """Midpoint boxcar smooth of size n for input data x.  If an even n
    is given, this adds 1 to it for the smoothing number -- note this
    is different than IDL behavior (which doubles it and then adds 1).
    """
    n = int(n)
#    if n%2 == 0:
    if n == 1:
        return x
    if n//2 == n/2.0:
        n += 1
    y = np.zeros(shape=np.shape(x))
    if stype == 'mean':
        for i in range(-n//2+1, n//2+1):
            y = y + np.roll(x, i, axis=axis)
    elif stype == 'rms':
        for i in range(-n//2+1, n//2+1):
            y = y + np.roll(x, i, axis=axis)**2
    if stype == 'mean':
        y = y/n
    elif stype == 'rms':
        y = np.sqrt(y/n)
    if periodic == False:
        if x.ndim > 1:
            x = np.moveaxis(x, axis, 0)
            y = np.moveaxis(y, axis, 0)
        if stype == 'mean':
            for j in range(n//2):
                y[j] = np.mean(x[:2*j+1])
                y[-j-1] = np.mean(x[-2*j-1:])
        elif stype == 'rms':
            for j in range(n//2):
                y[j] = rms(x[:2*j+1])
                y[-j-1] = rms(x[-2*j-1:])
        if x.ndim > 1:
            return np.moveaxis(y, 0, axis)
    return y


def deg_to_rad(deg):
    return deg*np.pi/180.0


def rad_to_deg(rad):
    return rad*180.0/np.pi


def probe_angles(vtype='octahedron', delphi=0.0):
    """Basic Mach probe tip phi and theta angles in radians for
    different vtypes, 'octahedron' or 'tetrahedron'.
    Enter azimuthal offset angle delphi in degrees.
    """
    base_theta = np.array([54.73561, 125.26439])
    if vtype.startswith('octa'):
        n_build = 3
    elif vtype.startswith('tetra'): 
        n_build = 2
    theta = np.tile(base_theta, n_build)
    phi = 360.0*np.arange(2*n_build)/(2*n_build)
    phi += delphi
    return deg_to_rad(phi), deg_to_rad(theta)


def trig_functions(phi, theta):
    """The xbar, ybar, and zbar functions as an array given phi, theta.
    """
    xbar = np.sin(theta)*np.cos(phi)
    ybar = np.sin(theta)*np.sin(phi)
    zbar = np.cos(theta)
    return np.array((xbar, ybar, zbar))


def probe_xyz_values(vtype='octahedron', delphi=0.0):
    """Basic Mach probe tip xbar, ybar, zbar values as an array for
    different vtypes, 'octahedron' or 'tetrahedron'.
    Enter azimuthal offset angle delphi in degrees.
    """
    return trig_functions(*probe_angles(vtype, delphi))


def rotate_x(th):
    """Matrix for rotation around the x axis."""
    return np.array((
        (1.0, 0.0, 0.0),
        (0.0, np.cos(th), -np.sin(th)),
        (0.0, np.sin(th), np.cos(th))
        )) 


def rotate_y(th):
    """Matrix for rotation around the y axis."""
    return np.array((
        (np.cos(th), 0.0, np.sin(th)),
        (0.0, 1.0, 0.0),
        (-np.sin(th), 0.0, np.cos(th))
        )) 


def rotate_z(th):
    """Matrix for rotation around the z axis."""
    return np.array((
        (np.cos(th), -np.sin(th), 0.0),
        (np.sin(th), np.cos(th), 0.0), 
        (0.0, 0.0, 1.0)
        ))


def rotate_xyz(r, thx=0.0, thy=0.0, thz=0.0):
    """
    Rotate set of points first around x axis, then y, then z.
    Enter angles in degrees.
    """
    thx = deg_to_rad(thx)
    thy = deg_to_rad(thy)
    thz = deg_to_rad(thz)
    return np.dot(rotate_z(thz), np.dot(rotate_y(thy),
        np.dot(rotate_x(thx), r)))


def bar_differences(vtype='octahedron', rx=0.0, ry=0.0, rz=0.0):
    """Mach probe matrix element differences for different vtypes,
    'octahedron' or 'tetrahedron'.
    Enter rotation angles rx, ry, rz in degrees.
    """
    xyz = rotate_xyz(probe_xyz_values(vtype), rx, ry, rz)
    xbar, ybar, zbar = xyz[0], xyz[1], xyz[2]
    x, y, z = {}, {}, {}
    if vtype.startswith('octa'):
        x['ad'] = xbar[0] - xbar[3]
        x['be'] = xbar[1] - xbar[4]
        x['cf'] = xbar[2] - xbar[5]
        y['ad'] = ybar[0] - ybar[3]
        y['be'] = ybar[1] - ybar[4]
        y['cf'] = ybar[2] - ybar[5]
        z['ad'] = zbar[0] - zbar[3]
        z['be'] = zbar[1] - zbar[4]
        z['cf'] = zbar[2] - zbar[5]
    elif vtype.startswith('tetra'): 
        x['abcd'] = xbar[0] + xbar[1] - xbar[2] - xbar[3]
        x['bcda'] = xbar[1] + xbar[2] - xbar[3] - xbar[0]
        x['acbd'] = xbar[0] + xbar[2] - xbar[1] - xbar[3]
        y['abcd'] = ybar[0] + ybar[1] - ybar[2] - ybar[3]
        y['bcda'] = ybar[1] + ybar[2] - ybar[3] - ybar[0]
        y['acbd'] = ybar[0] + ybar[2] - ybar[1] - ybar[3]
        z['abcd'] = zbar[0] + zbar[1] - zbar[2] - zbar[3]
        z['bcda'] = zbar[1] + zbar[2] - zbar[3] - zbar[0]
        z['acbd'] = zbar[0] + zbar[2] - zbar[1] - zbar[3]
    return x, y, z
    

def angle_matrix(vtype='octahedron', rx=0.0, ry=0.0, rz=0.0):
    """Mach probe angle matrix for different vtypes, 'octahedron' or
    'tetrahedron'.
    Enter rotation angles rx, ry, rz in degrees.
    """
    x, y, z = bar_differences(vtype, rx, ry, rz)
    if vtype.startswith('octa'):
        m = np.array((
            (x['ad'], y['ad'], z['ad']),
            (x['be'], y['be'], z['be']),
            (x['cf'], y['cf'], z['cf'])))
    elif vtype.startswith('tetra'): 
        m = np.array((
            (x['abcd'], y['abcd'], z['abcd']),
            (x['bcda'], y['bcda'], z['bcda']),
            (x['acbd'], y['acbd'], z['acbd'])))
    return m


def merge_shots(time, time2, isat, isat2,
                missing_tip, missing_tip2, merge_method):
    r"""
    Merge data from two shots with the probe rotated 180 degrees.
    """
    time_merged = {}
    isat_merged = {}
    time_arrays_equal = True
    for key in list(isat.keys())[1:]:
        if np.array_equal(time[list(isat.keys())[0]], time[key]):
            continue
        time_arrays_equal = False
    if not (time_arrays_equal):
        print('Time arrays not all equal')
        return None
    for key in isat.keys():
        time_merged[key] = time[key]
        if ((merge_method == 'first') or (merge_method == 'average')) and (key == missing_tip):
            isat_merged[key] = isat2[key]
        elif ((merge_method == 'second') or (merge_method == 'average')) and (key == missing_tip2):
            isat_merged[key] = isat[key]
        else:
            if merge_method == 'average':
                isat_merged[key] = (isat[key] + isat2[key])/2
            if merge_method == 'first':
                isat_merged[key] = isat[key]
            if merge_method == 'second':
                isat_merged[key] = isat2[key]
    return time_merged, isat_merged


#shot=1210719051
def mach_results(shot=1210818030, nsmooth=None, res='mach',
    delphi=-60.0, tmin=-0.001, tmax=0.07, comp='dave',
    vtype='octahedron', imin=5E-4, k=1.1, second_shot=None,
    tetra_missing_tip=None, rj_set=[20, 20, 20, 20, 2],
    rj_set2 = [5, 20, 20, 20, 2],
    mds_start='probe3_', merge_method='first'):
    """Return result depending on res keyword .startswith:
        'isat', tip measurements;
        'rat', ratios of isats depending on vtype;
        'log', logarithms of ratios with factors;
        'mat', angle matrix depending on vtype;
        'inv', inverse of angle matrix;
        'mach', three vector components of M.
    vtype is vertex type, 'octahedron' or 'tetrahedron'.
    Enter azimuthal offset angle delphi in degrees.
    """
    isat = {}
    time = {}
    isat2 = {}
    time2 = {}
    if vtype.startswith('octa'):
        tip_abc_name = ['a', 'b', 'c', 'd', 'e', 'f', 'ret']
        tip_num_name = ['1', '4', '5', '2', '3', '6', '7']
    if vtype.startswith('tetra'):
        tip_abc_name = ['a', 'b', 'c', 'd', 'ret']
        #second_tip_abc_name = ['b', 'a', 'd', 'c', 'ret']
        tip_num_name = ['4', '1', '3', '2', '5']
        tip_num_name2 = ['3', '2', '4', '1', '5']
    rj = dict(zip(tip_abc_name,
         rj_set))
    rj2 = dict(zip(tip_abc_name,
          rj_set2))
    tip_mds_name = [mds_start + str(n).zfill(2) for n in \
        tip_num_name]   
    tip_mds_name2 = [mds_start + str(n).zfill(2) for n in \
        tip_num_name2]
    mst_name = ['theta', 'phi', 'r']  # ['x', 'y', 'z'] in Cartesian
    for i in range(len(tip_abc_name)):
        abc = tip_abc_name[i]
        time[abc], isat[abc] = get_data(tip_mds_name[i], shot,
            comp=comp, tmin=tmin, tmax=tmax)
        #
        # if second_shot assume it is with probe rotated by 180 degrees
        if second_shot:
            time2[abc], isat2[abc] = get_data(tip_mds_name2[i], second_shot,
                                              comp=comp, tmin=tmin, tmax=tmax)
            
            if nsmooth is not None:
                isat2[abc] = smooth(isat2[abc], nsmooth)
            isat2[abc] /= rj2[abc]
            isat2[abc] = np.maximum(isat2[abc], imin)

        if nsmooth is not None:
            isat[abc] = smooth(isat[abc], nsmooth)
        isat[abc] /= rj[abc]
        isat[abc] = np.maximum(isat[abc], imin)

    if second_shot:
        time, isat = merge_shots(time, time2, isat, isat2,
                                 tip_abc_name[tip_num_name.index(str(tetra_missing_tip))],
                                 tip_abc_name[tip_num_name2.index(str(tetra_missing_tip))],
                                 merge_method=merge_method)

        if merge_method == 'second':
            tip_num_name = tip_num_name2

    for i in range(len(tip_abc_name)):
        abc = tip_abc_name[i]
        if abc == 'a':
            isat['sum'] = np.zeros(np.shape(isat['a']))
        if abc != 'ret':
            isat['sum'] += isat[abc]
    
    time_arrays_equal = True
    for name in tip_abc_name[1:]:
        if np.array_equal(time['a'], time[name]):
            continue
        time_arrays_equal = False
    if (time_arrays_equal):
        time = time['a']
    else:
        print('Time arrays not all equal')
        return None
    if res.lower().startswith('isat'):
        isat['t'] = time
        isat['name'] = dict(zip(tip_num_name, tip_abc_name))
        isat['rj'] = rj
        isat['units'] = 'A'
        return isat
    if vtype.startswith('octa'):
        ratio = np.array((isat['a']/isat['d'], isat['b']/isat['e'],
            isat['c']/isat['f']))
        rat_name = ['a/d', 'b/e', 'c/f']
    elif vtype.startswith('tetra'):
        ratio = np.array((isat['a']*isat['b']/isat['c']/isat['d'],
            isat['b']*isat['c']/isat['d']/isat['a'],
            isat['c']*isat['a']/isat['b']/isat['d']))
        rat_name = ['a*b/c/d', 'b*c/d/a', 'c*a/b/d']
    if res.startswith('rat'):
        return {'rat':ratio, 't':time, 'name':rat_name, 'units':None}
    log = - 1.0/k*np.log(ratio)
    if res.startswith('log'):
        return {'log':log, 't':time, 'name':rat_name, 'units':None}
    mat = angle_matrix(vtype=vtype, rz=delphi)
    if res.startswith('mat'):
        return {'mat':mat, 'mst_name':mst_name, 'ratio_name':rat_name}
    inv = nl.inv(mat)
    if res.startswith('inv'):
        return {'inv':inv, 'mst_name':mst_name, 'ratio_name':rat_name}
    vel = np.dot(inv, log)
    if res.startswith('mach'):
        return {'t':time, 'r':vel[mst_name.index('r')],
            'theta':vel[mst_name.index('theta')],
            'phi':vel[mst_name.index('phi')], 'units':'M'}


def plot_mach_results(shot=1220927065, second_shot=1220927077, nsmooth=51,
                      res='mach', delphi=-60.0, tmin=-0.001, tmax=0.07, comp='dave',
                      vtype='tetra', imin=5E-4, k=1.1, plot_sum=True,
                      tetra_missing_tip=4, mds_start='probe5_',
                      rj_set=[20, 20, 20, 20, 5], rj_set2=[20, 20, 20, 5, 5],
                      merge_method='first'):
    """Plot result depending on res keyword .startswith:
        'isat', tip measurements;
        'rat', ratios of isats depending on vtype;
        'log', logarithms of ratios with factors;
        'mach', three vector components of M.
    For other res keywords, just print data.
    vtype is vertex type, 'octahedron' or 'tetrahedron'.
    Enter azimuthal offset angle delphi in degrees.
    """
    #import mpl_toolkits.axes_grid.anchored_artists as maa
    # This looks easier:
    import matplotlib.offsetbox as mo
    data = mach_results(shot,
                        nsmooth, res, delphi, tmin, tmax, comp,
                        vtype, imin, k,
                        second_shot=second_shot,
                        tetra_missing_tip=tetra_missing_tip,
                        rj_set=rj_set,
                        rj_set2=rj_set2,
                        mds_start=mds_start, merge_method=merge_method)
    do_plot = True
    if res.lower().startswith('isat'):
        plot_name = list(data['name'].keys())
        plot_name.sort()
        plot_ylabel = 'I [' + data['units'] + ']'
        plot_data = {k: data[data['name'][k]]
            for k in data['name'].keys()}
    elif res.startswith('rat') or res.startswith('log'):
        plot_name = data['name']
        plot_ylabel = None
        plot_data = {k: data[res[:3]][data['name'].index(k)]
            for k in data['name']}
    elif res.startswith('mach'):
        plot_name = ['r', 'theta', 'phi']
        plot_ylabel = 'v [' + data['units'] + ']'
        plot_data = data
    else:
        do_plot = False
        for k in data.keys():
            print(k, data[k])
    if do_plot:
        mp.clf()
        mp.suptitle(', '.join([
            ' '.join([vtype.capitalize(), 'Mach probe']), str(shot)]))
        for i, p in enumerate(plot_name):
            mp.subplot(len(plot_name), 1, i+1)
            mp.plot(data['t'], plot_data[p], label=p)
                # Label only used if legend is called, see below.)
            mp.grid(True)
            if plot_ylabel is not None:
                mp.ylabel(plot_ylabel)
            if i < len(plot_name) - 1:
                mp.gca().axes.add_artist(mo.AnchoredText(p,
                    loc='upper right'))
                mp.gca().axes.xaxis.set_ticklabels([])
            else:
                if res.startswith('isat'):
                    if plot_sum:
                        #mp.plot(data['t'], data['sum'], alpha=0.5)
                        mp.plot(data['t'], data['sum'], label='sum')
                        mp.legend(loc='right')
                    else:
                        mp.gca().axes.add_artist(mo.AnchoredText(p,
                            loc='upper right'))
                else:
                    mp.gca().axes.add_artist(mo.AnchoredText(p,
                        loc='upper right'))
                mp.xlabel('t [s]')
    mp.show()    

