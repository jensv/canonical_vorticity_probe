from sys import version_info
#if version_info.major >= 3:
#    from importlib import reload
import matplotlib.pyplot as mp
import numpy as np
import numpy.linalg as nl
import MDSplus as mds


def get_data(name='probe3_01', shot=1210719051, tree='mst', comp='dave',
    t0=-0.015, dt=1E-6, tmin=-0.015, tmax=0.125, where=False):
    if comp == 'dave':
        # Works on computer with data
        shot = mds.Tree(tree, shot)
        node = shot.getNode('\\' + name)
    elif comp == 'juno':
        # Works on remote computer
        server = comp + '.physics.wisc.edu'
        conn = mds.Connection(server)
        conn.openTree(tree, shot)
        node = conn.get('\\' + name)
    y = node.data()
    x = node.dim_of().data()
    if comp == 'juno':
        x = t0 + dt*x
    if where:
        wh = np.where((x >= tmin) & (x < tmax))[0]
        y = y[wh]
        x = x[wh]
    return x, y


def return_smooth(x, n, axis=None, periodic=False, stype='mean'):
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
    if vtype == 'octahedron':
        n_build = 3
    elif vtype == 'tetrahedron': 
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
    if vtype == 'octahedron':
        x['ad'] = xbar[0] - xbar[3]
        x['be'] = xbar[1] - xbar[4]
        x['cf'] = xbar[2] - xbar[5]
        y['ad'] = ybar[0] - ybar[3]
        y['be'] = ybar[1] - ybar[4]
        y['cf'] = ybar[2] - ybar[5]
        z['ad'] = zbar[0] - zbar[3]
        z['be'] = zbar[1] - zbar[4]
        z['cf'] = zbar[2] - zbar[5]
    elif vtype == 'tetrahedron': 
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
    if vtype == 'octahedron':
        m = np.array((
            (x['ad'], y['ad'], z['ad']),
            (x['be'], y['be'], z['be']),
            (x['cf'], y['cf'], z['cf'])))
    elif vtype == 'tetrahedron': 
        m = np.array((
            (x['abcd'], y['abcd'], z['abcd']),
            (x['bcda'], y['bcda'], z['bcda']),
            (x['acbd'], y['acbd'], z['acbd'])))
    return m


def mach_results(shot=1210719051, comp='dave', smooth=None, delphi=30.0,
    t0=-0.015, dt=1E-6, tmin=-0.001, tmax=0.07, where=True,
    vtype='octahedron', imin=5E-4, k=1.1, ret=True, plot=False):
    """Enter azimuthal offset angle delphi in degrees.
    """
    isat = {}
    time = {}
    tip_abc_names = ['a', 'b', 'c', 'd', 'e', 'f']
    tip_num_names = [1, 4, 5, 2, 3, 6]
    tip_mds_names = ['probe3_' + str(n).zfill(2) for n in \
        tip_num_names]
    mst_names = ['theta', 'phi', 'r']  # ['x', 'y', 'z'] in Cartesian.
    for i in range(len(tip_abc_names)):
        abc = tip_abc_names[i]
        time[abc], isat[abc] = get_data(tip_mds_names[i], shot,
            comp=comp, t0=t0, dt=dt, tmin=tmin, tmax=tmax, where=where)
#        print(abc, np.min(time[abc]), np.max(time[abc]))
#        print(abc, np.min(isat[abc]), np.max(isat[abc]))
        if smooth is not None:
            isat[abc] = return_smooth(isat[abc], smooth)
        isat[abc] = np.maximum(isat[abc], imin)
    if (np.array_equal(time['a'], time['b'])
        and np.array_equal(time['a'], time['c'])
        and np.array_equal(time['a'], time['d'])
        and np.array_equal(time['a'], time['e'])
        and np.array_equal(time['a'], time['f'])):
        time = time['a']
    else:
        print('Time arrays not all equal')
        return None
    if vtype == 'octahedron':
        ratio = - 1.0/k*np.array((np.log(isat['a']/isat['d']),
            np.log(isat['b']/isat['e']), np.log(isat['c']/isat['f']))) 
    elif vtype == 'tetrahedron':
        ratio = - 1.0/k*np.array((
            np.log(isat['a']*isat['b']/isat['c']/isat['d']),
            np.log(isat['b']*isat['c']/isat['d']/isat['a']),
            np.log(isat['c']*isat['a']/isat['b']/isat['d'])
            )) 
    vel = np.dot(nl.inv(angle_matrix(vtype=vtype, rz=delphi)), ratio)
    results = {'t':time, 'r':vel[mst_names.index('r')],
        'theta':vel[mst_names.index('theta')],
        'phi':vel[mst_names.index('phi')]}
    if plot:
        mp.clf()
        mp.suptitle(' '.join([vtype.capitalize(), 'Mach probe, '])
            + str(shot))
        i = 0
        plot_names = ['r', 'theta', 'phi']
        for p in plot_names:
            mp.subplot(3, 1, i+1)
            mp.title(plot_names[i])
            mp.plot(results['t'], results[plot_names[i]])
            mp.grid(True)
            if i == len(plot_names) - 1:
                mp.xlabel('t [s]')
            else:
                mp.gca().axes.xaxis.set_ticklabels([])
            mp.ylabel('M')
            i += 1
    if ret:
        return results

