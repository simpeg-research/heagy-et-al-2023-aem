import numpy as np
import matplotlib.pyplot as plt

def streamplot_tree(
    mesh, data, normal='y', slice_location='C', range_x1=None, range_x2=None,
    nx1=128, nx2=128, ax=None, pcolor_opts=None, streamplot_opts=None
):
    if ax is None:
        ax = plt.gca()
    normal = normal.lower()
    if range_x1 is None:
        if normal == 'x':
            range_x1 = mesh.nodes_y[[0, -1]]
        else:
            range_x1 = mesh.nodes_x[[0, -1]]
    if range_x2 is None:
        if normal == 'z':
            range_x2 = mesh.nodes_y[[0, -1]]
        else:
            range_x2 = mesh.nodes_z[[0, -1]]
    
    if slice_location == 'C':
        nodes = getattr(mesh, f'nodes_{normal}')
        slice_location = 0.5*(mesh.nodes_x[0] + mesh.nodes_x[-1])
    
    grid_x1 = np.linspace(*range_x1, nx1)
    grid_x2 = np.linspace(*range_x2, nx2)
    X1, X2, X3 = np.meshgrid(grid_x1, grid_x2, [slice_location])
    if normal == 'x':
        stack = [X3, X1, X2]
    elif normal == 'y':
        stack = [X1, X3, X2]
    else:
        stack = [X1, X2, X3]
    slice_grid = np.stack(stack, axis=-1).reshape((-1, 3))
    X1 = X1[..., 0]
    X2 = X2[..., 0]
    X3 = X3[..., 0]
    
    if len(data) == mesh.n_faces:
        d_type = 'faces'
    else:
        d_type = 'edges'
        
    if normal != 'y':
        # then need y component
        interp = mesh.get_interpolation_matrix(slice_grid, d_type+"_y")
        dat_y = interp @ data
    else:
        dat_y = None
    if normal != 'x':
        # then need x component
        interp = mesh.get_interpolation_matrix(slice_grid, d_type+"_x")
        dat_x = interp @ data
    else:
        dat_x = None
    if normal != 'z':
        # then need z component
        interp = mesh.get_interpolation_matrix(slice_grid, d_type+"_z")
        dat_z = interp @ data
    
    if normal == 'x':
        dat = [dat_y.reshape(X1.shape), dat_z.reshape(X1.shape)]
    elif normal == 'y':
        dat = [dat_x.reshape(X1.shape), dat_z.reshape(X1.shape)]
    else:
        dat = [dat_x.reshape(X1.shape), dat_y.reshape(X1.shape)]
    dat_cc = getattr(mesh, f'average_{d_type[:-1]}_to_cell_vector') @ data
    dat_cc = dat_cc.reshape((-1, 3), order='F')
    dat_norm = np.linalg.norm(dat_cc, axis=-1)
        
    if pcolor_opts is None:
        pcolor_opts = {}
    if streamplot_opts is None:
        streamplot_opts = {}
    
    im0, = mesh.plot_slice(
        dat_norm, v_type='CC', normal=normal, slice_loc=slice_location, ax=ax, range_x=range_x1, range_y=range_x2, pcolor_opts=pcolor_opts
    )
    im1 = ax.streamplot(X1, X2, *dat, **streamplot_opts)
    return im0, im1