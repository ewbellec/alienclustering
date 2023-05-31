import numpy as np
import pylab as plt

plt.rc('image', cmap='gray_r')

###########################################################################################################################################
############################################                   Plot utilites                   ############################################
###########################################################################################################################################

def plot_3D_projections(data,
                        mask=None, alpha_mask=.3,
                        ax=None, fig=None,
                        cmap=None,
                        log_scale=False, 
                        fig_title=None):
    '''
    Make 3 figure of the 3-dimensional data projection along each axis. Possibility to overlay a mask.
    
    Parameters
    ----------
    :data: 3D scalar array
    :mask: 3D mask with 0 on unmasked pixels and 1 on masked pixels
    :alpha_mask: mask transparency
    :fig, ax: figure and axes (created by default)
    :cmap: data colormap
    :log_scale: apply log after projection
    :fig_title: figure suptitle
    '''
    
    if fig is None:
        fig, ax = plt.subplots(1,3, figsize=(12,4))
        
    for n in range(3):
        if log_scale:
            ax[n].imshow(np.log(np.nanmean(data, axis=n)),cmap=cmap, aspect='auto')
        else:
            ax[n].imshow(np.nanmean(data, axis=n),cmap=cmap, aspect='auto')
            
        if mask is not None :
            mask_plot = np.nanmean(mask, axis=n)
            mask_plot[mask_plot != 0.] = 1.
            ax[n].imshow( np.dstack([mask_plot, np.zeros(mask_plot.shape), np.zeros(mask_plot.shape), alpha_mask*mask_plot]), aspect='auto')
            
    if fig_title is not None:
        fig.suptitle(fig_title, fontsize=20)
        
    fig.tight_layout()
    return

def plot_3D_slice(data,
                  mask=None,
                  ax=None, fig=None,
                  cmap=None,
                  log_scale=False):
    '''
    Same as plot_3D_projections but with only slices at the center of the data array instead of projections along axis.
    '''
    
    if fig is None:
        fig, ax = plt.subplots(1,3, figsize=(12,4))
            
    for n in range(3):
        s = [slice(None) for n in range(3)]
        s[n] = data.shape[n]//2
    
        if log_scale:
            ax[n].matshow(np.log(data[s]),cmap=cmap, aspect='auto')
        else:
            ax[n].matshow(data[s],cmap=cmap, aspect='auto')

        if mask is not None :
            ax[n].matshow(mask[s],cmap='Reds', alpha=.7, aspect='auto')
        
    fig.tight_layout()
    return




###########################################################################################################################################
########################                  Plotly functions for a 3D scatter plot of the clusters                   ########################
###########################################################################################################################################

from matplotlib import cm
from matplotlib.colors import Normalize, to_hex
import plotly.express as px

def get_colors(arr, cmap='viridis'):
    '''
    transform an array of integers into an array of different colors
    
    Parameters
    ----------
    :arr: 1D array of integers
    :cmap: the colormap from which the colors are taken
    
    Returns
    ----------
    :colors: 1D array of same size as arr containing corresponding colors
    '''
    cmap = cm.get_cmap(cmap)
    n = len(np.unique(arr))
    colornorm = Normalize(vmin=1, vmax=n)
    hex_map = dict()
    for i, cl in enumerate(np.unique(arr)):
        hex_map[cl] = to_hex(cmap(colornorm(i + 1)))
    colors = list(map(lambda x: hex_map[x], arr))
    return colors

def plotly_scatter3D_clusters(pixels, pixels_labels,
                              Nb_pixels_partial = 10000,
                              marker_size=1.5):
    '''
    Make a clusters 3D scatter plot using plotly.express.scatter_3d and a different color for each cluster
    
    Parameters
    ----------
    :pixels: points positions with shape (number of points, 3)
    :pixels_labels: array of integer corresponding to the cluster label of each pixel. shape (number of points)
    :Nb_pixels_partial: number of pixels used in the plot. We don't use all of them for rapidity issues.
    :marker_size: size of the scattered points
    '''
    
    if Nb_pixels_partial is not None:
        if len(pixels)<Nb_pixels_partial:
            Nb_pixels_partial = len(pixels)
        rand_indices = np.random.choice(np.arange(pixels.shape[0]),Nb_pixels_partial)
        pixels_partial = pixels[rand_indices]
        pixels_labels_partial = pixels_labels[rand_indices]
        
    colors= get_colors(pixels_labels_partial, cmap='viridis')

    fig = px.scatter_3d(x=pixels_partial[...,0],y=pixels_partial[...,1],z=pixels_partial[...,2], color=colors)
    fig.update_traces(marker_size = marker_size)
    fig.show()
    
    return 