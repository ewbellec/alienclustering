import numpy as np
import pylab as plt

import ipywidgets as widgets
import os

from scipy.ndimage import median_filter, maximum_filter
from sklearn.cluster import DBSCAN

from plot_utilities import *

plt.rc('image', cmap='gray_r')


def check_path_create(path):
    if not os.path.exists(path):
        os.mkdir(path)
    return

###########################################################################################################################################
###########################################                   Preprocess data                   ###########################################
###########################################################################################################################################


def hot_pixel_filter(data,
                     threshold=1e2):
    '''
    Remove hot pixels (using a median filter) that can mess up the data preprocessing
    
    Parameters
    ----------
    :data: diffraction data
    :threshold: mask threshold used after a median filter. Mask pixels that are threshold times higher than neighboring pixels.
    
    Returns
    ----------
    :data_clean: data without hot pixels
    '''
    data_median = median_filter(data, size=2)
    mask = (data < threshold*(data_median+1))
    data_clean = data*mask
    return data_clean

def log_scale_matteotype_renormalization(data, 
                                         plot=False):
    '''
    rescaling of the data into a custom log scale. 
    Special attention is taken to the 0's. 
    resulting data_log is bounded 0 < data_log < 1
    
    Parameters
    ----------
    :plot: plot the preprocessing result
    
    Returns
    ----------
    :data_log: data in log scaling
    '''
    
    mask_zeros = (data==0.)
    
    data_log = np.copy(data)
    data_log = data_log.astype('float64')
    data_log[mask_zeros] = np.nan
    data_log = np.log(data_log)
    data_log = (data_log-np.nanmin(data_log))/(np.nanmax(data_log)-np.nanmin(data_log))
    data_log[mask_zeros] = 0.
    
    if plot:
        plot_3D_projections(data_log, fig_title='log renormalized data')
        
    return data_log

###########################################################################################################################################
#####################################                   Select intensity threshold                   ######################################
###########################################################################################################################################

def create_intensity_threshold_mask(data_log, threshold, 
                                    plot=False, data=None):
    '''
    Create a mask using a threshold in intensity
    
    Parameters
    ----------
    :data_log: rescaled data (from "log_scale_matteotype_renormalization" function)
    :threshold: intensity threshold for the mask (0 < threshold <1)
    :plot: plot the resulting mask
    :data: original non-rescaled data. Only used for the plot
    
    Returns
    ----------
    :mask: intensity threshold mask
    '''
    
    mask = (data_log> threshold )
    if plot:
        if data is not None:
            plot_3D_projections(data, log_scale=True, mask=mask)
        else:
            plot_3D_projections(data_log, mask=mask)
    return mask


def mask_smoothing(mask, 
                   size_filter=5, 
                   plot=False, data=None):
    '''
    Smooth the mask to merge fringes. 
    Only use this if the aliens are far enough from the Bragg peak.
    Otherwise the alien might be clustered with the central Bragg peak.
    Mask quality is better with a smoothing. Use it if possible.
    
    Parameters
    ----------
    :mask: intensity threshold mask
    :size_filter: size of the scipy.ndimage.maximum_filter filter. 
                  Should be around (slightly smaller) than the fringe spacing (in pixels).
                  If too large, the alien mask might touch the central Bragg and be clustered with it.
                  size_filter is around 2-5 for typical BCDI data.
    :plot: plot the resulting smoothed mask
    :data: original non-rescaled data. Only used for the plot
    
    Returns
    ----------
    :mask_smooth: smoothed mask
    '''
    
    mask_smooth = maximum_filter(mask, size=size_filter)
    if plot:
        plot_3D_projections(data, log_scale=True, mask=mask_smooth)
    return mask_smooth
    

def intensity_threshold_pixels(mask):
    '''
    Create pixels positions corresponding to the intensity threshold mask.
    
    Parameters
    ----------
    :mask: intensity threshold mask
    
    Returns
    ----------
    :pixels: pixels 3D position array with shape ( number of pixels covered by the mask, 3)
    '''
    
    z,y,x = np.indices(mask.shape)
    pixels = np.zeros((np.sum(mask),3))
    pixels[...,0] += z[mask==1]
    pixels[...,1] += y[mask==1]
    pixels[...,2] += x[mask==1]
        
    return pixels

###########################################################################################################################################
#####################################                           Clustering                           ######################################
###########################################################################################################################################


def pixels_clustering(pixels, 
                      eps=np.sqrt(2), min_samples=8, 
                      verbose=True,
                      plot=False):
    
    '''
    Make the clustering using sklearn.cluster.DBSCAN
    
    Parameters
    ----------
    :pixels: mask pixels from intensity_threshold_pixels function. shape ( number of pixels covered by the mask, 3)
    :eps, min_samples: parameters for DBSCAN
    :verbose: print informations    
    :plot: make a 3D scatter plot of the clusters
    Returns
    ----------
    :pixels_labels: cluster labels for each pixel
    
    '''
    
    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(pixels)
    
    pixels_labels = clustering.labels_
    
    if verbose:
        print('number of clusters : ', len(np.unique(pixels_labels)))
        
    if plot:
        plotly_scatter3D_clusters(pixels, pixels_labels)
        
    return pixels_labels


###########################################################################################################################################
#####################################                   Filtering out a cluster                      ######################################
###########################################################################################################################################

def filter_out_labels(pixels, pixels_labels,
                      labels_to_remove):
    '''
    Remove clusters having the label in labels_to_remove
    
    Parameters
    ----------
    :pixels: pixels position in the mask. shape ( number of pixels covered by the mask, 3)
    :pixels_labels: array of integer corresponding to the cluster label of each pixel. shape (number of points)
    :labels_to_remove: array of integer corresponding to the clusters' labels to be removed
    
    Returns
    ----------
    :pixels_clean: pixels without the removed labels
    :pixels_labels_clean: pixels_labels without the removed labels
    '''

    
    indexes = np.logical_not( np.isin(pixels_labels, labels_to_remove))
    pixels_clean = pixels[indexes]
    pixels_labels_clean = pixels_labels[indexes]
    
    return pixels_clean, pixels_labels_clean


###########################################################################################################################################
#####################################                    Remove central cluster                      ######################################
###########################################################################################################################################

def remove_central_peak_cluster(data, pixels, pixels_labels):
    '''
    Renove the central cluster since this one isn't an alien.
    Careful : The BCDI array should be centered on the Bragg peak for this function to work.
    
    Parameters
    ----------
    :data: The original data (only the shape is used. Not great coding, sorry)
    :pixels: clustered pixels
    :pixels_labels: array of integer corresponding to the cluster label of each pixel
    
    Returns
    ----------
    :pixels, pixels_labels: same as the input but without the central cluster
    
    '''
    label_central_peak = pixels_labels[np.where(np.all(pixels == np.round(np.array(data.shape)/2.),axis=1))[0][0]]
    
    pixels, pixels_labels = filter_out_labels(pixels, pixels_labels, label_central_peak)
    
    return pixels, pixels_labels


###########################################################################################################################################
#################################                    Keep only the largest clusters                      ##################################
###########################################################################################################################################

def get_clusters_size(pixels_labels):
    '''
    Compute size of each clusters
    
    Parameters
    ----------
    :pixels_labels: array of integer corresponding to the cluster label of each pixel
    
    Returns
    ----------
    :cluster_size: size of each cluster
    :labels: corresponding cluster labels
    
    '''
    labels = np.unique(pixels_labels)
    cluster_size = np.zeros(len(labels))
    for n in range(len(labels)):
        cluster_size[n] += np.sum(pixels_labels==labels[n])
    return cluster_size, labels

def keep_only_largest_clusters(pixels, pixels_labels,
                               nb_clusters_kept = 10,
                               plot=False):
    '''
    Remove smallest clusters
    
    Parameters
    ----------
    :pixels: clustered pixels
    :pixels_labels: array of integer corresponding to the cluster label of each pixel
    :nb_clusters_kept: number of clusters to be kept
    :plot: plot the cluster size
    Returns
    ----------
    :pixels, pixels_labels: same as the input but without the smallest clusters
    '''

    cluster_size, labels =  get_clusters_size(pixels_labels)
    indices_sort = np.argsort(cluster_size)
    cluster_size = cluster_size[indices_sort]
    labels = labels[indices_sort]
    labels_small = labels[:-nb_clusters_kept]
    
    pixels, pixels_labels = filter_out_labels(pixels, pixels_labels, labels_small)
       
    if plot:
        plt.figure()
        plt.plot(cluster_size, 'o')
        plt.axvline(x=len(labels)-nb_clusters_kept, color='r', linewidth=.5)
        plt.ylabel('cluster size', fontsize=15)
        
    return pixels, pixels_labels

###########################################################################################################################################
########################                      Using asymmetry matrix to sort clusters                          ############################
###########################################################################################################################################

def compute_asymmetry_matrix(data,
                             plot=False):
    '''
    compute assymetry array on the BCDI data
    
    Parameters
    ----------
    :data: 3D BCDI array
    :plot: plot the projection of the assymetry array
    
    Returns
    ----------
    :asym: assymetry 3D array
    '''
    data_inv = np.flip(data, axis=(0, 1, 2))
    asym = 2.*np.abs(data - data_inv)/(data + data_inv)
    asym[np.logical_or(data == 0, data_inv == 0)] = np.nan
    
    if plot:
        plot_3D_projections(asym, fig_title='asymmetry matrix')
    return asym

def compute_clusters_asymmetry(asym, pixels, pixels_labels):
    '''
    calculate the average assymetry for each cluster
    
    Parameters
    ----------
    :asym: assymetry 3D array
    :pixels: clustered pixels
    :pixels_labels: array of integer corresponding to the cluster label of each pixel
    
    Returns
    ----------
    :labels: array on labels
    :labels_asym: average assymetry for each clusters
    '''
    labels = np.unique(pixels_labels)
    labels_asym = np.zeros(len(labels))
    for n in range(len(labels)):
        labels_asym[n] += np.nanmean(asym[tuple(pixels[pixels_labels == labels[n]].astype('int').swapaxes(0,1))])
    return labels, labels_asym

###########################################################################################################################################
########################                          Other methods to sort clusters                               ############################
###########################################################################################################################################

def compute_clusters_max(data, pixels, pixels_labels):
    '''
    compute the maximum intensity valure inside each clusters
    
    Parameters
    ----------
    :data: 3D BCDI array
    :pixels: clustered pixels
    :pixels_labels: array of integer corresponding to the cluster label of each pixel
    
    Returns
    ----------
    :labels: array on labels
    :labels_max_intensity: maximum intensity inside each clusters
    '''
    labels = np.unique(pixels_labels)
    labels_max_intensity = np.zeros(len(labels))
    for n in range(len(labels)):
        labels_max_intensity[n] += np.nanmax(data[tuple(pixels[pixels_labels == labels[n]].astype('int').swapaxes(0,1))])
    return labels, labels_max_intensity


###########################################################################################################################################
#########################                      Cluster selection using 2D projections                          ############################
###########################################################################################################################################

def plot_2d_projection_cluster(data, pixels, pixels_labels, label, 
                               fig=None, ax=None):
    '''
    Plot 2D projection of the cluster "label"
    
    Parameters
    ----------
    :data: 3D BCDI array
    :pixels: clustered pixels
    :pixels_labels: array of integer corresponding to the cluster label of each pixel
    :label: label of the cluster to plot
    :fig, ax: figure and axes (created by default)
    
    '''
    mask = np.zeros(data.shape)
    for pix in pixels[pixels_labels == label].astype('int'):
        mask[tuple(pix)] += 1
    fig_title = 'label : {}'.format(label)
    plot_3D_projections(data, log_scale=True, mask=mask, fig_title=fig_title, fig=fig, ax=ax)
    return


def plot_clusters_select_2d_projections(data, pixels, pixels_labels, 
                                        sorting='size'):
    '''
    Plot clusters for user selection
    
    Parameters
    ----------
    :data: 3D BCDI array
    :pixels: clustered pixels
    :pixels_labels: array of integer corresponding to the cluster label of each pixel
    :sorting: sorting algorithm to plot the most important clusters first.
              Available algorithm are 'size', 'max', 'asym', 'None'
              'size' : sort cluster by number of pixels in it
              'max' : sort using the maximum pixel intensity in each cluster
              'asym' : sort using average asymmetry of each cluster
              'None' : no sorting
    
    Returns
    ----------
    :labels: a list of labels in pixels_labels
    :check_list: boolean array containing True for each label selected by the user
    
    '''
                
    if sorting=='asym' :
#         data_log = log_scale_matteotype_renormalization(data)
#         asym = compute_asymmetry_matrix(data_log)
        asym = compute_asymmetry_matrix(data)
        labels, labels_asym = compute_clusters_asymmetry(asym, pixels, pixels_labels)
        labels = labels[np.argsort(labels_asym)[::-1]]
    elif sorting=='max':
        labels, labels_max_intensity = compute_clusters_max(data, pixels, pixels_labels)
        labels = labels[np.argsort(labels_max_intensity)[::-1]]
    elif sorting=='size':
        cluster_size, labels = get_clusters_size(pixels_labels)
        labels = labels[np.argsort(cluster_size)[::-1]]
    elif sorting=='None':
        print('no sorting is used')
        labels = np.unique(pixels_labels)
    else:
        raise ValueError('sorting possible values are \'size\', \'max\', \'asym\' and \'None\' ')
    
    vbox = widgets.VBox()
    check_list = []
    fig, ax = plt.subplots(1,3, figsize=(12,4))
    for label in labels:

        plot_2d_projection_cluster(data, pixels, pixels_labels, label, fig=fig,ax=ax)
        fig.savefig('temp.jpg')

        file = open('temp.jpg', "rb")
        image = file.read()
        img = widgets.Image(value=image)

        check = widgets.Checkbox(description='select or not')
        check_list.append(check)
        vbox.children = tuple(list(vbox.children) + [img,check])
    display(vbox)
    fig.clf()
    os.remove('temp.jpg')

    return check_list, labels


###########################################################################################################################################
#######################                      Alien mask creation from selected clusters                          ##########################
###########################################################################################################################################

def create_mask_from_selected_clusters(data, check_list, pixels, pixels_labels, labels,
                                       plot=False):
    '''
    Create alien mask from clusters selected by the user
    
    Parameters
    ----------
    :data: 3D BCDI array
    :labels: a list of labels in pixels_labels
    :check_list: boolean array containing True for each label selected by the user
    :pixels: clustered pixels
    :pixels_labels: array of integer corresponding to the cluster label of each pixel
    :plot: if True, plot the resulting mask
    
    Returns
    ----------
    :mask_alien: final BCDI alien mask
    
    '''
    mask_alien = np.zeros(data.shape)
    for check,lab in zip(check_list, labels):
        if check.value:
            for pix in pixels[pixels_labels == lab].astype('int'):
                mask_alien[tuple(pix)] += 1
                
    if plot:
        plot_3D_projections(data, log_scale=True, mask=mask_alien, fig_title='data with mask')
                
    return mask_alien


###########################################################################################################################################
############################                          "Press button" functions                              ###############################
###########################################################################################################################################

def preprocessing(data, 
                  remove_hot_pixels=True, 
                  filter_small_values=True):
    '''
    Preprocessing "press-button" function
    
    Parameters
    ----------
    :data: 3D BCDI array
    :remove_hot_pixels: if True, remove hot pixels that could mess up the log rescaling
    :filter_small_values: if True, remove values <1 that could mess up the log rescaling
    
    Returns
    ----------
    :data: 3D BCDI array without hot pixels (if remove_hot_pixels = True)
    :data_log: log rescaled data from "log_scale_matteotype_renormalization" function
    
    '''
    
    # Remove hot pixels
    if remove_hot_pixels:
        data = hot_pixel_filter(data)
    
    # Filter out very small pixels values
    if filter_small_values:
        data[data<1] = 0.
    
    data_log = log_scale_matteotype_renormalization(data)
    return data, data_log



def clustering_and_filtering(mask, data,
                             nb_clusters_kept = 10,
                             sorting='size'):
    '''
    "press-button" function doing clustering, filtering and cluster plot for user selection
    
    Parameters
    ----------
    :mask: intensity threshold mask
    :data: 3D BCDI array
    :nb_clusters_kept: number of clusters to be kept for the plots
    :sorting: sorting algorithm to plot the most important clusters first.
              Available algorithm are 'size', 'max', 'asym', 'None'
              'size' : sort cluster by number of pixels in it
              'max' : sort using the maximum pixel intensity in each cluster
              'asym' : sort using average asymmetry of each cluster
              'None' : no sorting
    
    Returns
    ----------
    :pixels: clustered pixels (after filtering)
    :pixels_labels: array of integer corresponding to the cluster label of each pixel (after filtering)
    :labels: a list of labels in pixels_labels
    :check_list: boolean array containing True for each label selected by the user
    
    '''
    
    # Create pixels positions from intensity threshold mask
    pixels = intensity_threshold_pixels(mask)
    
    # Clustering
    eps = np.sqrt(2)
    min_samples = 8
    pixels_labels = pixels_clustering(pixels, 
                               eps=eps, min_samples=min_samples)
    
    # Filtering out central Bragg peak cluster
    pixels, pixels_labels = remove_central_peak_cluster(data, pixels, pixels_labels)
    
    # Remove "noise" DBSCAN label
    pixels, pixels_labels = filter_out_labels(pixels, pixels_labels, -1)
    
    # Filtering small clusters
    pixels, pixels_labels = keep_only_largest_clusters(pixels, pixels_labels,
                                                       nb_clusters_kept = nb_clusters_kept)
    
    print('number of clusters to plot :', len(np.unique(pixels_labels)))
    # Cluster selection 
    check_list, labels = plot_clusters_select_2d_projections(data, pixels, pixels_labels, 
                                                             sorting=sorting)
    
    return pixels, pixels_labels, check_list, labels



def get_clean_data(data_original, mask_alien, 
                   remove_hot_pixels=False,
                   replace_by_zeros=True,
                   plot=False, log_scale=True):
    '''
    Create clean data forcing 0's at the alien mask positions
    
    Parameters
    ----------
    :data_original: 3D BCDI array
    :mask_alien: final BCDI alien mask. Should be 1's at masked position and 0's everywhere else
    :replace_by_zeros: replace masked pixels by 0's
    :plot: if True, plot the result
    :log_scale: if True, plot the result in log scale
    
    Returns
    ----------
    :data_clean: 3D BCDI array with 0's at masked positions
    
    '''
    data_clean = np.copy(data_original)
    
    if remove_hot_pixels:
        data_clean = hot_pixel_filter(data_clean)
    
    if replace_by_zeros:
        data_clean[mask_alien==1.] = 0.
    else:
        data_clean = data_clean.astype('float64')
        data_clean[mask_alien==1.] = np.nan
    
    if plot:
        plot_3D_projections(data_original, log_scale=log_scale, fig_title='original data')
        plot_3D_projections(data_clean, log_scale=log_scale, fig_title='cleaned data')
    return data_clean