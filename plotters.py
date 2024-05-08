import os
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
tfd = tfp.distributions
tfb = tfp.bijectors
from timeit import default_timer as timer
import matplotlib
import matplotlib.pyplot as plt
from fpdf import FPDF
from PIL import Image
#import Trainer_2 as Trainer
#import Metrics_old as Metrics
from statistics import mean,median
import matplotlib.lines as mlines
from GMetrics.corner import corner, extend_corner_range, get_1d_hist
from GMetrics.utils import HPD_intervals, HPD_quotas, get_CI_from_sigma

def make_pdf_from_img(img):
    """Make pdf from image
    Used to circumvent bud in plot_model which does not allow to export pdf"""
    img_pdf = os.path.splitext(img)[0]+".pdf"
    cover = Image.open(img)
    width, height = cover.size
    pdf = FPDF(unit = "pt", format = [width, height])
    pdf.add_page()
    pdf.image(img, 0, 0)
    pdf.output(img_pdf, "F")

def plot_corners(dist_1,
                 dist_2,
                 max_points = 50_000,
                 max_dim = 32,
                 n_bins = 50,
                 w1 = None,
                 w2 = None,
                 sigma_contours = None,
                 show_intervals_1d = True,
                 extend_range_percent = 0,
                 title1 = None,
                 title2 = None,
                 color1 = 'green',
                 color2 = 'red',
                 plot_title = "Params contours",
                 title_kwargs = None,
                 legend_labels = None,
                 legend_kwargs = None,
                 figdir = None,
                 figname = None,
                 save = False,
                 show = True):
    start = timer()
    try:
        tf.random.set_seed(0)
        np.random.seed(0)
        print("Sampling from dist_1...")
        start_samp1 = timer()
        samp_1 = dist_1.sample(max_points).numpy()
        end_samp1 = timer()
        print(f"Sampling from dist_1 done in {end_samp1-start_samp1} s.")
    except:
        samp_1 = dist_1
    try:
        print("Sampling from dist_2...")
        start_samp2 = timer()
        samp_2 = dist_2.sample(max_points).numpy()
        end_samp2 = timer()
        print(f"Sampling from dist_2 done in {end_samp2-start_samp2} s.")
    except:
        samp_2 = dist_2
    shape_1 = samp_1.shape
    shape_2 = samp_2.shape
    if shape_1 != shape_2:
        raise ValueError("The two samples have different shapes.")
    else:
        shape = shape_1
    samp_1_no_nans = samp_1[~np.isnan(samp_1).any(axis=1), :]
    samp_2_no_nans = samp_2[~np.isnan(samp_2).any(axis=1), :]
    if len(samp_1) != len(samp_1_no_nans):
        print("Points in sample_1 containing nan have been removed. The fraction of nans over the total samples was:", str((len(samp_1)-len(samp_1_no_nans))/len(samp_1)),".")
    if len(samp_2) != len(samp_2_no_nans):
        print("Points in sample_2 containing nan have been removed. The fraction of nans over the total samples was:", str((len(samp_2)-len(samp_2_no_nans))/len(samp_2)),".")
    samp_1 = samp_1_no_nans[:shape[0]]
    samp_2 = samp_2_no_nans[:shape[0]]
    sigma_contours = sigma_contours if sigma_contours else [1, 2, 3]
    labels = []
    for i in range(1,shape[1]+1):
        labels.append(r"$\mathbf{x}_{%d}$" % i)
        i = i+1
    thin = int(shape[1]/max_dim)+1
    if thin<=2:
        thin = 1
    samp_1 = samp_1[:, ::thin]
    samp_2 = samp_2[:, ::thin]
    nndim = samp_1.shape[1]
    print("Computing HPD intervals...")
    start_hpdi = timer()
    HPI_intervals1 = [HPD_intervals(samp_1[:,i], 
                                    intervals = get_CI_from_sigma(sigma_contours), # type: ignore
                                    weights = w1, 
                                    nbins = n_bins, 
                                    print_hist = False, 
                                    reduce_binning = True) for i in range(nndim)]
    HPI_intervals2 = [HPD_intervals(samp_2[:,i], 
                                    intervals = get_CI_from_sigma(sigma_contours), # type: ignore
                                    weights = w2, 
                                    nbins = n_bins, 
                                    print_hist = False, 
                                    reduce_binning = True) for i in range(nndim)]
    end_hpdi = timer()
    print(f"HPD intervals computed in {end_hpdi-start_hpdi} s.")
    print("Computing HPD quotas...")
    start_hpdq = timer()
    levels1 = np.array([[np.sort(HPD_quotas(samp_1[:,[i,j]], 
                                            nbins = n_bins, 
                                            intervals = get_CI_from_sigma(sigma_contours), # type: ignore
                                            weights = w1)).tolist() for j in range(nndim)] for i in range(nndim)]) # type: ignore
    levels2 = np.array([[np.sort(HPD_quotas(samp_2[:,[i,j]], 
                                            nbins = n_bins, 
                                            intervals = get_CI_from_sigma(sigma_contours), # type: ignore
                                            weights = w2)).tolist() for j in range(nndim)] for i in range(nndim)]) # type: ignore
    end_hpdq = timer()
    print(f"HPD quotas computed in {end_hpdq-start_hpdq} s.")
    labels = list(np.array(labels)[::thin])
    ranges = extend_corner_range(samp_1,
                                 samp_2,
                                 ilist = None, 
                                 percent = extend_range_percent)
    linewidth = 1.3
    fig, axes = plt.subplots(nndim, nndim, figsize=(3*nndim, 3*nndim))
    print("Plotting corner plot...")
    figure1 = corner(samp_1, bins=n_bins, weights=w1, labels=[r"%s" % s for s in labels], fig=fig, max_n_ticks=6, color=color1, plot_contours=True, smooth=True, smooth1d=True, range=ranges, plot_datapoints=True, plot_density=False, fill_contours=False, normalize1d=True,
                     hist_kwargs={'color': color1, 'linewidth': '1.5'}, label_kwargs={'fontsize': 16}, show_titles=False, title_kwargs={"fontsize": 18}, levels_lists=levels1, data_kwargs={"alpha": 1}, contour_kwargs={"linestyles": ["dotted", "dashdot", "dashed"][:len(HPI_intervals1[0])], "linewidths": [linewidth, linewidth, linewidth][:len(HPI_intervals1[0])]},
                     no_fill_contours=False, contourf_kwargs={"colors": ["white", "lightgreen", color1], "alpha": 1})  # , levels=(0.393,0.68,))
    figure2 = corner(samp_2, bins=n_bins, weights=w2, labels=[r"%s" % s for s in labels], fig=fig, max_n_ticks=6, color=color2, plot_contours=True, smooth=True, range=ranges, smooth1d=True, plot_datapoints=True, plot_density=False, fill_contours=False, normalize1d=True,
                     hist_kwargs={'color': color2, 'linewidth': '1.5'}, label_kwargs={'fontsize': 16}, show_titles=False, title_kwargs={"fontsize": 18}, levels_lists=levels2, data_kwargs={"alpha": 1}, contour_kwargs={"linestyles": ["dotted", "dashdot", "dashed"][0:len(HPI_intervals1[0])], "linewidths": [linewidth, linewidth, linewidth][:len(HPI_intervals1[0])]},
                     no_fill_contours=False, contourf_kwargs={"colors": ["white", "tomato", color2], "alpha": 1})  
    axes = np.array(figure1.axes).reshape((nndim, nndim))
    for i in range(nndim):
        ax = axes[i, i]
        title = ""
        ax.grid(True, linestyle='--', linewidth=1, alpha=0.3)
        ax.tick_params(axis='both', which='major', labelsize=16)
        if show_intervals_1d:
            HPI681 = HPI_intervals1[i][0][1]
            HPI951 = HPI_intervals1[i][1][1]
            HPI3s1 = HPI_intervals1[i][2][1]
            HPI682 = HPI_intervals2[i][0][1]
            HPI952 = HPI_intervals2[i][1][1]
            HPI3s2 = HPI_intervals2[i][2][1]
            hists_1d_1 = get_1d_hist(i, samp_1, nbins=n_bins, ranges=ranges,
                                     weights=w1, normalize1d=True)[0]
            hists_1d_2 = get_1d_hist(i, samp_2, nbins=n_bins, ranges=ranges,
                                     weights=w2, normalize1d=True)[0]
            for j in HPI3s1:
                ax.axvline(hists_1d_1[0][hists_1d_1[0] >= j[0]][0],
                        color=color1, alpha=1, linestyle=":", linewidth=linewidth)
                ax.axvline(hists_1d_1[0][hists_1d_1[0] <= j[1]][-1],
                        color=color1, alpha=1, linestyle=":", linewidth=linewidth)
            for j in HPI3s2:
                ax.axvline(hists_1d_2[0][hists_1d_2[0] >= j[0]][0],
                           color=color2, alpha=1, linestyle=":", linewidth=linewidth)
                ax.axvline(hists_1d_2[0][hists_1d_2[0] <= j[1]][-1],
                           color=color2, alpha=1, linestyle=":", linewidth=linewidth)
            for j in HPI951:
                ax.axvline(hists_1d_1[0][hists_1d_1[0] >= j[0]][0],
                           color=color1, alpha=1, linestyle="-.", linewidth=linewidth)
                ax.axvline(hists_1d_1[0][hists_1d_1[0] <= j[1]][-1],
                           color=color1, alpha=1, linestyle="-.", linewidth=linewidth)
            for j in HPI952:
                ax.axvline(hists_1d_2[0][hists_1d_2[0] >= j[0]][0],
                           color=color2, alpha=1, linestyle="-.", linewidth=linewidth)
                ax.axvline(hists_1d_2[0][hists_1d_2[0] <= j[1]][-1],
                           color=color2, alpha=1, linestyle="-.", linewidth=linewidth)
            for j in HPI681:
                ax.axvline(hists_1d_1[0][hists_1d_1[0] >= j[0]][0],
                           color=color1, alpha=1, linestyle="--", linewidth=linewidth)
                ax.axvline(hists_1d_1[0][hists_1d_1[0] <= j[1]][-1],
                           color=color1, alpha=1, linestyle="--", linewidth=linewidth)
                title1 = title1 if title1 else "Sample 1"
                title = title + title1 + ": ["+'{0:1.2e}'.format(j[0])+","+'{0:1.2e}'.format(j[1])+"]"
            title = title+"\n"
            for j in HPI682:
                ax.axvline(hists_1d_2[0][hists_1d_2[0] >= j[0]][0],
                           color=color2, alpha=1, linestyle="--", linewidth=linewidth)
                ax.axvline(hists_1d_2[0][hists_1d_2[0] <= j[1]][-1],
                           color=color2, alpha=1, linestyle="--", linewidth=linewidth)
                title2 = title2 if title2 else "Sample 2"
                title = title + title2 + ": ["+'{0:1.2e}'.format(j[0])+","+'{0:1.2e}'.format(j[1])+"]"
        if i == 0:
            x1, x2, _, _ = ax.axis()
            ax.set_xlim(x1*1.3, x2)
        if show_intervals_1d:
            ax.set_title(title, fontsize=12)
    for yi in range(nndim):
        for xi in range(yi):
            ax = axes[yi, xi]
            if xi == 0:
                x1, x2, _, _ = ax.axis()
                ax.set_xlim(x1*1.3, x2)
            ax.grid(True, linestyle='--', linewidth=1)
            ax.tick_params(axis='both', which='major', labelsize=16)
    plt.tight_layout()
    title_kwargs = title_kwargs if title_kwargs else {"fontsize": 26}
    fig.suptitle(r'%s'%plot_title, **title_kwargs)
    colors = [color1,color2,'black', 'black', 'black']
    red_patch = matplotlib.patches.Patch(color=colors[0])#, label='The red data')
    blue_patch = matplotlib.patches.Patch(color=colors[1])#, label='The blue data')
    line1 = matplotlib.lines.Line2D([0], [0], color=colors[0], lw=12)
    line2 = matplotlib.lines.Line2D([0], [0], color=colors[1], lw=12)
    line3 = matplotlib.lines.Line2D([0], [0], color=colors[2], linewidth=3, linestyle='--')
    line4 = matplotlib.lines.Line2D([0], [0], color=colors[3], linewidth=3, linestyle='-.')
    line5 = matplotlib.lines.Line2D([0], [0], color=colors[4], linewidth=3, linestyle=':')
    lines = [line1,line2,line3,line4,line5]
    legend_kwargs = legend_kwargs if legend_kwargs else {"fontsize": 26, "loc": (0.65,0.8)}
    fig.legend(lines, legend_labels, **legend_kwargs)#, bbox_transform=ax.transAxes)
    if save:
        figdir = figdir if figdir else "."
        figname = figname if figname else "cornerplot.png"
        figure_path = os.path.join(figdir, figname)
        _, file_extension = os.path.splitext(figname)
        save_kwargs = {}
        if file_extension.lower() in ['.jpg', '.jpeg', '.png']:
            save_kwargs['pil_kwargs'] = {'quality': 50}
        plt.savefig(figure_path, **save_kwargs)
    if show:
        plt.show()
    plt.close()
    end = timer()
    print("Plot done and saved in", end-start, "s.")
 
def cornerplotter(dist_1,
                  dist_2,
                  path_to_plots,
                  figure_name = "corner_plot.png",
                  max_points = 50_000,
                  max_dim = 32, 
                  n_bins = 50,
                  title = None,
                  corner_kwargs = {},
                  colors = ["red", "blue"],
                  show = False,
                  save = True
                 ) -> None:
    try:
        tf.random.set_seed(0)
        np.random.seed(0)
        samp_1 = dist_1.sample(max_points).numpy()
    except:
        samp_1 = dist_1
    try:
        samp_2 = dist_2.sample(max_points).numpy()
    except:
        samp_2 = dist_2
    shape_1 = samp_1.shape
    shape_2 = samp_2.shape
    if shape_1 != shape_2:
        raise ValueError("The two samples have different shapes.")
    else:
        shape = shape_1
    samp_1_no_nans = samp_1[~np.isnan(samp_1).any(axis=1), :]
    samp_2_no_nans = samp_2[~np.isnan(samp_2).any(axis=1), :]
    if len(samp_1) != len(samp_1_no_nans):
        print("Points in sample_1 containing nan have been removed. The fraction of nans over the total samples was:", str((len(samp_1)-len(samp_1_no_nans))/len(samp_1)),".")
    if len(samp_2) != len(samp_2_no_nans):
        print("Points in sample_2 containing nan have been removed. The fraction of nans over the total samples was:", str((len(samp_2)-len(samp_2_no_nans))/len(samp_2)),".")
    samp_1 = samp_1_no_nans[:shape[0]]
    samp_2 = samp_2_no_nans[:shape[0]]
    labels = []
    for i in range(1,shape[1]+1):
        labels.append(r"$\mathbf{x}_{%d}$" % i)
        i = i+1
    thin = int(shape[1]/max_dim)+1
    if thin<=2:
        thin = 1
    samp_1 = samp_1[:, ::thin]
    samp_2 = samp_2[:, ::thin]
    ndims_eff = samp_1.shape[1]
    labels = list(np.array(labels)[::thin])
    line_1 = mlines.Line2D([], [], color = colors[0], label = '$\mathbf{X}_{1}$')
    line_2 = mlines.Line2D([], [], color = colors[1], label = '$\mathbf{X}_{2}$')
    corner_kwargs_1 = dict(corner_kwargs)
    corner_kwargs_2 = dict(corner_kwargs)
    if "data_kwargs" in corner_kwargs.keys():
        data_kwargs_bkp = dict(corner_kwargs["data_kwargs"])
    figure = corner(samp_1, 
                           color = colors[0],
                           bins = n_bins,
                           labels = [r"%s" % s for s in labels],
                           normalize1d = True,
                           **corner_kwargs_1)
                           #**{"plot_density": False, "plot_contours": False, "data_kwargs": {"alpha": 0.01}})
    if "data_kwargs" in corner_kwargs.keys():
        corner_kwargs_2["data_kwargs"] = data_kwargs_bkp
    corner(samp_2,
                  color = colors[1],
                  bins = n_bins,
                  fig = figure, 
                  normalize1d = True,
                  **corner_kwargs_2)
    plt.legend(handles = [line_1, line_2], 
               bbox_to_anchor = (-ndims_eff+1.8, ndims_eff+.3, ndims_eff-1, 0.) ,
               fontsize='xx-large')
    if title:
        figure.suptitle(title, fontsize=20)
    if save:
        figure_path = os.path.join(path_to_plots,figure_name)
        _, file_extension = os.path.splitext(figure_name)
        save_kwargs = {}
        if file_extension.lower() in ['.jpg', '.jpeg', '.png']:
            save_kwargs['pil_kwargs'] = {'quality': 50}
        plt.savefig(figure_path, **save_kwargs)
    if show:
        plt.show()
    plt.close()
    return

def marginal_plot(target_test_data,sample_nf,path_to_plots,ndims):
    n_bins=50
    if ndims<=4:
        fig, axs = plt.subplots(int(ndims/4), 4, tight_layout=True)
        for dim in range(ndims):
            column=int(dim%4)
            axs[column].hist(target_test_data[:,dim], bins=n_bins,density=True,histtype='step',color='red')
            axs[column].hist(sample_nf[:,dim], bins=n_bins,density=True,histtype='step',color='blue')
            x_axis = axs[column].axes.get_xaxis()
            x_axis.set_visible(False)
            y_axis = axs[column].axes.get_yaxis()
            y_axis.set_visible(False)
    elif ndims>=100:
        fig, axs = plt.subplots(int(ndims/10), 10, tight_layout=True)
        for dim in range(ndims):
            row=int(dim/10)
            column=int(dim%10)
            axs[row,column].hist(target_test_data[:,dim], bins=n_bins,density=True,histtype='step',color='red')
            axs[row,column].hist(sample_nf[:,dim], bins=n_bins,density=True,histtype='step',color='blue')
            x_axis = axs[row,column].axes.get_xaxis()
            x_axis.set_visible(False)
            y_axis = axs[row,column].axes.get_yaxis()
            y_axis.set_visible(False)
    else:
        fig, axs = plt.subplots(int(ndims/4), 4, tight_layout=True)
        for dim in range(ndims):
            row=int(dim/4)
            column=int(dim%4)
            axs[row,column].hist(target_test_data[:,dim], bins=n_bins,density=True,histtype='step',color='red')
            axs[row,column].hist(sample_nf[:,dim], bins=n_bins,density=True,histtype='step',color='blue')
            x_axis = axs[row,column].axes.get_xaxis()
            x_axis.set_visible(False)
            y_axis = axs[row,column].axes.get_yaxis()
            y_axis.set_visible(False)
    fig.savefig(path_to_plots+'/marginal_plot.pdf',dpi=300)
    fig.clf()
    return
   
def plot_corr_matrix(dist: np.ndarray,
                     path_to_plots,
                     figure_name = "corre_matrix_plot.pdf",
                     max_points = 10_000,
                     title = None,
                     show_labels = True,
                     show = False,
                     save = True
                    ) -> None:
    """
    """
    try:
        tf.random.set_seed(0)   
        np.random.seed(0)
        samp = dist.sample(max_points).numpy()
    except:
        samp = dist
    shape = samp.shape
    labels = []
    for i in range(1,shape[1]+1):
        labels.append(r"$\mathbf{x}_{%d}$" % i)
        i = i+1
    #df: pd.DataFrame = pd.DataFrame(samp, columns=labels)
    #f: plt.Figure = plt.figure(figsize=(18, 18))
    #plt.matshow(df.corr(), fignum=f.number)
    #cb = plt.colorbar()
    #plt.grid(False)

    df = pd.DataFrame(samp, columns=labels)

    # Create figure and plot correlation matrix
    fig, ax = plt.subplots(figsize=(10, 10))
    cax = ax.imshow(df.corr(), interpolation='nearest', vmin=-1, vmax=1)
    fig.colorbar(cax)

    # Set axis labels
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    if show_labels:
        ax.set_xticklabels(labels)
        ax.set_yticklabels(labels, rotation=90)
    else:
        ax.set_xticklabels([])
        ax.set_yticklabels([])

    # Turn off the grid
    ax.grid(False)
    
    if title:
        fig.suptitle(title, fontsize=20)
    
    if save:
        figure_path = os.path.join(path_to_plots,figure_name)
        _, file_extension = os.path.splitext(figure_name)
        save_kwargs = {}
        if file_extension.lower() in ['.jpg', '.jpeg', '.png']:
            save_kwargs['pil_kwargs'] = {'quality': 50}
        plt.savefig(figure_path, **save_kwargs)
    if show:
        plt.show()
    return

def plot_corr_matrix_side_by_side(dist_1: tfp.distributions.Distribution,
                                  dist_2: tfp.distributions.Distribution,
                                  path_to_plots,
                                  figure_name="corre_matrix_plot_side_by_side.pdf",
                                  max_points=10_000,
                                  title = None,
                                  show_labels=True,
                                  show=False,
                                  save=False) -> None:
    """
    Plots two correlation matrices side by side with a single colorbar in the middle.

    Parameters:
    - dist_1, dist_2: Distributions from which samples are drawn to compute the correlation matrices.
    - path_to_plots: Directory path to save the figure.
    - figure_name: Name of the file to save the figure.
    - max_points: Maximum number of points to sample from each distribution.
    - show_labels: Whether to show labels on the axes.
    - show: Whether to display the plot.
    - save: Whether to save the plot to a file.
    """
    try:
        tf.random.set_seed(0)
        np.random.seed(0)
        samp_1 = dist_1.sample(max_points).numpy()
    except:
        samp_1 = dist_1
    try:
        samp_2 = dist_2.sample(max_points).numpy()
    except:
        samp_2 = dist_2
    labels = [r"$\mathbf{x}_{%d}$" % i for i in range(1, samp_1.shape[1] + 1)]

    # Create DataFrames
    df1 = pd.DataFrame(samp_1, columns=labels)
    df2 = pd.DataFrame(samp_2, columns=labels)

    # Create figure and axes for subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10), gridspec_kw={'width_ratios': [1, 1], 'wspace': 0.2})

    # Plot correlation matrices
    cax1 = ax1.imshow(df1.corr(), interpolation='nearest', vmin=-1, vmax=1)
    cax2 = ax2.imshow(df2.corr(), interpolation='nearest', vmin=-1, vmax=1)

    # Set axis labels
    for ax in (ax1, ax2):
        ax.set_xticks(range(len(labels)))
        ax.set_yticks(range(len(labels)))
        ax.xaxis.set_ticks_position('bottom')
        ax.yaxis.set_ticks_position('left')
        if show_labels:
            ax.set_xticklabels(labels)
            ax.set_yticklabels(labels, rotation=90)
        else:
            ax.set_xticklabels([])
            ax.set_yticklabels([])

    # Disable grid
    ax1.grid(False)
    ax2.grid(False)

    # Add one colorbar in the middle
    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])  # Adjust these values to move and resize the colorbar
    fig.colorbar(cax1, cax=cbar_ax)
    
    if title:
        fig.suptitle(title, fontsize=20)

    # Saving the figure
    if save:
        figure_path = os.path.join(path_to_plots, figure_name)
        _, file_extension = os.path.splitext(figure_name)
        save_kwargs = {}
        if file_extension.lower() in ['.jpg', '.jpeg', '.png']:
            save_kwargs['pil_kwargs'] = {'quality': 50}
        plt.savefig(figure_path, **save_kwargs)

    # Show plot if requested
    if show:
        plt.show()

    plt.close()
    return
