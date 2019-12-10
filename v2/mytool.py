import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import tfplot


def _heatmap(data, row_labels, col_labels, ax=None,
             cbar_kw={}, cbarlabel="", **kwargs):
    """
    Create a heatmap from a numpy array and two lists of labels.

    Arguments:
        data       : A 2D numpy array of shape (N,M)
        row_labels : A list or array of length N with the labels
                     for the rows
        col_labels : A list or array of length M with the labels
                     for the columns
    Optional arguments:
        ax         : A matplotlib.axes.Axes instance to which the heatmap
                     is plotted. If not provided, use current axes or
                     create a new one.
        cbar_kw    : A dictionary with arguments to
                     :meth:`matplotlib.Figure.colorbar`.
        cbarlabel  : The label for the colorbar
    All other arguments are directly passed on to the imshow call.
    """

    if not ax:
        ax = plt.gca()

    # Plot the heatmap
    im = ax.imshow(data, **kwargs)

    # Create colorbar
    cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")

    # We want to show all ticks...
    ax.set_xticks(np.arange(data.shape[1]))
    ax.set_yticks(np.arange(data.shape[0]))
    # ... and label them with the respective list entries.
    ax.set_xticklabels(col_labels)
    ax.set_yticklabels(row_labels)

    # Let the horizontal axes labeling appear on top.
    ax.tick_params(top=True, bottom=False,
                   labeltop=True, labelbottom=False)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=-30, ha="right",
             rotation_mode="anchor")

    # Turn spines off and create white grid.
    for edge, spine in ax.spines.items():
        spine.set_visible(False)

    ax.set_xticks(np.arange(data.shape[1] + 1) - .5, minor=True)
    ax.set_yticks(np.arange(data.shape[0] + 1) - .5, minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)

    return im, cbar


def _annotate_heatmap(im, data=None, valfmt="{x:.2f}",
                      textcolors=["black", "white"],
                      threshold=None, **textkw):
    """
    A function to annotate a heatmap.

    Arguments:
        im         : The AxesImage to be labeled.
    Optional arguments:
        data       : Data used to annotate. If None, the image's data is used.
        valfmt     : The format of the annotations inside the heatmap.
                     This should either use the string format method, e.g.
                     "$ {x:.2f}", or be a :class:`matplotlib.ticker.Formatter`.
        textcolors : A list or array of two color specifications. The first is
                     used for values below a threshold, the second for those
                     above.
        threshold  : Value in data units according to which the colors from
                     textcolors are applied. If None (the default) uses the
                     middle of the colormap as separation.

    Further arguments are passed on to the created text labels.
    """

    if not isinstance(data, (list, np.ndarray)):
        data = im.get_array()

    # Normalize the threshold to the images color range.
    if threshold is not None:
        threshold = im.norm(threshold)
    else:
        threshold = im.norm(data.max()) / 2.

    # Set default alignment to center, but allow it to be
    # overwritten by textkw.
    kw = dict(horizontalalignment="center",
              verticalalignment="center")
    kw.update(textkw)

    # Get the formatter in case a string is supplied
    if isinstance(valfmt, str):
        valfmt = matplotlib.ticker.StrMethodFormatter(valfmt)

    # Loop over the data and create a `Text` for each "pixel".
    # Change the text's color depending on the data.
    texts = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            kw.update(color=textcolors[im.norm(data[i, j]) > threshold])
            text = im.axes.text(j, i, valfmt(data[i, j], None), **kw)
            texts.append(text)

    return text


def confusion_matrix_heatmap(confusion_matrix, labels=None,
                             tensor_name='/confusion_matrix', normalize=False):
    cm = confusion_matrix
    if normalize:
        cm = cm.astype('float') * 10 / cm.sum(axis=1)[:, np.newaxis]
        cm = np.nan_to_num(cm, copy=True)
        cm = cm.astype('int')

    np.set_printoptions(precision=2)
    fig, ax = plt.subplots()
    im, cbar = _heatmap(cm, labels, labels, ax=ax,
                        cmap="YlGn", cbarlabel="confusion matrix")
    texts = _annotate_heatmap(im, valfmt="{x:0.0f}")
    fig.tight_layout()

    # Add image summary
    summary = tfplot.figure.to_summary(fig, tag=tensor_name)

    return summary


def _line_diagram(array, title, xlabel='step', ylabel='rate', tick_spacing=5, tensor_name='scalars/image'):
    fig, ax = plt.subplots()
    x = np.arange(1, len(array) + 1)
    plt.plot(x, array)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    fname = title + ".svg"
    plt.savefig(fname=fname, format="svg")
    summary = tfplot.figure.to_summary(fig, tag=tensor_name)
    return summary


def _scalars_plot(data, labels=None, title="Lasers", xlabel='step', ylabel='rate', tick_spacing=5,
                  tensor_name='scalars/image'):
    x = np.arange(1, len(data[0]) + 1)
    fig, ax = plt.subplots()
    if labels is None:
        labels = [str("Class " + str(x)) for x in range(len(data))]
    for i in range(len(data)):
        plt.plot(x, data[i], label=labels[i])
    ax.xaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()

    summary = tfplot.figure.to_summary(fig, tag=tensor_name)
    return summary


def _precision(TP, FP):
    return TP / (TP + FP)


def _recall(TP, FN):
    return TP / (TP + FN)


def _F1(TP, FP, FN):
    return 2 * TP / (2 * TP + FP + FN)


def _process_data(cm_array):
    precision_list = list()
    recall_list = list()
    F1_list = list()

    class_num = cm_array.shape[1]
    steps = cm_array.shape[0]
    for cm in cm_array:
        col_sum = np.sum(cm, axis=0)
        row_sum = np.sum(cm, axis=1)
        for i in range(class_num):
            TP = cm[i][i]
            FP = row_sum[i] - TP
            FN = col_sum[i] - TP
            TN = row_sum[i] - FN
            pre = _precision(TP, FP)
            precision_list.append(pre)
            recall = _recall(TP, FN)
            recall_list.append(recall)
            f1 = _F1(TP, FP, FN)
            F1_list.append(f1)
    precision_array = np.transpose(np.array(precision_list).reshape((steps, class_num)))
    recall_array = np.transpose(np.array(recall_list).reshape((steps, class_num)))
    F1_array = np.transpose(np.array(F1_list).reshape((steps, class_num)))

    return precision_array, recall_array, F1_array


def drawing(cm_array, labels):
    precision_array, recall_array, F1_array = _process_data(cm_array)
    precision = np.sum(precision_array, axis=0)
    recall = np.sum(recall_array, axis=0)
    F1 = np.sum(F1_array, axis=0)
    summary1 = _scalars_plot(precision_array, labels=labels, title="precision", tensor_name="/test/precision")
    summary2 = _scalars_plot(recall_array, labels=labels, title="recall", tensor_name="/test/recall")
    summary3 = _scalars_plot(F1_array, labels=labels, title="F1", tensor_name="/test/F1")

    summary6 = _line_diagram(precision, title='precision', xlabel='step', ylabel='rate', tick_spacing=5,
                             tensor_name='scalars/precision')
    summary4 = _line_diagram(recall, title='recall', xlabel='step', ylabel='rate', tick_spacing=5,
                             tensor_name='scalars/recall')
    summary5 = _line_diagram(F1, title='F1', xlabel='step', ylabel='rate', tick_spacing=5, tensor_name='scalars/F1')
    summary_list = [summary1, summary2, summary3, summary6, summary4, summary5]
    print(precision)
    print(recall)
    print(F1)
    return summary_list
