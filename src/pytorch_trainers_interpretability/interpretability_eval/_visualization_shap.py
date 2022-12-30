##################################################################### 
# Visualization library get from here                               #
# https://github.com/slundberg/shap/blob/master/shap/plots/_image.py#
# It is edited to work only for rank one visualization              #
#####################################################################
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy as sp
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer
from scipy.sparse import issparse
import shap.plots.colors as colors
from matplotlib.gridspec import GridSpec



def image_plot(shap_values,
          pixel_values = None,
          labels = None,
          true_labels = None,
          width = 20,
          aspect = 0.2,
          hspace = 0.2,
          labelpad = None,
          cmap = colors.red_transparent_blue):
    """ Plots SHAP values for image inputs.
    Parameters
    ----------
    shap_values : [numpy.array]
        List of arrays of SHAP values. Each array has the shap (# samples x width x height x channels), and the
        length of the list is equal to the number of model outputs that are being explained.
    pixel_values : numpy.array
        Matrix of pixel values (# samples x width x height x channels) for each image. It should be the same
        shape as each array in the shap_values list of arrays.
    labels : list or np.ndarray
        List or np.ndarray (# samples x top_k classes) of names for each of the model outputs that are being explained.
    true_labels: list
        List of a true image labels to plot
    width : float
        The width of the produced matplotlib plot.
    labelpad : float
        How much padding to use around the model output labels.
    show : bool
        Whether matplotlib.pyplot.show() is called before returning. Setting this to False allows the plot
        to be customized further after it has been created.
    """

    # support passing an explanation object
    if str(type(shap_values)).endswith("Explanation'>"):
        shap_exp = shap_values
        # feature_names = [shap_exp.feature_names]
        # ind = 0
        if len(shap_exp.output_dims) == 1:
            shap_values = [shap_exp.values[..., i] for i in range(shap_exp.values.shape[-1])]
        elif len(shap_exp.output_dims) == 0:
            shap_values = shap_exp.values
        else:
            raise Exception("Number of outputs needs to have support added!! (probably a simple fix)")
        if pixel_values is None:
            pixel_values = shap_exp.data
        if labels is None:
            labels = shap_exp.output_names

    multi_output = True
    if not isinstance(shap_values, list):
        multi_output = False
        shap_values = [shap_values]

    if len(shap_values[0].shape) == 3:
        shap_values = [v.reshape(1, *v.shape) for v in shap_values]
        pixel_values = pixel_values.reshape(1, *pixel_values.shape)

    # labels: (rows (images) x columns (top_k classes) )
    if labels is not None:
        if isinstance(labels, list):
            labels = np.array(labels).reshape(1, -1)

    # if labels is not None:
    #     labels = np.array(labels)
    #     if labels.shape[0] != shap_values[0].shape[0] and labels.shape[0] == len(shap_values):
    #         labels = np.tile(np.array([labels]), shap_values[0].shape[0])
    #     assert labels.shape[0] == shap_values[0].shape[0], "Labels must have same row count as shap_values arrays!"
    #     if multi_output:
    #         assert labels.shape[1] == len(shap_values), "Labels must have a column for each output in shap_values!"
    #     else:
    #         assert len(labels[0].shape) == 1, "Labels must be a vector for single output shap_values."

    label_kwargs = {} if labelpad is None else {'pad': labelpad}

    # plot our explanations
    x = pixel_values
    num_cols = np.ceil(x.shape[0] / 8).astype(int)
    fig = plt.figure(figsize=(5*num_cols, 30))
    subfigs = fig.subfigures(nrows=1, ncols=num_cols)
    k = 0
    for j, sub in enumerate(subfigs):
        axes = sub.subplots(nrows=8, ncols=2)
        for row in range(8):
            if(k == x.shape[0]):
                break
            x_curr = x[k].copy()

            # make sure we have a 2D array for grayscale
            if len(x_curr.shape) == 3 and x_curr.shape[2] == 1:
                x_curr = x_curr.reshape(x_curr.shape[:2])

            # if x_curr.max() > 1:
            #     x_curr /= 255.

            # get a grayscale version of the image
            if len(x_curr.shape) == 3 and x_curr.shape[2] == 3:
                x_curr_gray = (
                        0.2989 * x_curr[:, :, 0] + 0.5870 * x_curr[:, :, 1] + 0.1140 * x_curr[:, :, 2])  # rgb to gray
                x_curr_disp = x_curr
            elif len(x_curr.shape) == 3:
                x_curr_gray = x_curr.mean(2)

                # for non-RGB multi-channel data we show an RGB image where each of the three channels is a scaled k-mean center
                flat_vals = x_curr.reshape([x_curr.shape[0] * x_curr.shape[1], x_curr.shape[2]]).T
                flat_vals = (flat_vals.T - flat_vals.mean(1)).T
                means = kmeans(flat_vals, 3, round_values=False).data.T.reshape([x_curr.shape[0], x_curr.shape[1], 3])
                x_curr_disp = (means - np.percentile(means, 0.5, (0, 1))) / (
                        np.percentile(means, 99.5, (0, 1)) - np.percentile(means, 1, (0, 1)))
                x_curr_disp[x_curr_disp > 1] = 1
                x_curr_disp[x_curr_disp < 0] = 0
            else:
                x_curr_gray = x_curr
                x_curr_disp = x_curr

            axes[row, 0].imshow(x_curr_disp, cmap=plt.get_cmap('gray'))
            if true_labels:
                axes[row, 0].set_title(true_labels[k], **label_kwargs)
            axes[row, 0].axis('off')
            if len(shap_values[0][k].shape) == 2:
                abs_vals = np.stack([np.abs(shap_values[i]) for i in range(len(shap_values))], 0).flatten()
            else:
                abs_vals = np.stack([np.abs(shap_values[i].sum(-1)) for i in range(len(shap_values))], 0).flatten()
            max_val = np.nanpercentile(abs_vals, 99.9)
            for i in range(len(shap_values)):
                if labels is not None:
                    axes[row, i + 1].set_title(labels[k, i], **label_kwargs)
                sv = shap_values[i][k] if len(shap_values[i][k].shape) == 2 else shap_values[i][k].sum(-1)
                axes[row, i + 1].imshow(x_curr_gray, cmap=plt.get_cmap('gray'), alpha=0.15,
                                        extent=(-1, sv.shape[1], sv.shape[0], -1))
                im = axes[row, i + 1].imshow(sv, cmap=cmap, vmin=-max_val, vmax=max_val)
                axes[row, i + 1].axis('off')
            k+=1
    plt.show()
    fig = plt.figure(figsize=(5, 0.2))
    ax = plt.subplot()
    cb = fig.colorbar(im, cax=ax, label="SHAP value", orientation="horizontal")
    cb.outline.set_visible(False)
    plt.show()

def kmeans(X, k, round_values=True):
    """ Summarize a dataset with k mean samples weighted by the number of data points they
    each represent.
    Parameters
    ----------
    X : numpy.array or pandas.DataFrame or any scipy.sparse matrix
        Matrix of data samples to summarize (# samples x # features)
    k : int
        Number of means to use for approximation.
    round_values : bool
        For all i, round the ith dimension of each mean sample to match the nearest value
        from X[:,i]. This ensures discrete features always get a valid value.
    Returns
    -------
    DenseData object.
    """

    group_names = [str(i) for i in range(X.shape[1])]
    if str(type(X)).endswith("'pandas.core.frame.DataFrame'>"):
        group_names = X.columns
        X = X.values

    # in case there are any missing values in data impute them
    imp = SimpleImputer(missing_values=np.nan, strategy='mean')
    X = imp.fit_transform(X)

    kmeans = KMeans(n_clusters=k, random_state=0).fit(X)

    if round_values:
        for i in range(k):
            for j in range(X.shape[1]):
                xj = X[:,j].toarray().flatten() if issparse(X) else X[:, j] # sparse support courtesy of @PrimozGodec
                ind = np.argmin(np.abs(xj - kmeans.cluster_centers_[i,j]))
                kmeans.cluster_centers_[i,j] = X[ind,j]
    return DenseData(kmeans.cluster_centers_, group_names, None, 1.0*np.bincount(kmeans.labels_))


class Instance:
    def __init__(self, x, group_display_values):
        self.x = x
        self.group_display_values = group_display_values


def convert_to_instance(val):
    if isinstance(val, Instance):
        return val
    else:
        return Instance(val, None)


class InstanceWithIndex(Instance):
    def __init__(self, x, column_name, index_value, index_name, group_display_values):
        Instance.__init__(self, x, group_display_values)
        self.index_value = index_value
        self.index_name = index_name
        self.column_name = column_name

    def convert_to_df(self):
        index = pd.DataFrame(self.index_value, columns=[self.index_name])
        data = pd.DataFrame(self.x, columns=self.column_name)
        df = pd.concat([index, data], axis=1)
        df = df.set_index(self.index_name)
        return df


def convert_to_instance_with_index(val, column_name, index_value, index_name):
    return InstanceWithIndex(val, column_name, index_value, index_name, None)


def match_instance_to_data(instance, data):
    assert isinstance(instance, Instance), "instance must be of type Instance!"

    if isinstance(data, DenseData):
        if instance.group_display_values is None:
            instance.group_display_values = [instance.x[0, group[0]] if len(group) == 1 else "" for group in data.groups]
        assert len(instance.group_display_values) == len(data.groups)
        instance.groups = data.groups


class Model:
    def __init__(self, f, out_names):
        self.f = f
        self.out_names = out_names


def convert_to_model(val):
    if isinstance(val, Model):
        return val
    else:
        return Model(val, None)


def match_model_to_data(model, data):
    assert isinstance(model, Model), "model must be of type Model!"
    
    try:
        if isinstance(data, DenseDataWithIndex):
            out_val = model.f(data.convert_to_df())
        else:
            out_val = model.f(data.data)
    except:
        print("Provided model function fails when applied to the provided data set.")
        raise

    if model.out_names is None:
        if len(out_val.shape) == 1:
            model.out_names = ["output value"]
        else:
            model.out_names = ["output value "+str(i) for i in range(out_val.shape[0])]
    
    return out_val



class Data:
    def __init__(self):
        pass


class SparseData(Data):
    def __init__(self, data, *args):
        num_samples = data.shape[0]
        self.weights = np.ones(num_samples)
        self.weights /= np.sum(self.weights)
        self.transposed = False
        self.groups = None
        self.group_names = None
        self.groups_size = data.shape[1]
        self.data = data


class DenseData(Data):
    def __init__(self, data, group_names, *args):
        self.groups = args[0] if len(args) > 0 and args[0] is not None else [np.array([i]) for i in range(len(group_names))]

        l = sum(len(g) for g in self.groups)
        num_samples = data.shape[0]
        t = False
        if l != data.shape[1]:
            t = True
            num_samples = data.shape[1]

        valid = (not t and l == data.shape[1]) or (t and l == data.shape[0])
        assert valid, "# of names must match data matrix!"

        self.weights = args[1] if len(args) > 1 else np.ones(num_samples)
        self.weights /= np.sum(self.weights)
        wl = len(self.weights)
        valid = (not t and wl == data.shape[0]) or (t and wl == data.shape[1])
        assert valid, "# weights must match data matrix!"

        self.transposed = t
        self.group_names = group_names
        self.data = data
        self.groups_size = len(self.groups)


class DenseDataWithIndex(DenseData):
    def __init__(self, data, group_names, index, index_name, *args):
        DenseData.__init__(self, data, group_names, *args)
        self.index_value = index
        self.index_name = index_name

    def convert_to_df(self):
        data = pd.DataFrame(self.data, columns=self.group_names)
        index = pd.DataFrame(self.index_value, columns=[self.index_name])
        df = pd.concat([index, data], axis=1)
        df = df.set_index(self.index_name)
        return df


def convert_to_data(val, keep_index=False):
    if isinstance(val, Data):
        return val
    elif type(val) == np.ndarray:
        return DenseData(val, [str(i) for i in range(val.shape[1])])
    elif str(type(val)).endswith("'pandas.core.series.Series'>"):
        return DenseData(val.values.reshape((1,len(val))), list(val.index))
    elif str(type(val)).endswith("'pandas.core.frame.DataFrame'>"):
        if keep_index:
            return DenseDataWithIndex(val.values, list(val.columns), val.index.values, val.index.name)
        else:
            return DenseData(val.values, list(val.columns))
    elif sp.sparse.issparse(val):
        if not sp.sparse.isspmatrix_csr(val):
            val = val.tocsr()
        return SparseData(val)
    else:
        assert False, "Unknown type passed as data object: "+str(type(val))

class Link:
    def __init__(self):
        pass


class IdentityLink(Link):
    def __str__(self):
        return "identity"

    @staticmethod
    def f(x):
        return x

    @staticmethod
    def finv(x):
        return x






class LogitLink(Link):
    def __str__(self):
        return "logit"

    @staticmethod
    def f(x):
        return np.log(x/(1-x))

    @staticmethod
    def finv(x):
        return 1/(1+np.exp(-x))


def convert_to_link(val):
    if isinstance(val, Link):
        return val
    elif val == "identity":
        return IdentityLink()
    elif val == "logit":
        return LogitLink()
    else:
        assert False, "Passed link object must be a subclass of iml.Link"