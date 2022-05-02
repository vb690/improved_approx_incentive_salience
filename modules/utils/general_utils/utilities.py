import os
from copy import deepcopy

import pickle

import numpy as np
from scipy import stats

from sklearn.preprocessing import QuantileTransformer, KBinsDiscretizer
from skimage.exposure import equalize_hist

from tensorflow.keras.models import load_model
import tensorflow as tf


def make_list_flat(list_to_flatten):
    """Short summary.

    Args:
        list_to_flatten (type): Description of parameter `list_to_flatten`.

    Returns:
        type: Description of returned object.

    """
    flat_list = []
    flat_list.extend([list_to_flatten]) if (
        type(list_to_flatten) is not list
    ) else [flat_list.extend(make_list_flat(e)) for e in list_to_flatten]
    return flat_list


def generate_dir(path):
    """Short summary.

    Args:
        path (type): Description of parameter `path`.

    Returns:
        type: Description of returned object.

    """
    if not os.path.exists(path):
        os.makedirs(path)
    return None


def count_files_dir(path):
    """Short summary.

    Args:
        path (type): Description of parameter `path`.

    Returns:
        type: Description of returned object.

    """
    path, dirs, files = next(os.walk(path))
    files_count = len(files)
    return files_count


def top_k_variance(df, columns, k=50, no_variance_filter=True):
    """Short summary.

    Args:
        df (type): Description of parameter `df`.
        columns (type): Description of parameter `columns`.
        k (type): Description of parameter `k`.
        no_variance_filter (type): Description of parameter `no_variance_filter`.

    Returns:
        type: Description of returned object.

    """
    X = df[columns].values
    columns = np.array(columns)
    var = np.var(X, axis=0)
    if no_variance_filter:
        mask = var != 0
        columns = columns[mask]
    sorted_var = np.sort(var)[::-1]
    threshold = sorted_var[k - 1]

    mask = var <= threshold
    columns_to_drop = columns[mask]
    # df = df.drop(columns_to_drop, axis=1)
    return df, columns_to_drop


def group_wise_binning(array, n_bins, grouper=None, method=None, **kwargs):
    """Short summary.

    Args:
        array (type): Description of parameter `array`.
        n_bins (type): Description of parameter `n_bins`.
        grouper (type): Description of parameter `grouper`.
        method (type): Description of parameter `method`.
        **kwargs (type): Description of parameter `**kwargs`.

    Returns:
        type: Description of returned object.

    """

    def binning(array):
        """Short summary.

        Args:
            array (type): Description of parameter `array`.

        Returns:
            type: Description of returned object.

        """
        ranked = stats.rankdata(array)
        data_percentile = ranked / len(array) * 100
        binned = np.digitize(
            data_percentile, [i for i in range(1, n_bins + 1)], right=True
        )
        return binned

    array = deepcopy(array)
    grouper = deepcopy(grouper)
    listed_input = False

    if isinstance(array, list):
        lengths = [len(a.flatten()) for a in array]
        array = np.hstack([a.flatten() for a in array])
        grouper = np.hstack([g.flatten() for g in grouper])
        listed_input = True

    array = array.flatten()
    if grouper is None:
        grouper = np.zeros(shape=(len(array)))
    else:
        grouper = np.array(grouper)
    for unique_group in np.unique(grouper):

        indices = np.argwhere(grouper == unique_group).flatten()
        if method == "eq_hist":
            array[indices] = equalize_hist(array[indices], nbins=n_bins)
        elif method == "quant_uni":
            array[indices] = (
                QuantileTransformer(**kwargs)
                .fit_transform(array[indices].reshape(-1, 1))
                .flatten()
            )
        elif method == "quant_norm":
            array[indices] = (
                QuantileTransformer(output_distribution="normal", **kwargs)
                .fit_transform(array[indices].reshape(-1, 1))
                .flatten()
            )
        elif method == "discret":
            array[indices] = (
                KBinsDiscretizer(n_bins=n_bins, encode="ordinal", **kwargs)
                .fit_transform(array[indices].reshape(-1, 1))
                .flatten()
            )
        else:
            array[indices] = binning(array[indices])

    if listed_input:
        new_array = []
        current_index = 0
        for length in lengths:

            new_array.append(array[current_index : current_index + length])
            current_index += length

        return new_array

    return array


def generate_3d_pad(list_arrays, shape, pad=np.nan):
    """Short summary.

    Args:
        list_arrays (type): Description of parameter `list_arrays`.
        shape (type): Description of parameter `shape`.
        pad (type): Description of parameter `pad`.

    Returns:
        type: Description of returned object.

    """
    padded_array = np.empty(shape=shape)
    padded_array[:] = pad
    index = 0
    for array in list_arrays:

        size = array.shape[0]
        length = array.shape[1]
        padded_array[index : size + index, 0:length, :] = array
        index += size

    return padded_array


def generate_exp_decay_weights(length, gamma=0.1):
    """Short summary.

    Args:
        length (type): Description of parameter `length`.
        gamma (type): Description of parameter `gamma`.

    Returns:
        type: Description of returned object.

    """
    weights = []
    for t in range(length):

        weights.append(gamma**t)

    weights = np.array(weights)
    return weights


def save_arrays(arrays, dir_name):
    """Short summary.

    Args:
        arrays (type): Description of parameter `arrays`.
        dir_name (type): Description of parameter `dir_name`.

    Returns:
        type: Description of returned object.

    """
    save_dir = "data\\{}".format(dir_name)
    generate_dir(save_dir)
    for name, array in arrays.items():

        path = "{}\\{}.npy".format(save_dir, name)
        np.save(array, path, allow_pickle=True)


def load_arrays(arrays, dir_name):
    """Short summary.

    Args:
        arrays (type): Description of parameter `arrays`.
        dir_name (type): Description of parameter `dir_name`.

    Returns:
        type: Description of returned object.

    """
    load_dir = "data\\{}".format(dir_name)
    loaded_arrays = {}
    for array in arrays:

        path = "{}\\{}.npy".format(load_dir, array)
        loaded_array = np.load(path, allow_pickle=True)
        loaded_arrays[array] = loaded_array

    return loaded_arrays


def save_objects(objects, dir_name):
    """Short summary.

    Args:
        objects (type): Description of parameter `objects`.
        dir_name (type): Description of parameter `dir_name`.

    Returns:
        type: Description of returned object.

    """
    save_dir = "results\\{}".format(dir_name)
    generate_dir(save_dir)
    for name, obj in objects.items():

        path = "{}\\{}.pkl".format(save_dir, name)

        with open(path, "wb") as out:
            pickle.dump(obj, out, pickle.HIGHEST_PROTOCOL)


def load_objects(objects, dir_name):
    """Short summary.

    Args:
        objects (type): Description of parameter `objects`.
        dir_name (type): Description of parameter `dir_name`.

    Returns:
        type: Description of returned object.

    """
    load_dir = "results\\{}".format(dir_name)
    loaded_objects = {}
    for obj in objects:

        path = "{}\\{}.pkl".format(load_dir, obj)
        with open(path, "rb") as inp:
            loaded_obj = pickle.load(inp)
            loaded_objects[obj] = loaded_obj

    return loaded_objects


def save_full_model(model, path="results\\saved_models\\{}"):
    """Short summary.

    Args:
        model (type): Description of parameter `model`.
        path (type): Description of parameter `path`.

    Returns:
        type: Description of returned object.

    """
    name = model.get_model_tag()
    path = path.format(name)
    generate_dir(path)

    keras_model = model.get_model()
    tf.saved_model.save(keras_model, "{}\\engine\\".format(path))

    with open("{}\\scaffolding.pkl".format(path), "wb") as output:
        pickle.dump(model, output, pickle.HIGHEST_PROTOCOL)


def load_full_model(
    name,
    custom_objects=None,
    path="results\\saved_models\\{}",
    **compile_schema
):
    """Short summary.

    Args:
        name (type): Description of parameter `name`.
        custom_objects (type): Description of parameter `custom_objects`.
        path (type): Description of parameter `path`.
        **compile_schema (type): Description of parameter `**compile_schema`.

    Returns:
        type: Description of returned object.

    """
    path = path.format(name)

    keras_model = load_model(
        "{}\\engine\\".format(path),
        custom_objects=custom_objects,
        compile=False,
    )
    keras_model.compile(**compile_schema)
    with open("{}\\scaffolding.pkl".format(path), "rb") as input:
        model = pickle.load(input)

    try:
        model.set_model(keras_model)
    except Exception:
        import types

        def set_model(self, model):
            setattr(self, "_model", model)
            setattr(self, "n_parameters", model.count_params())

        model.set_model = types.MethodType(set_model, model)
        model.set_model(keras_model)

    return model
