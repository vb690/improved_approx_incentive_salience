from tqdm import tqdm

import numpy as np

import pandas as pd

from sklearn.ensemble import IsolationForest as ifo


def outliers_removal(df, features, **kwargs):
    """Short summary.

    Args:
        df (type): Description of parameter `df`.
        features (type): Description of parameter `features`.
        **kwargs (type): Description of parameter `**kwargs`.

    Returns:
        type: Description of returned object.

    """
    outliers_report = pd.DataFrame(columns=["t_step", "entries", "outliers"])
    report_index = 0
    for t_step in tqdm(df["session_order"].unique()):

        temporal_slice = df[df["session_order"] == t_step]
        temporal_slice = temporal_slice[features].copy().values

        labels = ifo(**kwargs).fit_predict(temporal_slice)
        inliers = np.argwhere(labels == 1).flatten()
        outliers = np.argwhere(labels == -1).flatten()

        outliers_report.loc[report_index] = [t_step, len(labels), len(outliers)]

        for feature_ind, feature in enumerate(features):

            filler = np.random.choice(
                temporal_slice[inliers, feature_ind], size=len(outliers)
            )

            temporal_slice[outliers, feature_ind] = filler
            temporal_column = temporal_slice[:, feature_ind]
            df.loc[df["session_order"] == t_step, feature] = temporal_column

        report_index += 1

    return df, outliers_report


def __tr_ts_df_split(df, train_size, id_key="user_id"):
    """Short summary.

    Args:
        df (type): Description of parameter `df`.
        train_size (type): Description of parameter `train_size`.
        id_key (type): Description of parameter `id_key`.

    Returns:
        type: Description of returned object.

    """
    unique_ids = df[id_key].unique()
    np.random.shuffle(unique_ids)
    cutpoint = int(len(unique_ids) * train_size)

    df_tr = df.set_index(id_key).loc[unique_ids[:cutpoint]]
    df_tr[id_key] = list(df_tr.index)
    df_tr.reset_index(drop=True, inplace=True)

    df_ts = df.set_index(id_key).loc[unique_ids[cutpoint:]]
    df_ts[id_key] = list(df_ts.index)
    df_ts.reset_index(drop=True, inplace=True)
    return df_tr, df_ts


def preprocessing_df(
    df, features_keys, scaler, id_key="user_id", train_size=0.8
):
    """Short summary.

    Args:
        df (type): Description of parameter `df`.
        features_keys (type): Description of parameter `features_keys`.
        scaler (type): Description of parameter `scaler`.
        id_key (type): Description of parameter `id_key`.
        train_size (type): Description of parameter `train_size`.

    Returns:
        type: Description of returned object.

    """
    df_tr, df_ts = __tr_ts_df_split(df=df, train_size=train_size, id_key=id_key)
    df_tr[features_keys] = scaler.fit_transform(df_tr[features_keys].values)
    df_ts[features_keys] = scaler.transform(df_ts[features_keys].values)
    return df_tr, df_ts, scaler
