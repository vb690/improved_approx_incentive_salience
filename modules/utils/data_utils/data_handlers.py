import gc

import pickle

import numpy as np
import pandas as pd

from tensorflow.keras.utils import Sequence

from ..general_utils.utilities import count_files_dir, generate_dir
from ..general_utils.utilities import make_list_flat, save_objects
from ..data_utils.data_preprocessers import preprocessing_df


class DataGenerator(Sequence):
    """
    Class implementing a data generator
    """
    def __init__(self, list_batches, inputs, targets,
                 train=True, shuffle=True):
        """
        """
        self.list_batches = list_batches
        self.inputs = inputs
        self.targets = targets
        self.shuffle = shuffle
        self.root_dir = 'data\\train' if train else 'data\\test'
        self.on_epoch_end()

    def __len__(self):
        """
        Denotes the number of batches per epoch'
        """
        return int(len(self.list_batches))

    def __getitem__(self, index):
        """
        Generate one batch of data
        """
        # Pick a batch
        batch = self.list_batches[index]
        # Generate X and y
        X, y = self.__data_generation(batch)
        return X, y

    def on_epoch_end(self):
        """
        Updates indexes after each epoch
        """
        if self.shuffle is True:
            np.random.shuffle(self.list_batches)

    def __data_generation(self, batch):
        """Generates data containing batch_size samples"""
        X = []
        y = []
        for subdir in self.inputs:

            X.append(
                np.load('{}\\inputs\\{}\\{}.npy'.format(
                    self.root_dir, subdir, batch
                    )
                )
            )

        for subdir in self.targets:

            y.append(
                np.load('{}\\targets\\{}\\{}.npy'.format(
                    self.root_dir, subdir, batch
                    )
                )
            )

        return X, y


def activity_filler(row):
    """
    """
    for index, activity in enumerate(row['activity_index_type']):

        if len(activity) == 0:
            continue
        else:
            row[f'freq_{index}'] = np.int32(activity[0])
            row[f'act_{index}'] = np.int32(activity[1])

    return row


def create_features_batches(df, features_keys, train=True, id_key='user_id',
                            sorting_keys=['user_id', 'session_order'],
                            grouping_key='max_sess_cut', batch_size=256):
    """
    """
    root_dir = 'data\\train' if train else 'data\\test'
    generate_dir(f'{root_dir}\\inputs\\continuous_features')
    batch_index_filename = count_files_dir(
        f'{root_dir}\\inputs\\continuous_features'
    )
    df = df.sort_values(sorting_keys)
    # generate array of behavioural features
    for key, group in df.groupby(grouping_key):

        unique_ids = len(group[id_key].unique())

        array = np.array(group[features_keys])
        array = array.reshape((unique_ids, key, -1))
        array = array[:, :-1, :]
        num_batches = (array.shape[0] + batch_size - 1) // batch_size
        print('Dumping group {}'.format(key))
        print('With {} unique ids'.format(unique_ids))

        for batch_index in range(num_batches):

            minimum = min(array.shape[0], (batch_index + 1) * batch_size)
            batch = array[batch_index * batch_size: minimum]
            batch = np.float32(batch)
            np.save(
                '{}\\inputs\\continuous_features\\{}.npy'.format(
                    root_dir,
                    batch_index_filename
                ),
                arr=batch
            )
            batch_index_filename += 1
            gc.collect()
    return None


def create_embedding_batches(df, embeddings_keys, train=True, id_key='user_id',
                             sorting_keys=['user_id', 'session_order'],
                             grouping_key='max_sess_cut', batch_size=256):
    """
    """
    root_dir = 'data\\train' if train else 'data\\test'
    df = df.sort_values(sorting_keys)

    for key, group in df.groupby(grouping_key):

        unique_ids = len(group[id_key].unique())
        arrays = {}

        for embedding in embeddings_keys:

            generate_dir(f'{root_dir}\\inputs\\{embedding}')
            batch_index_filename = count_files_dir(
                f'{root_dir}\\inputs\\{embedding}'
            )
            array = np.array(group[embedding])
            array = array.reshape((unique_ids, key))
            array = array[:, :-1]
            num_batches = (array.shape[0] + batch_size - 1) // batch_size
            arrays[embedding] = array
            gc.collect()

        print('Dumping group {}'.format(key))
        print('With {} unique ids'.format(unique_ids))
        for batch_index in range(num_batches):

            for embedding, array in arrays.items():

                minimum = min(array.shape[0], (batch_index + 1) * batch_size)
                batch = array[batch_index * batch_size: minimum]
                batch = np.int32(batch)
                np.save(
                    '{}\\inputs\\{}\\{}.npy'.format(
                        root_dir,
                        embedding,
                        batch_index_filename
                    ),
                    arr=batch
                )
                gc.collect()

            batch_index_filename += 1

    return None


def create_activities_batches(df, act_columns, freq_columns, train=True,
                              id_key='user_id',
                              sorting_keys=['user_id', 'session_order'],
                              grouping_key='max_sess_cut', batch_size=256):
    """
    """
    root_dir = 'data\\train' if train else 'data\\test'
    df = df.sort_values(sorting_keys)
    generate_dir(f'{root_dir}\\inputs\\act')
    generate_dir(f'{root_dir}\\inputs\\freq')
    batch_index_filename = count_files_dir(
        f'{root_dir}\\inputs\\act'
    )
    for key, group in df.groupby(grouping_key):

        unique_ids = len(group[id_key].unique())

        act_array = np.array(group[act_columns])
        act_array = act_array.reshape((unique_ids, key, len(act_columns)))
        act_array = act_array[:, :-1, :]

        freq_array = np.array(group[freq_columns])
        freq_array = freq_array.reshape(
            (unique_ids, key, len(act_columns))
        )
        freq_array = freq_array / freq_array.sum(axis=2).reshape(
            (unique_ids, key, 1)
        )
        freq_array = np.nan_to_num(freq_array, 0)
        freq_array = freq_array[:, :-1, :]

        gc.collect()

        num_batches = (act_array.shape[0] + batch_size - 1) // batch_size

        print('Dumping group {}'.format(key))
        print('With {} unique ids'.format(unique_ids))
        for batch_index in range(num_batches):

            minimum = min(act_array.shape[0], (batch_index + 1) * batch_size)

            act_batch = act_array[batch_index * batch_size: minimum]
            act_batch = np.int32(act_batch)
            np.save(
                '{}\\inputs\\act\\{}.npy'.format(
                    root_dir,
                    batch_index_filename
                ),
                arr=act_batch,
            )

            freq_batch = freq_array[batch_index * batch_size: minimum]
            freq_batch = np.float32(freq_batch)
            np.save(
                '{}\\inputs\\freq\\{}.npy'.format(
                    root_dir,
                    batch_index_filename
                ),
                arr=freq_batch,
            )

            gc.collect()
            batch_index_filename += 1

    return None


def create_targets_batches(df, targets_keys, train=True, id_key='user_id',
                           sorting_keys=['user_id', 'session_order'],
                           grouping_key='max_sess_cut', batch_size=256):
    """
    """
    root_dir = 'data\\train' if train else 'data\\test'
    df = df.sort_values(sorting_keys)

    for key, group in df.groupby(grouping_key):

        unique_ids = len(group[id_key].unique())
        arrays = {}

        for target in targets_keys:

            generate_dir(f'{root_dir}\\targets\\{target}')
            batch_index_filename = count_files_dir(
                f'{root_dir}\\targets\\{target}'
            )
            array = np.array(group[target])
            array = array.reshape((unique_ids, key, 1))
            array = array[:, :-1, :]
            if target == 'user_id':
                array = array[:, 0, :]
            num_batches = (array.shape[0] + batch_size - 1) // batch_size
            arrays[target] = array
            gc.collect()

        print('Dumping group {}'.format(key))
        print('With {} unique ids'.format(unique_ids))
        for batch_index in range(num_batches):

            for target, array in arrays.items():

                minimum = min(array.shape[0], (batch_index + 1) * batch_size)
                batch = array[batch_index * batch_size: minimum]
                if target == 'user_id':
                    batch = batch.astype(str)
                else:
                    batch = np.float32(batch)
                np.save(
                    '{}\\targets\\{}\\{}.npy'.format(
                        root_dir,
                        target,
                        batch_index_filename
                    ),
                    arr=batch
                )
                gc.collect()

            batch_index_filename += 1

    return None


def csv_handling(games_list, features_keys, embeddings_keys, activity_key,
                 scaler, id_key='user_id', grouping_key='max_sess_cut',
                 sorting_keys=['user_id', 'session_order'],
                 train_size=0.8, n_slices=5):
    """
    """
    global_df = []
    df_tr = []
    df_ts = []
    scalers = {}
    for game in games_list:

        print('Handling game {}'.format(game))
        df = pd.read_csv(
            'data\\csv\\cleaned\\{}.csv'.format(game),
            converters={'activity_index_type': eval}
        )
        df['activity_index_type'] = df['activity_index_type'].apply(
            lambda x: [[] if len(i) == 0 else [
                np.int32(i[0]), i[1]] for i in x
            ]
        )
        df = df.sort_values(sorting_keys)

        global_df.append(df)

    global_df = pd.concat(global_df, ignore_index=True)
    max_activities = global_df[activity_key].apply(len).max()

    global_df = global_df.sort_values(sorting_keys)
    df_tr, df_ts, fit_scaler = preprocessing_df(
        df=global_df,
        features_keys=features_keys,
        scaler=scaler(),
        train_size=train_size
    )
    df_tr = df_tr.sort_values(sorting_keys)
    df_ts = df_ts.sort_values(sorting_keys)
    scalers['global'] = fit_scaler

    save_objects(
        objects=scalers,
        dir_name='saved_objects\\scalers'
    )

    mappers = {}
    for embedding in embeddings_keys:

        unique_values = df_tr[embedding].unique()
        # zero needs to be sved for the missing values
        mapper = {value: code for code, value in enumerate(unique_values, 1)}
        df_tr[embedding] = df_tr[embedding].map(mapper)
        df_ts[embedding] = df_ts[embedding].map(mapper)
        df_ts = df_ts.fillna(0)
        mappers[embedding] = mapper

    unique_values = df_tr[activity_key].to_list()
    unique_values = set(make_list_flat(unique_values))
    unique_values = [
        value for value in unique_values if type(value) is str
    ]
    mapper = {value: code for code, value in enumerate(unique_values, 1)}
    mappers[activity_key] = mapper

    save_objects(
        objects=mappers,
        dir_name='saved_objects\\mappers'
    )

    for full_df, type_df in zip([df_tr, df_ts], ['train', 'test']):

        unique_ids = full_df[id_key].unique()

        for slice_ind, slice_ids in enumerate(
                np.array_split(unique_ids, n_slices)):

            print(f'Dumping {type_df} Slice {slice_ind}')

            slice_df = full_df[full_df[id_key].isin(slice_ids)]
            slice_df.to_csv(
                f'data\\{type_df}\\df_{type_df}_{slice_ind}.csv',
                index=False
            )

    return max_activities


def array_dumping(df, activity_key, targets_keys, features_keys,
                  embeddings_keys, max_activities, train,
                  id_key='user_id', grouping_key='max_sess_cut',
                  batch_size=256, sorting_keys=['user_id', 'session_order']):
    """
    """
    with open(
        'results\\saved_objects\\mappers\\activity_index_type.pkl',
        'rb'
    ) as pickle_file:
        activity_mapper = pickle.load(pickle_file)

    df[activity_key] = df[activity_key].apply(
        lambda x: [[] if len(i) == 0 else [
            np.int32(i[0]), np.int32(activity_mapper[i[1]])] for i in x
        ]
    )

    freq_columns = [f'freq_{column}' for column in range(max_activities)]
    act_columns = [f'act_{column}' for column in range(max_activities)]

    for column in range(max_activities):

        df[f'freq_{column}'] = np.int32(0)
        df[f'act_{column}'] = np.int32(0)

    df = df.apply(lambda x: activity_filler(x), axis=1)

    create_activities_batches(
        df=df,
        act_columns=act_columns,
        freq_columns=freq_columns,
        train=train,
        id_key=id_key,
        sorting_keys=sorting_keys,
        grouping_key=grouping_key,
        batch_size=batch_size
    )
    create_targets_batches(
        df=df,
        targets_keys=targets_keys,
        train=train,
        id_key=id_key,
        sorting_keys=sorting_keys,
        grouping_key=grouping_key,
        batch_size=batch_size
    )
    create_features_batches(
        df=df,
        features_keys=features_keys,
        train=train,
        id_key=id_key,
        sorting_keys=sorting_keys,
        grouping_key=grouping_key,
        batch_size=batch_size
    )
    create_embedding_batches(
        df=df,
        embeddings_keys=embeddings_keys,
        train=train,
        id_key=id_key,
        sorting_keys=sorting_keys,
        grouping_key=grouping_key,
        batch_size=batch_size
    )
    return None
