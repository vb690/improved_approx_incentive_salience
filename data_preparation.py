import numpy as np

import pandas as pd

from sklearn.preprocessing import MinMaxScaler as mms

from modules.utils.data_utils.data_handlers import csv_handling, array_dumping

features = [
    'delta_sessions',
    'active_time',
    'session_time',
    'activity'
]
targets = [
    'user_id',
    'tar_delta_sessions',
    'tar_active_time',
    'tar_session_time',
    'tar_activity',
    'tar_sessions'
]
embeddings = [
    'context',
    'country',
    'hour',
    'month',
    'day_m',
    'day_w'
]
games = [
    'madness',
    'lisbf',
    'lis',
    'hmg',
    'jc3',
    'jc4'
]

###############################################################################

for game in games:

    print(f'Preprocessing {game}')

    df = pd.read_csv(
        f'data\\csv\\{game}.csv',
        converters={'activity_index_type': eval}
    )
    df = df.sort_values(['user_id', 'session_order'])

    df['user_id'] = df['user_id'] + df['context']
    df = df.drop_duplicates(subset=['user_id', 'session_order'])

    df['activity_index_type'] = df['activity_index_type'].apply(
        lambda x: [i.split('-') for i in x]
    )
    df['activity_index_type'] = df['activity_index_type'].apply(
        lambda x: [[] if i[1] == 'null' else i for i in x]
    )

    df['activity'] = df['activity_index_type'].apply(
        lambda x: np.sum([0 if len(i) == 0 else int(i[0]) for i in x])
    )

    df = df.rename(columns={'session_played_time': 'active_time'})
    df['delta_sessions'] = df['delta_sessions'] // 60

    df['t_stamp'] = pd.to_datetime(df['t_stamp'])

###############################################################################

    df['hour'] = df['t_stamp'].dt.hour
    df['month'] = df['t_stamp'].dt.month
    df['day_m'] = df['t_stamp'].dt.day
    df['day_w'] = df['t_stamp'].dt.dayofweek

###############################################################################

    if df.isnull().values.any():
        print(df[df.isna().any(axis=1)])
        print('')
        print('')
        print(len(df[df.isna().any(axis=1)]['user_id'].unique()))
    df = df.fillna(0)
    # # OUTLIERS REMOVAL
    # df, outliers_report = outliers_removal(
    #     df=df,
    #     contamination=0.025,
    #     n_estimators=200,
    #     max_samples=5000,
    #     features=[
    #         'delta_sessions',
    #         'active_time',
    #         'session_time',
    #         'activity'
    #     ],
    #     n_jobs=-1
    # )

###############################################################################

    # ACTIVE TIME RAW
    null_filler = df[df['active_time'] > 0]['active_time'].mean()
    df['active_time'] = df['active_time'].apply(
        lambda x: x if x > 0 else null_filler
    )

###############################################################################

    # SESSION TIME
    null_filler = df[df['session_time'] > 0]['session_time'].mean()
    df['session_time'] = df['session_time'].apply(
        lambda x: x if x > 0 else null_filler
    )
    df['session_time'] = np.where(
        df['session_time'] - df['active_time'] < 0,
        df['active_time'],
        df['session_time']
    )
    # create target
    df['tar_session_time'] = df.groupby('user_id')['session_time'].shift(-1)

###############################################################################

    # ABSENCE
    null_filler = df[df['delta_sessions'] > 0]['delta_sessions'].mean()
    df['delta_sessions'] = df['delta_sessions'].apply(
        lambda x: x if x > 0 else null_filler
    )
    # create target
    df['tar_delta_sessions'] = df.groupby(
        'user_id')['delta_sessions'].shift(-1)

###############################################################################

    # ACTIVE TIME PERCENTAGE
    df['active_time'] = df['active_time'] / df['session_time'] * 100
    df['active_time'] = round(df['active_time'], 2)
    # create target
    df['tar_active_time'] = df.groupby('user_id')['active_time'].shift(-1)


###############################################################################

    # ACTIVITY
    null_filler = int(df[df['activity'] > 0]['activity'].mean())
    df['activity'] = df['activity'].apply(
        lambda x: x if x >= 0 else null_filler
    )
    df['activity'] = df['activity'] / df['session_time']
    # create target
    df['tar_activity'] = df.groupby('user_id')['activity'].shift(-1)

###############################################################################

    # SESSION
    # create target
    df['tar_sessions'] = df['maximum_sessions'] - df['session_order']

    df['max_sess_cut'] = df.groupby('user_id')['session_order'].transform(
        np.max
    )

###############################################################################

    df = df.fillna(0)
    df = df[
        [
            'user_id',
            'context',
            'session_order',

            'country',
            'hour',
            'month',
            'day_m',
            'day_w',

            'delta_sessions',
            'active_time',
            'session_time',
            'activity',

            'tar_delta_sessions',
            'tar_active_time',
            'tar_session_time',
            'tar_activity',
            'tar_sessions',

            'activity_index_type',

            'max_sess_cut'
        ]
    ]
    df = df.sort_values(['user_id', 'session_order'])
    df = df[df['max_sess_cut'] > 1]
    df.to_csv(f'data\\csv\\cleaned\\{game}.csv', index=False)

###############################################################################

n_slices = 8

# # start the data extraction
max_activities = csv_handling(
    games_list=games,
    activity_key='activity_index_type',
    embeddings_keys=embeddings,
    features_keys=features,
    scaler=mms,
    grouping_key='max_sess_cut',
    sorting_keys=['user_id', 'session_order'],
    train_size=0.90,
    n_slices=n_slices
)

print(f'Max Activities {max_activities}')

for type_df in ['train', 'test']:

    for slice in range(n_slices):

        df = pd.read_csv(
            f'data\\{type_df}\\df_{type_df}_{slice}.csv',
            converters={'activity_index_type': eval}
        )

        print(f'Dumping arrays from slice {slice}')

        array_dumping(
            df=df,
            activity_key='activity_index_type',
            targets_keys=targets,
            features_keys=features,
            embeddings_keys=embeddings,
            max_activities=9,
            train=type_df == 'train',
            id_key='user_id',
            grouping_key='max_sess_cut',
            batch_size=256,
            sorting_keys=['user_id', 'session_order']
        )
