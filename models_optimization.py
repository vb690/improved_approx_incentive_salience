import os

import numpy as np

from tensorflow.keras.callbacks import EarlyStopping

from kerastuner.tuners import Hyperband

from modules.models.supervised.baselines import TimeDistributedMLP
from modules.models.supervised.recurrent_model import RecurrentModel
from modules.utils.data_utils.data_handlers import DataGenerator

from modules.utils.general_utils.utilities import save_full_model

os.environ['PATH'] += os.pathsep + 'C:\\Program Files (x86)\\Graphviz2.38\\bin'

##############################################################################

TUN_PATH = 'data\\test\\inputs\\context'

VAL_FRAC = 0.3

MAX_EPOCHS = 40
HB_ITERATIONS = 1

BTCH = [i for i in range(len(os.listdir(TUN_PATH)))]
BTCH = np.random.choice(BTCH, len(BTCH), replace=False)

VAL_CUT = int(VAL_FRAC * len(BTCH))

TU_BTCH = BTCH[:-VAL_CUT]
VAL_TU_BTCH = BTCH[-VAL_CUT:]

INPUTS = [
    'continuous_features',
    'context'
]

TARGETS = [
    'tar_delta_sessions',
    'tar_active_time',
    'tar_session_time',
    'tar_activity',
    'tar_sessions'
]

MODELS = {
    'RNN': RecurrentModel(
        n_features=4,
        max_activities=9,
        adjust_for_env=False,
        adjust_for_events=False,
        model_tag='RNN'
    ),
    'RNN_env': RecurrentModel(
        n_features=4,
        max_activities=9,
        adjust_for_env=True,
        adjust_for_events=False,
        model_tag='RNN_env'
    ),
    'RNN_even': RecurrentModel(
        n_features=4,
        max_activities=9,
        adjust_for_env=False,
        adjust_for_events=True,
        model_tag='RNN_even'
    ),
    'RNN_env_even': RecurrentModel(
        n_features=4,
        max_activities=9,
        adjust_for_env=True,
        adjust_for_events=True,
        model_tag='RNN_env_even'
    ),
    'MLP': TimeDistributedMLP(
        n_features=4,
        max_activities=9,
        model_tag='MLP'
    )
}

##############################################################################

for name in ['MLP', 'RNN', 'RNN_env', 'RNN_even', 'RNN_env_even']:

    model = MODELS[name]

    ES = EarlyStopping(
        monitor='val_loss',
        min_delta=0.0001,
        patience=5,
        verbose=1,
        mode='auto',
        restore_best_weights=True
    )

    if name == 'RNN_env':
        INPUTS = [
            'continuous_features',
            'context',
            'country',
            'hour',
            'month',
            'day_w',
            'day_m'
        ]
    elif name == 'RNN_even':
        INPUTS = [
            'continuous_features',
            'context',
            'act',
            'freq'
        ]
    elif name in ['RNN_env_even', 'MLP']:
        INPUTS = [
            'continuous_features',
            'context',
            'country',
            'hour',
            'month',
            'day_w',
            'day_m',
            'act',
            'freq'
        ]

    TU_GEN = DataGenerator(
        list_batches=TU_BTCH,
        inputs=INPUTS,
        targets=TARGETS,
        train=True,
        shuffle=True
    )
    VAL_TU_GEN = DataGenerator(
        list_batches=VAL_TU_BTCH,
        inputs=INPUTS,
        targets=TARGETS,
        train=True,
        shuffle=True
    )

    model.tune(
        tuner=Hyperband,
        generator=TU_GEN,
        verbose=2,
        validation_data=VAL_TU_GEN,
        epochs=MAX_EPOCHS,
        max_epochs=MAX_EPOCHS,
        hyperband_iterations=HB_ITERATIONS,
        objective='val_loss',
        callbacks=[ES],
        directory='o',
        project_name='{}'.format(name)
    )

    save_full_model(model=model)
