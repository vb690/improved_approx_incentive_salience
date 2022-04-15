import os

from tensorflow.keras.callbacks import EarlyStopping

from modules.utils.data_utils.data_handlers import DataGenerator
from modules.utils.model_utils.metrics_losses import smape_k
from modules.utils.general_utils.utilities import load_full_model
from modules.utils.general_utils.utilities import save_full_model
from modules.utils.general_utils.embedding_handlers import extract_embedding


FEATURES_PATH = 'data\\train\\inputs\\continuous_features'

DS_FACTOR = 10

TARGETS = [
    'tar_delta_sessions',
    'tar_active_time',
    'tar_session_time',
    'tar_activity',
    'tar_sessions',
]

BTCH = [i for i in range(len(os.listdir(FEATURES_PATH)))]
TS_BTCH = BTCH[0::DS_FACTOR]
TR_BTCH = [btch for btch in BTCH if btch not in TS_BTCH]
VL_BTCH = TR_BTCH[0::20]
TR_BTCH = [btch for btch in TR_BTCH if btch not in VL_BTCH]

###############################################################################

encoders = {
    'RNN': {
        'out_layer': ['0_lstm_layer_features'],
        'inputs': [
            'continuous_features',
            'context'
        ]
    },

    'RNN_env_even': {
        'out_layer': [
            '0_lstm_layer_features',
            '0_lstm_layer_events',
            '0_lstm_layer_env',
            '0_lstm_layer_shared'
        ],
        'inputs': [
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
    }
}
encoder_objs = {}

###############################################################################

for model_name, model_dict in encoders.items():

    print(f'Extracting embedding for {model_name}')

    INPUTS = model_dict['inputs']
    out_layers = model_dict['out_layer']

    model = load_full_model(
        name=model_name,
        optimizer='adam',
        loss={
            'output_absence_act': smape_k,
            'output_active_act': smape_k,
            'output_sess_time_act': smape_k,
            'output_activity_act': smape_k,
            'output_sess_act': smape_k
        },
        metrics={
            'output_absence_act': smape_k,
            'output_active_act': smape_k,
            'output_sess_time_act': smape_k,
            'output_activity_act': smape_k,
            'output_sess_act': smape_k
        }
    )

    TR_GEN = DataGenerator(
        list_batches=TR_BTCH,
        inputs=INPUTS,
        targets=TARGETS,
        train=True,
        shuffle=True
    )

    VAL_GEN = DataGenerator(
        list_batches=VL_BTCH,
        inputs=INPUTS,
        targets=TARGETS,
        train=True,
        shuffle=True
    )

    ES = EarlyStopping(
        monitor='val_loss',
        min_delta=0.0001,
        patience=5,
        verbose=1,
        mode='auto',
        restore_best_weights=True
    )

    model.fit(
        x=TR_GEN,
        validation_data=VAL_GEN,
        epochs=200,
        verbose=2,
        callbacks=[ES],
        workers=8,
        max_queue_size=100
    )

    # save the trained model
    save_full_model(
        model,
        path='results\\saved_trained_models\\{}'
    )

    for out_layer in out_layers:

        encoder = model.get_encoder(
            out_layer=out_layer
        )
        encoder.save(
            f'results\\saved_encoders\\{model_name}_{out_layer}'
        )

        encoder_objs[f'{model_name}_{out_layer}'] = {
                'encoder': encoder,
                'inp_names': INPUTS
        }

embeddings = extract_embedding(
    encoder_objs=encoder_objs,
    root='data\\train\\inputs',
    batches=TS_BTCH,
)
