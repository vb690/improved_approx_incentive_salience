import os

import pickle

from tqdm import tqdm

import numpy as np

from tensorflow.keras.optimizers import Adam

from modules.utils.model_utils.metrics_losses import smape_k, smape_np
from modules.utils.general_utils.utilities import load_full_model
from modules.utils.general_utils.utilities import generate_exp_decay_weights

###############################################################################

INPUTS_PATH = "data\\train\\inputs\\{}"
TARGETS_PATH = "data\\train\\targets\\{}"

DS_FACTOR = 10

BTCH = [
    i for i in range(len(os.listdir(INPUTS_PATH.format("continuous_features"))))
]
BTCH = BTCH[0::DS_FACTOR]

SNAPSHOTS = 4

USER_PATH = TARGETS_PATH.format("user_id\\{}.npy")

TARGETS = [
    "tar_delta_sessions",
    "tar_active_time",
    "tar_session_time",
    "tar_activity",
    "tar_sessions",
]
INPUTS_CONT = ["delta_sessions", "active_time", "session_time", "activity"]
INPUTS_EMB = ["country", "hour", "day_w", "month"]
MODELS_NAME = ["RNN", "RNN_env_even"]

##############################################################################

MODELS = {}
for model in MODELS_NAME:

    MODELS[model] = load_full_model(
        name=model,
        optimizer=Adam(),
        custom_objects={"smape_k": smape_k},
        loss={
            "output_absence_act": smape_k,
            "output_active_act": smape_k,
            "output_sess_time_act": smape_k,
            "output_activity_act": smape_k,
            "output_sess_act": smape_k,
        },
        metrics={
            "output_absence_act": smape_k,
            "output_active_act": smape_k,
            "output_sess_time_act": smape_k,
            "output_activity_act": smape_k,
            "output_sess_act": smape_k,
        },
        path="results\\saved_trained_models\\{}",
    )

with open("results\\saved_objects\\scalers\\global.pkl", "rb") as pickle_file:
    SCALER = pickle.load(pickle_file)

###############################################################################

DATA_CONTAINER = {}

# inputs
inputs_temporal = {
    input_metric: {snapshot: [] for snapshot in range(SNAPSHOTS)}
    for input_metric in INPUTS_CONT
}
emb_temporal = {
    input_emb: {snapshot: [] for snapshot in range(SNAPSHOTS)}
    for input_emb in INPUTS_EMB
}
act_temporal = {snapshot: [] for snapshot in range(SNAPSHOTS)}
freq_temporal = {snapshot: [] for snapshot in range(SNAPSHOTS)}

# discounted targets
predictions_temporal = {
    model_name: {
        target_name: {snapshot: [] for snapshot in range(SNAPSHOTS)}
        for target_name in TARGETS
    }
    for model_name in MODELS_NAME
}
ground_truths_temporal = {
    target_name: {snapshot: [] for snapshot in range(SNAPSHOTS)}
    for target_name in TARGETS
}

errors_temporal = {
    model_name: {
        target_name: {snapshot: [] for snapshot in range(SNAPSHOTS)}
        for target_name in TARGETS
    }
    for model_name in MODELS_NAME
}

users_temporal = {snapshot: [] for snapshot in range(SNAPSHOTS)}
cont_temporal = {snapshot: [] for snapshot in range(SNAPSHOTS)}

###############################################################################

for btch in tqdm(BTCH):

    user_id_array = np.load(f"data\\train\\targets\\user_id\\{btch}.npy")
    cont_feat_array = np.load(
        f"data\\train\\inputs\\continuous_features\\{btch}.npy"
    )
    act_array = np.load(f"data\\train\\inputs\\act\\{btch}.npy")
    freq_array = np.load(f"data\\train\\inputs\\freq\\{btch}.npy")
    cont_array = np.load(f"data\\train\\inputs\\context\\{btch}.npy")

    max_snapshots = min(SNAPSHOTS, cont_feat_array.shape[1])

    # load users and game acts
    for snapshot in range(max_snapshots):

        users_temporal[snapshot].append(user_id_array[:, 0])
        cont_temporal[snapshot].append(cont_array[:, 0])
        act_temporal[snapshot].append(act_array[:, : snapshot + 1, :])
        freq_temporal[snapshot].append(freq_array[:, : snapshot + 1, :])

    # load embeddings inputs
    for emb_name in INPUTS_EMB:

        emb_array = np.load(f"{INPUTS_PATH.format(emb_name)}\\{btch}.npy")

        for snapshot in range(max_snapshots):

            emb_temporal[emb_name][snapshot].append(
                emb_array[:, : snapshot + 1]
            )

    # load targets
    for target_index, target_name in enumerate(TARGETS):

        ground_truth_array = np.load(
            f"{TARGETS_PATH.format(target_name)}\\{btch}.npy"
        )

        # input metrics
        if target_name != "tar_sessions":
            input_name = INPUTS_CONT[target_index]
            cont_feat_shape = cont_feat_array.shape
            cont_feat_array = cont_feat_array.reshape((-1, cont_feat_shape[2]))
            cont_feat_array = SCALER.inverse_transform(cont_feat_array)
            cont_feat_array = cont_feat_array.reshape(cont_feat_shape)

        for snapshot in range(max_snapshots):

            if target_name != "tar_sessions":
                inputs_temporal[input_name][snapshot].append(
                    cont_feat_array[:, : snapshot + 1, target_index]
                )

            weights = generate_exp_decay_weights(
                ground_truth_array[:, snapshot:, :].shape[1],
            )

            # ground truths discounted sum
            discounted_sum_ground_truth = (
                ground_truth_array[:, snapshot:, :] * weights[:, np.newaxis]
            )
            discounted_sum_ground_truth = discounted_sum_ground_truth.sum(
                axis=1
            )
            ground_truths_temporal[target_name][snapshot].append(
                discounted_sum_ground_truth
            )

    for model_name, model in MODELS.items():

        input_features = []

        if model_name == "RNN":
            INPUTS = ["continuous_features", "context"]
        elif model_name == "RNN_env_even":
            INPUTS = [
                "continuous_features",
                "context",
                "country",
                "hour",
                "month",
                "day_w",
                "day_m",
                "act",
                "freq",
            ]

        for inp in INPUTS:

            array = np.load(f"{INPUTS_PATH.format(inp)}\\{btch}.npy")
            input_features.append(array)

        list_pred = model._model.predict(
            input_features, batch_size=array.shape[0]
        )

        for target_index, target_name in enumerate(TARGETS):

            prediction_array = list_pred[target_index]
            ground_truth_array = np.load(
                f"{TARGETS_PATH.format(target_name)}\\{btch}.npy"
            )
            error_array = smape_np(ground_truth_array, prediction_array, axis=2)

            for snapshot in range(max_snapshots):

                weights = generate_exp_decay_weights(
                    prediction_array[:, snapshot:, :].shape[1],
                )

                # predictions discounted sum
                discounted_sum_predictions = (
                    prediction_array[:, snapshot:, :] * weights[:, np.newaxis]
                )
                discounted_sum_predictions = discounted_sum_predictions.sum(
                    axis=1
                )
                predictions_temporal[model_name][target_name][snapshot].append(
                    discounted_sum_predictions
                )

                # errors
                errors_temporal[model_name][target_name][snapshot].append(
                    error_array[:, snapshot]
                )

###############################################################################

for snapshot in range(SNAPSHOTS):

    users_temporal[snapshot] = np.hstack(users_temporal[snapshot])
    cont_temporal[snapshot] = np.hstack(cont_temporal[snapshot])
    act_temporal[snapshot] = np.vstack(act_temporal[snapshot])
    freq_temporal[snapshot] = np.vstack(freq_temporal[snapshot])

    for emb_name in INPUTS_EMB:

        emb_temporal[emb_name][snapshot] = np.vstack(
            emb_temporal[emb_name][snapshot]
        )

    for target_index, target_name in enumerate(TARGETS):

        if target_name != "tar_sessions":
            input_name = INPUTS_CONT[target_index]
            inputs_temporal[input_name][snapshot] = np.vstack(
                inputs_temporal[input_name][snapshot]
            )

        ground_truths_temporal[target_name][snapshot] = np.vstack(
            ground_truths_temporal[target_name][snapshot]
        )

        for model_name in MODELS_NAME:

            predictions_temporal[model_name][target_name][snapshot] = np.vstack(
                predictions_temporal[model_name][target_name][snapshot]
            )

            errors_temporal[model_name][target_name][snapshot] = np.hstack(
                errors_temporal[model_name][target_name][snapshot]
            )

###############################################################################

DATA_CONTAINER["cont_feat"] = inputs_temporal
DATA_CONTAINER["context"] = cont_temporal
DATA_CONTAINER["act"] = act_temporal
DATA_CONTAINER["freq"] = freq_temporal
DATA_CONTAINER["user_id"] = users_temporal
DATA_CONTAINER["embeddings"] = emb_temporal
DATA_CONTAINER["prediction"] = predictions_temporal
DATA_CONTAINER["ground_truth"] = ground_truths_temporal
DATA_CONTAINER["error"] = errors_temporal


with open(
    "results\\saved_objects\\saved_data_containers\\rnn_vs_env_even.pkl", "wb"
) as container:
    pickle.dump(DATA_CONTAINER, container, pickle.HIGHEST_PROTOCOL)
