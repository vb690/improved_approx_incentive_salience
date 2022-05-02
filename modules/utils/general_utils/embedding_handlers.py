import pickle

import numpy as np

import pandas as pd

from sklearn.preprocessing import StandardScaler as ss

from tqdm import tqdm

from .utilities import generate_dir


def extract_embedding(encoder_objs, root, batches, max_seq=4):
    """Short summary.

    Args:
        encoder_objs (type): Description of parameter `encoder_objs`.
        root (type): Description of parameter `root`.
        batches (type): Description of parameter `batches`.
        max_seq (type): Description of parameter `max_seq`.

    Returns:
        type: Description of returned object.

    """
    generate_dir("results\\saved_emb")
    for enc_name, enc_obj in encoder_objs.items():

        print(f"Extracting embedding for {enc_name}")
        generate_dir(f"results\\saved_emb\\{enc_name}")

        collection_snapshots = {snapshot: [] for snapshot in range(max_seq)}
        final_state = []

        for batch in tqdm(batches):

            inputs = []
            for inp_name in enc_obj["inp_names"]:

                inp = np.load(f"{root}\\{inp_name}\\{batch}.npy")
                inputs.append(inp)

            predicted_embedding = enc_obj["encoder"].predict(inputs)
            final_state.append(predicted_embedding[:, -1, :])
            batch_size, time_size, space_size = predicted_embedding.shape
            for time_step in range(max_seq):

                if time_step > (time_size - 1):
                    empty_batch = np.empty((batch_size, space_size))
                    empty_batch[:] = np.nan
                    collection_snapshots[time_step].append(empty_batch)
                else:
                    collection_snapshots[time_step].append(
                        predicted_embedding[:, time_step, :]
                    )

        for snapshot, embedding in collection_snapshots.items():

            embedding = np.vstack(embedding)
            np.save(
                file=f"results\\saved_emb\\{enc_name}\\{snapshot}",
                arr=embedding,
            )

        final_state = np.vstack(final_state)
        np.save(
            file=f"results\\saved_emb\\{enc_name}\\final_state", arr=final_state
        )


def create_relationships(users, t_steps=9):
    """Short summary.

    Args:
        users (type): Description of parameter `users`.
        t_steps (type): Description of parameter `t_steps`.

    Returns:
        type: Description of returned object.

    """
    relationships = []
    for t in range(t_steps):

        pre_df = pd.DataFrame(users[t]).set_index(0)
        pre_df["code"] = [i for i in range(len(pre_df))]

        post_df = pd.DataFrame(users[t + 1]).set_index(0)
        post_df["code"] = [i for i in range(len(post_df))]

        merge = pd.merge(pre_df, post_df, left_index=True, right_index=True)
        relationships.append(dict(merge.values))

    return relationships


def reduce_dimensions(
    reducer,
    path,
    name,
    snapshots,
    context_aware=False,
    standardize=True,
    **kwargs,
):
    """Short summary.

    Args:
        reducer (type): Description of parameter `reducer`.
        path (type): Description of parameter `path`.
        name (type): Description of parameter `name`.
        snapshots (type): Description of parameter `snapshots`.
        context_aware (type): Description of parameter `context_aware`.
        standardize (type): Description of parameter `standardize`.
        **kwargs (type): Description of parameter `**kwargs`.

    Returns:
        type: Description of returned object.

    """
    componets = kwargs["n_components"]
    reducer_name = reducer["name"]
    save_path = f"results\\saved_dim_reduction\\{name}\\{componets}D\\"
    generate_dir(f"results\\saved_dim_reduction\\{name}\\")
    generate_dir(f"results\\saved_dim_reduction\\{name}\\{componets}D\\")
    for snapshot in snapshots:

        embedding = np.load(f"{path}\\{name}\\{snapshot}.npy")
        if standardize:
            embedding = ss().fit_transform(embedding)

        # finding nans and masking
        nan_mask = np.isnan(embedding).any(axis=1)
        embedding = embedding[~nan_mask]

        # re-populate an array maintaining NaN values where needed
        reduction = np.empty(shape=(len(nan_mask), componets))
        reduction[:] = np.nan

        if context_aware:
            with (
                open(f"results\\saved_data_containers\\melchior.pkl", "rb")
            ) as container:

                contexts = pickle.load(container)
                contexts = contexts["context"][snapshot]

            for unique_context in np.unique(contexts):

                print(
                    f"Transforming {name}_{snapshot} for \
                    context {unique_context} with {reducer_name}"
                )

                context_indices = np.argwhere(
                    contexts == unique_context
                ).flatten()

                reducer_algo = reducer["algo"](**kwargs)
                reduced = reducer_algo.fit_transform(embedding[context_indices])
                non_nan = reduction[~nan_mask]
                non_nan[context_indices] = reduced
                reduction[~nan_mask] = non_nan

            np.save(
                f"{save_path}{reducer_name}_{name}_{snapshot}_cont_aware.npy",
                reduction,
            )
        else:
            reducer_algo = reducer["algo"](**kwargs)

            print(
                f"Transforming {name} snapshot {snapshot} with {reducer_name}"
            )

            reduction[~nan_mask] = reducer_algo.fit_transform(embedding)
            np.save(f"{save_path}{reducer_name}{snapshot}.npy", reduction)
    return None
