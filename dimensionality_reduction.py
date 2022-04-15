from sklearn.decomposition import PCA

from umap import UMAP

from modules.utils.general_utils.embedding_handlers import reduce_dimensions

##############################################################################

SNAPSHOTS = [0, 1, 2, 3]
PATH = 'results\\saved_emb\\'
NAMES = [
    'RNN_env_even_0_lstm_layer_shared',
    'RNN_env_even_0_lstm_layer_features',
    'RNN_env_even_0_lstm_layer_events',
    'RNN_env_even_0_lstm_layer_env',
    'RNN_0_lstm_layer_features',
]

for name in NAMES:

    reduce_dimensions(
        reducer={'name': 'pca', 'algo': PCA},
        path=PATH,
        name=name,
        snapshots=SNAPSHOTS,
        n_components=2,
    )

    reduce_dimensions(
        reducer={'name': 'umap', 'algo': UMAP},
        path=PATH,
        name=name,
        snapshots=SNAPSHOTS,
        n_components=2,
        verbose=True,
        n_neighbors=100,
        n_epochs=1000,
        min_dist=0.8,
        metric='manhattan',
        low_memory=True
    )
