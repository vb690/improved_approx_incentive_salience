import numpy as np

from kneed import KneeLocator

from ..general_utils.visualizers import visualize_auto_elbow


def auto_elbow(n_clusters, inertias, save_name, verbose=0):
    """Short summary.

    Args:
        n_clusters (type): Description of parameter `n_clusters`.
        inertias (type): Description of parameter `inertias`.
        save_name (type): Description of parameter `save_name`.
        verbose (type): Description of parameter `verbose`.

    Returns:
        type: Description of returned object.

    """
    y = (inertias[0], inertias[-1])
    x = (n_clusters[0], n_clusters[-1])

    kneedle = KneeLocator(
        n_clusters, inertias, S=1.0, curve="convex", direction="decreasing"
    )

    alpha, beta = np.polyfit(x, y, 1)
    grad_line = [beta + (alpha * k) for k in n_clusters]
    optimal_k = np.argmax([grad - i for grad, i in zip(grad_line, inertias)])

    print(optimal_k)
    optimal_k = n_clusters[optimal_k]
    if verbose > 0:
        print(f"Optimal k found at {optimal_k}")
        print(f"Kneedle found at {kneedle.knee}")
        visualize_auto_elbow(
            n_clusters=n_clusters,
            inertias=inertias,
            grad_line=grad_line,
            optimal_k=optimal_k,
            save_name=save_name,
        )
    return optimal_k
