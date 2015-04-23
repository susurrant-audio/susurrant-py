import numpy as np
from index_clusters import create_index_tree, get_tree_items


def test_create_index_tree():
    items = np.random.rand(100, 10)
    tree = create_index_tree(items)
    items_ = get_tree_items(tree)
    assert np.allclose(items, items_)
