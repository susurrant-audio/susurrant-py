from parse_dbscan import parse_dbscan, dbscan_clusters
import glob


def test_parse():
    dbscans = glob.glob("../vocab/train/*.dbscan")
    assert len(dbscans) > 0
    for db_dir in dbscans:
        parsed = parse_dbscan(db_dir)
        exemplars = []
        assert len(parsed['clusters']) > 0
        for cluster in parsed['clusters']:
            exemplars.append(cluster['exemplar'])
            ids = cluster['ids']
            assert cluster['count'] == len(ids)
        dim = len(exemplars[0])
        assert dim > 10
        assert all(len(x) == dim for x in exemplars)


def test_make_cluster_array():
    dbscans = glob.glob("../vocab/train/*.dbscan")
    assert len(dbscans) > 0
    for db_dir in dbscans:
        parsed = parse_dbscan(db_dir)
        clusters = dbscan_clusters(parsed)
        print clusters.shape
        assert clusters.shape[0] > 50
        assert clusters.shape[1] > 10
