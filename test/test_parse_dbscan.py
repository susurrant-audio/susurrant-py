from parse_dbscan import parse_dbscan
import glob


def test_parse():
    for db_dir in glob.glob("../vocab/train/*.dbscan"):
        parsed = parse_dbscan(db_dir)
        exemplars = []
        assert len(parsed['clusters']) > 0
        for cluster in parsed['clusters']:
            exemplars.append(cluster['exemplar'])
            ids = cluster['ids']
            assert cluster['count'] == len(ids)
        dim = len(exemplars[0])
        assert all(len(x) == dim for x in exemplars)
