import sys
sys.path.append('src')
from load_data import load_elliptic_data, clean_and_merge, split_labeled_unlabeled

def test_data_shapes():
    features, classes, edges = load_elliptic_data('data')
    assert features.shape[0] == 203769
    assert classes.shape[0] == 203769
    assert edges.shape[0] == 234355

def test_label_split():
    features, classes, edges = load_elliptic_data('data')
    df = clean_and_merge(features, classes)
    df_all, df_labeled = split_labeled_unlabeled(df)
    assert len(df_all) == 203769
    assert len(df_labeled) == 46564
    assert 'unknown' in df_all['label'].values
    assert set(df_labeled['label'].unique()).issubset({0, 1})
