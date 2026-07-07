import pandas as pd
import os

def test_graph_features_file_exists():
    assert os.path.exists('data/processed/graph_features_final.csv')

def test_graph_features_columns():
    gf = pd.read_csv('data/processed/graph_features_final.csv')
    expected = ['txId', 'in_degree', 'out_degree', 'total_degree', 'pagerank', 'clustering', 'k_core', 'weak_component_size']
    for col in expected:
        assert col in gf.columns, f'Missing column: {col}'

def test_graph_features_no_missing():
    gf = pd.read_csv('data/processed/graph_features_final.csv')
    assert gf.isnull().sum().sum() == 0

def test_pagerank_valid_range():
    gf = pd.read_csv('data/processed/graph_features_final.csv')
    assert gf['pagerank'].min() > 0
    assert gf['pagerank'].max() < 1
