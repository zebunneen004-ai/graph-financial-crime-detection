.PHONY: all test notebooks clean

all: test notebooks

test:
	venv\Scripts\python.exe -m pytest tests\ -v

notebooks:
	venv\Scripts\python.exe -m jupyter nbconvert --execute notebooks\01_data_loading_eda.ipynb --to notebook --output-dir notebooks\executed
	venv\Scripts\python.exe -m jupyter nbconvert --execute notebooks\02_graph_construction.ipynb --to notebook --output-dir notebooks\executed
	venv\Scripts\python.exe -m jupyter nbconvert --execute notebooks\03_graph_features.ipynb --to notebook --output-dir notebooks\executed
	venv\Scripts\python.exe -m jupyter nbconvert --execute notebooks\04_statistical_tests.ipynb --to notebook --output-dir notebooks\executed
	venv\Scripts\python.exe -m jupyter nbconvert --execute notebooks\05_modeling.ipynb --to notebook --output-dir notebooks\executed
	venv\Scripts\python.exe -m jupyter nbconvert --execute notebooks\06_feature_importance.ipynb --to notebook --output-dir notebooks\executed

clean:
	-Remove-Item "results\*.csv" -ErrorAction SilentlyContinue
	-Remove-Item "figures\*.png" -ErrorAction SilentlyContinue
	-Remove-Item "notebooks\executed\*.ipynb" -ErrorAction SilentlyContinue
