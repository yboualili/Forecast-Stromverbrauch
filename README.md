# bda-analytics-challenge-template

## Task

Project description can be found in docs

Master 2. Semester

### Install and generate data set

## Generate data containing only NO_PV households
```
pip install -r requirements.txt
pip install -e .

>>>

python src/preprocessing.py data/raw/ data/preprocessed/
2022-07-19 12:43:00,207 - __main__ - INFO -
 ___ ___    _
 | _ )   \ /_\
 | _ \ |) / _ \
 |___/___/_/ \_\

2022-07-19 12:43:00,207 - __main__ - INFO - starting pre-processing...
2022-07-19 12:43:00,208 - data.dataprocessor - INFO - processing load data... 💡
2022-07-19 12:43:00,208 - data.dataprocessor - INFO - loading 'data/raw/2018_data_15min.hdf5'.
2022-07-19 12:43:02,918 - data.dataprocessor - INFO - loading 'data/raw/2019_data_15min.hdf5'.
2022-07-19 12:43:07,546 - data.dataprocessor - INFO - loading 'data/raw/2020_data_15min.hdf5'.
2022-07-19 12:43:15,045 - data.dataprocessor - INFO - writing 'data/preprocessed/load.pkl'.
2022-07-19 12:43:15,468 - data.dataprocessor - INFO - processing heatpump data... 🥵
2022-07-19 12:43:15,532 - data.dataprocessor - INFO - loading 'data/raw/2018_data_15min.hdf5'.
2022-07-19 12:43:16,325 - data.dataprocessor - INFO - loading 'data/raw/2019_data_15min.hdf5'.
2022-07-19 12:43:17,019 - data.dataprocessor - INFO - loading 'data/raw/2020_data_15min.hdf5'.
2022-07-19 12:43:18,543 - data.dataprocessor - INFO - writing 'data/preprocessed/heatpump.pkl'.
2022-07-19 12:43:18,702 - __main__ - INFO - All done! ✨ 🍰 ✨

python src/additional_features.py data/preprocessed/ data/preprocessed/
2022-07-19 12:44:57,872 - __main__ - INFO -
 ___ ___    _
 | _ )   \ /_\
 | _ \ |) / _ \
 |___/___/_/ \_\

2022-07-19 12:44:57,872 - __main__ - INFO - starting adding features...
2022-07-19 12:44:57,873 - data.featureprocessor - INFO - adding features to load data... 💡
2022-07-19 12:44:57,873 - data.featureprocessor - INFO - reading 'data/preprocessed/load.pkl'.
2022-07-19 12:45:00,059 - data.featureprocessor - INFO - finished removing columns with >= 80.0 % pd.NA values.
2022-07-19 12:47:48,024 - data.featureprocessor - INFO - finished replacing pd.NA.
2022-07-19 12:47:49,475 - data.featureprocessor - INFO - finished making 'S_TOT', 'P_TOT', 'Q_TOT'.
2022-07-19 12:47:51,826 - data.featureprocessor - INFO - writing 'data/preprocessed/load.pkl'.
2022-07-19 12:47:52,074 - data.featureprocessor - INFO - adding features to heatpump data... 🥵
2022-07-19 12:47:52,074 - data.featureprocessor - INFO - reading 'data/preprocessed/heatpump.pkl'.
2022-07-19 12:47:52,645 - data.featureprocessor - INFO - finished removing columns with >= 80.0 % pd.NA values.
2022-07-19 12:48:17,163 - data.featureprocessor - INFO - finished replacing pd.NA.
2022-07-19 12:48:17,901 - data.featureprocessor - INFO - finished making 'S_TOT', 'P_TOT', 'Q_TOT'.
2022-07-19 12:48:18,830 - data.featureprocessor - INFO - writing 'data/preprocessed/heatpump.pkl'.
2022-07-19 12:48:18,949 - __main__ - INFO - All done! ✨ 🍰 ✨
```

## Generate data containing only PV households

```
pip install -r requirements.txt
pip install -e .

>>>

python src/preprocessing.py data/raw/ data/preprocessed/ --pv
2022-07-19 12:42:16,538 - __main__ - INFO -
 ___ ___    _
 | _ )   \ /_\
 | _ \ |) / _ \
 |___/___/_/ \_\

2022-07-19 12:42:16,538 - __main__ - INFO - starting pre-processing...
2022-07-19 12:42:16,538 - data.dataprocessor - INFO - processing load data... 💡
2022-07-19 12:42:16,538 - data.dataprocessor - INFO - loading 'data/raw/2018_data_15min.hdf5'.
2022-07-19 12:42:17,504 - data.dataprocessor - INFO - loading 'data/raw/2019_data_15min.hdf5'.
2022-07-19 12:42:18,182 - data.dataprocessor - INFO - loading 'data/raw/2020_data_15min.hdf5'.
2022-07-19 12:42:19,181 - data.dataprocessor - INFO - writing 'data/preprocessed/load_solar.pkl'.
2022-07-19 12:42:19,273 - data.dataprocessor - INFO - processing heatpump data... 🥵
2022-07-19 12:42:19,281 - data.dataprocessor - INFO - loading 'data/raw/2018_data_15min.hdf5'.
2022-07-19 12:42:19,470 - data.dataprocessor - INFO - loading 'data/raw/2019_data_15min.hdf5'.
2022-07-19 12:42:19,700 - data.dataprocessor - INFO - loading 'data/raw/2020_data_15min.hdf5'.
2022-07-19 12:42:20,002 - data.dataprocessor - INFO - writing 'data/preprocessed/heatpump_solar.pkl'.
2022-07-19 12:42:20,045 - __main__ - INFO - All done! ✨ 🍰 ✨

python src/additional_features.py data/preprocessed/ data/preprocessed/ --pv
2022-07-19 12:42:47,250 - __main__ - INFO -
 ___ ___    _
 | _ )   \ /_\
 | _ \ |) / _ \
 |___/___/_/ \_\

2022-07-19 12:42:47,250 - __main__ - INFO - starting adding features...
2022-07-19 12:42:47,251 - data.featureprocessor - INFO - adding features to load data... 💡
2022-07-19 12:42:47,251 - data.featureprocessor - INFO - reading 'data/preprocessed/load_solar.pkl'.
2022-07-19 12:42:47,359 - data.featureprocessor - INFO - finished removing columns with >= 80.0 % pd.NA values.
2022-07-19 12:42:48,822 - data.featureprocessor - INFO - finished replacing pd.NA.
2022-07-19 12:42:48,986 - data.featureprocessor - INFO - finished making 'S_TOT', 'P_TOT', 'Q_TOT'.
2022-07-19 12:42:49,262 - data.featureprocessor - INFO - writing 'data/preprocessed/load_solar.pkl'.
2022-07-19 12:42:49,302 - data.featureprocessor - INFO - adding features to heatpump data... 🥵
2022-07-19 12:42:49,303 - data.featureprocessor - INFO - reading 'data/preprocessed/heatpump_solar.pkl'.
2022-07-19 12:42:49,372 - data.featureprocessor - INFO - finished removing columns with >= 80.0 % pd.NA values.
2022-07-19 12:42:49,787 - data.featureprocessor - INFO - finished replacing pd.NA.
2022-07-19 12:42:49,867 - data.featureprocessor - INFO - finished making 'S_TOT', 'P_TOT', 'Q_TOT'.
2022-07-19 12:42:50,043 - data.featureprocessor - INFO - writing 'data/preprocessed/heatpump_solar.pkl'.
2022-07-19 12:42:50,062 - __main__ - INFO - All done! ✨ 🍰 ✨

```


### Set-up pre-commit hooks
```
pre-commit install
pre-commit run --all-files

>>>
pre-commit run --all-files
isort (python)...........................................................Passed
doc8.................................................(no files to check)Skipped
black....................................................................Passed
trim trailing whitespace.................................................Passed
mixed line ending........................................................Passed
check BOM - deprecated: use fix-byte-order-marker........................Passed
check for merge conflicts................................................Passed
check for broken symlinks............................(no files to check)Skipped
flake8...................................................................Passed
bandit...................................................................Passed
```

## Project Organization

------------

```
 ├── README.md        <-- this file. insert group members here
 ├── .gitignore           <-- prevents you from submitting several clutter files
 ├── data
 │   ├── modeling
 │   │   ├── dev       <-- your development set goes here
 │   │   ├── test       <-- your test set goes here
 │   │   └── train       <-- your train set goes here goes here
 │   ├── preprocessed      <-- your preprocessed data goes here
 │   └── raw        <-- the provided raw data for modeling goes here
 ├── docs        <-- provided explanation of raw input data goes here
 │
 ├── models        <-- dump models here
 ├── notebooks       <-- your playground for juptyer notebooks
 ├── requirements.txt      <-- required packages to run your submission (use a virtualenv!)
 └── src
     ├── additional_features.py    <-- your creation of additional features/data goes here
     ├── predict.py       <-- your prediction script goes here
     ├── preprocessing.py     <-- your preprocessing script goes here
     └── train.py       <-- your training script goes here

```
