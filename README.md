# Federated GI Segmentation

This is an attempt to solve the [UW-Madison GI Tract Image Segmentation](https://www.kaggle.com/competitions/uw-madison-gi-tract-image-segmentation). The idea of the project, is to do multiclass segmentation, and not stop at that. We will try to make the model federated, with the help of [Federated Learning](https://ai.googleblog.com/2017/04/federated-learning-collaborative.html).

## Setup

-   First to run this code, you would have to create a virtual environment.

```commandLine
python -m venv <env-name>
```

-   Then we activate the environment and install the required packages mentioned in the `requirements.txt` and `dev-requirements.txt`.

```commandLine
source <env-name>/bin/activate
pip install -r requirements.txt
pip install -r dev-requirements.txt
```

## Load the Data

-   You will have to install the original dataset from the original link of [UW-Madison GI Tract Image Segmentation](https://www.kaggle.com/competitions/uw-madison-gi-tract-image-segmentation).
-   Then, you will have to update the paths in the `src/config.py`

```py
class Config:
    BASE_DIR = "path/to/data/train/"
    COLORS = {"large_bowel": (1, 0, 0), "small_bowel": (0, 1, 0), "stomach": (0, 0, 1)}
    DATASET_CKPT_DIR = "path/to/store/dataset"
    TRAIN_CSV_PATH = "path/to/train/csv"
    NEW_TRAIN_CSV_PATH = "path/to/new/train/csv"
```

-   Then run the `src/preprocess.py`.

```commandLine
cd src
python preprocess.py
```
