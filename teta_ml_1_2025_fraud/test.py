import pandas as pd
import reverse_geocoder as rg
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor


def get_state(coord: list[tuple]):
    try:
        location = rg.search(coord)
        return location
    except:
        return "unknown"


def preprocessing_data_catboost_v2(df_fraud: pd.DataFrame):
    df_train = df_fraud.copy(deep=True)

    df_train_lat_lon = df_train[["lat", "lon"]].drop_duplicates()
    tuples_coords = tuple(df_train_lat_lon.itertuples(index=False, name=None))

    print(get_state(tuples_coords))

    df_train_lat_lon = df_train[["merchant_lat", "merchant_lon"]].drop_duplicates()
    merchan_tuples_coords = tuple(df_train_lat_lon.itertuples(index=False, name=None))

    print(get_state(merchan_tuples_coords))

    return df_train


if __name__ == "__main__":
    # df_fraud = pd.read_csv("./data/train.csv")

    # preprocessing_data_catboost_v2(df_fraud)
    coord = (32.6176, -86.9475), (31.872266, -87.828247)
    print(type(coord))

    print(get_state(coord))
