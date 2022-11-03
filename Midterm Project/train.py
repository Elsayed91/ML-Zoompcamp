import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split


def enforce_values(df):
    """
    Removes outliers from the data
    """
    df["trip_duration"] = df["trip_duration"][(df["trip_duration"] > 0)]
    df["passenger_count"] = df["passenger_count"][
        (df["passenger_count"].isin(list(range(1, 8))))
    ]
    df["VendorID"] = df["VendorID"][(df["VendorID"].isin([0, 1, 2]))]
    df["RatecodeID"] = df["RatecodeID"][(df["RatecodeID"].isin([0, 1, 2, 3, 4, 5, 6]))]
    df["payment_type"] = df["payment_type"][
        (df["payment_type"].isin([1, 2, 3, 4, 5, 6]))
    ]
    df = df[(df["fare_amount"] < 250) & (df["fare_amount"] >= 2.5)]
    df = df[df["pickup_lat"] != 0]
    df = df[df["pickup_long"] != 0]
    df = df[df["dropoff_lat"] != 0]
    df = df[df["dropoff_long"] != 0]
    df.dropna(inplace=True)

    return df


def keep_important_cols(df):
    cols = [
        "passenger_count",
        "fare_amount",
        "pickup_long",
        "pickup_lat",
        "dropoff_long",
        "dropoff_lat",
        "trip_duration",
        "day",
        "month",
        "year",
        "day_of_week",
        "hour",
        "distance",
        "pickup_jfk_distance",
        "dropoff_jfk_distance",
        "pickup_ewr_distance",
        "dropoff_ewr_distance",
        "pickup_lga_distance",
        "dropoff_lga_distance",
    ]
    df = df[cols]
    return df


def data_splitter(df, target):
    y = df[target]
    X = df.drop([target], axis=1)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, random_state=1, test_size=0.3
    )
    return X_train, X_test, y_train, y_test


def ny_taxi_geo_merger(df1, df2):
    df_merge = pd.merge(
        df1, df2, how="left", left_on=["PULocationID"], right_on=["LocationID"]
    )
    df_merge.rename(
        columns={"longitude": "pickup_long", "latitude": "pickup_lat"}, inplace=True
    )
    df_merge.drop(["LocationID", "PULocationID", "borough"], axis=1, inplace=True)
    df_merge = pd.merge(
        df_merge, df2, how="left", left_on=["DOLocationID"], right_on=["LocationID"]
    )
    df_merge.rename(
        columns={"longitude": "dropoff_long", "latitude": "dropoff_lat"}, inplace=True
    )
    df_merge.drop(["LocationID", "DOLocationID", "borough"], axis=1, inplace=True)
    return df_merge


def distance(s_lat, s_lng, e_lat, e_lng):

    # approximate radius of earth in km
    R = 6373.0

    s_lat = s_lat * np.pi / 180.0
    s_lng = np.deg2rad(s_lng)
    e_lat = np.deg2rad(e_lat)
    e_lng = np.deg2rad(e_lng)

    d = (
        np.sin((e_lat - s_lat) / 2) ** 2
        + np.cos(s_lat) * np.cos(e_lat) * np.sin((e_lng - s_lng) / 2) ** 2
    )

    return 2 * R * np.arcsin(np.sqrt(d))


def add_distances_from_airport(df):
    # coordinates of all these airports
    jfk_coords = (40.639722, -73.778889)
    ewr_coords = (40.6925, -74.168611)
    lga_coords = (40.77725, -73.872611)

    df["pickup_jfk_distance"] = distance(
        jfk_coords[0], jfk_coords[1], df.pickup_lat, df.pickup_long
    )
    df["dropoff_jfk_distance"] = distance(
        jfk_coords[0], jfk_coords[1], df.dropoff_lat, df.dropoff_long
    )

    df["pickup_ewr_distance"] = distance(
        ewr_coords[0], ewr_coords[1], df.pickup_lat, df.pickup_long
    )
    df["dropoff_ewr_distance"] = distance(
        ewr_coords[0], ewr_coords[1], df.dropoff_lat, df.dropoff_long
    )

    df["pickup_lga_distance"] = distance(
        lga_coords[0], lga_coords[1], df.pickup_lat, df.pickup_long
    )
    df["dropoff_lga_distance"] = distance(
        lga_coords[0], lga_coords[1], df.dropoff_lat, df.dropoff_long
    )

    return df


def extra_features(df):
    df["trip_duration"] = (
        df["tpep_dropoff_datetime"] - df["tpep_pickup_datetime"]
    ).astype("timedelta64[m]")
    df["day"] = df["tpep_pickup_datetime"].dt.day
    df["month"] = df["tpep_pickup_datetime"].dt.month
    df["year"] = df["tpep_pickup_datetime"].dt.year
    df["day_of_week"] = df["tpep_pickup_datetime"].dt.weekday
    df["hour"] = df["tpep_pickup_datetime"].dt.hour
    df["store_and_fwd_flag"] = np.where(df["store_and_fwd_flag"] == "Y", 1, 0)
    df.drop(["tpep_dropoff_datetime", "tpep_pickup_datetime"], inplace=True, axis=1)
    df["distance"] = df.apply(
        lambda x: distance(
            x["pickup_long"], x["pickup_lat"], x["dropoff_long"], x["dropoff_lat"]
        ),
        axis=1,
    )
    add_distances_from_airport(df)
    return df


def prep_data(data_url):
    df = pd.read_parquet(data_url)
    df = ny_taxi_geo_merger(df, df_loc)
    df = extra_features(df)
    df = enforce_values(df)
    df = keep_important_cols(df)
    X_train, X_test, y_train, y_test = data_splitter(df, "fare_amount")
    return X_train, X_test, y_train, y_test


def main(data_url, params):
    X_train, X_test, y_train, y_test = prep_data(data_url)
    model = xgb.XGBRegressor(params=params)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return y_pred[0]


if __name__ == "__main__":

    params = {
        "colsample_bytree": 0.6,
        "gamma": 0.3,
        "max_depth": 4,
        "min_child_weight": 5,
        "n_estimators": 100,
        "subsample": 0.8,
        "tree_method": "gpu_hist",
    }

    main(data_url, params)
