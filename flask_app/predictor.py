__author__ = "Vipul J"

__version__ = "0.0.1"
__maintainer__ = "Vipul J"
__email__ = "messagevipul@gmail.com"


import pandas as pd
from pprint import pprint
import json
from joblib import load
import geopy.distance


# Fields
cat_dummies = set()
# Model
clf = None
scaler = None
manager_scores = None


def get_feature_list():
    return ['bathrooms',
 'bedrooms',
 'latitude',
 'longitude',
 'price',
 'num_photos',
 'num_features',
 'num_description_words',
 'created_day_of_month',
 'is_weekday',
 'manager_score',
 'dist_from_ctr',
 'has_display_address',
 'has_street_address',
 'total_rooms',
 'price_per_total_rooms',
 'created_month__5',
 'created_hour__1',
 'created_hour__2',
 'created_hour__8',
 'created_hour__23',
 'created_hour__19',
 'created_hour__0',
 'created_hour__9',
 'created_hour__12',
 'created_hour__3',
 'created_hour__21',
 'created_hour__4',
 'created_hour__5',
 'created_hour__22',
 'created_hour__10',
 'created_month__4',
 'created_hour__13',
 'created_hour__16',
 'created_hour__20',
 'created_hour__11',
 'created_hour__17',
 'created_hour__18',
 'created_hour__6',
 'created_hour__14',
 'created_month__6',
 'created_hour__15',
 'created_hour__7',
 'has_elevator',
 'has_cats_allowed',
 'has_hardwood_floors',
 'has_dogs_allowed',
 'has_doorman',
 'has_dishwasher',
 'has_no_fee',
 'has_laundry_in_building',
 'has_fitness_center',
 'has_pre-war',
 'has_laundry_in_unit',
 'has_roof_deck',
 'has_outdoor_space',
 'has_dining_room',
 'has_high_speed_internet',
 'has_balcony',
 'has_swimming_pool',
 'has_new_construction',
 'has_terrace',
 'has_exclusive',
 'has_loft',
 'has_garden/patio',
 'has_wheelchair_access',
 'has_common_outdoor_space']


def read_scaler():
    global scaler
    scaler = load('data/scaler.save')
    if scaler is None:
        # Throw Exception
        print('Loaded scaler is Null')


def read_manager_scores():
    global manager_scores
    manager_scores = json.load(open('data/manager_scores.json'))
    if manager_scores is None:
        # Throw error
        print("Loaded manager scores are Null")


def read_precomputes():
    read_manager_scores()
    read_scaler()


def read_model():
    global clf
    clf = load("data/model.joblib")
    # print(clf.feature_importances_)
    if clf is None:
        # Throw Exception
        print('Loaded model is Null')


def predict(test_df):
    test_df = feature_engineering(test_df)

    X_test = test_df[get_feature_list()]

    # X_test = scaler.transform(X_test)
    y = clf.predict_proba(X_test)

    labels2idx = {label: i for i, label in enumerate(clf.classes_)}

    sub = pd.DataFrame()
    sub["listing_id"] = test_df["listing_id"]
    for label in ["high", "medium", "low"]:
        sub[label] = y[:, labels2idx[label]]

    return sub


def feature_engineering(df):
    df["num_photos"] = df["photos"].apply(len)
    df["num_features"] = df["features"].apply(len)
    df["num_description_words"] = df["description"].apply(lambda x: len(x.split(" ")))

    # Create date month year
    df["created"] = pd.to_datetime(df["created"])
    df["created_month"] = df["created"].dt.month
    df["created_day"] = df["created"].dt.day
    df["created_hour"] = df["created"].dt.hour
    df["created_day_of_week"] = df["created"].dt.dayofweek
    df["created_day_of_month"] = df["created"].dt.day
    df['is_weekday'] = ((df.created_day_of_week) // 5 == 1).astype(float)
    df = df.apply(apply_manager_scores, axis=1)
    df = df.apply(price_per_bedroom, axis=1)
    df = process_home_features(df)
    df = df.apply(distance_from_centre, axis=1)

    # if length is more than 2 - has address
    df['has_display_address'] = df['display_address'].apply(lambda x : 1 if len(x) > 2 else 0)
    df['has_street_address'] = df['street_address'].apply(lambda x : 1 if len(x) > 2 else 0)
    df['total_rooms'] = df['bedrooms'] + df['bathrooms']
    df = df.apply(price_per_total_room, axis=1)

    # One hot encodings
    df = one_hot_encode(df, ['created_hour', 'created_month'])
    # Didn't work
    # df = df.apply(create_has_photos_has_description, axis=1)
    # df = df.apply(price_per_bathroom, axis=1)
    # df = df.apply(bath_bed_ratio, axis=1)
    # df["bed_bath_differnce"] = df['bedrooms'] - df['bathrooms']
    # df["bed_bath_sum"] = df["bedrooms"] + df['bathrooms']

    # Add missing columns in test data - These would be one-hot encoded values

    # only to be called for test_set
    df = add_missing_values(df)
    print("Completed feature engineering")
    return df


def add_missing_values(test_df):
    missing_cols = set(get_feature_list()) - set(test_df.columns)
    # Add a missing column in test set with default value equal to 0
    for c in missing_cols:
        test_df[c] = 0
    return test_df


def one_hot_encode(df, cols):
    # Get one hot encoding of column
    df_processed = pd.get_dummies(df, prefix_sep="__",
                              columns=cols)
    # save all categorical variables
    global cat_dummies
    cat_dummies = set([col for col in df_processed
               if "__" in col
               and col.split("__")[0] in cols])

    return df_processed


def distance_from_centre(row):
    centre = (40.718, -74.008)
    lat_long = (row['latitude'], row['longitude'])
    distance = geopy.distance.geodesic(centre, lat_long).miles
    row['dist_from_ctr'] = distance
    return row


def create_has_photos_has_description(row):
    row['has_photos'] = 1 if row['num_photos'] > 0 else 0
    row['has_description'] = 1 if row['num_description_words'] > 0 else 0
    return row


# Key : Feature in data | Value : column name to be created for category variable
home_features_dict = {'Elevator': 'has_elevator',
 'Cats Allowed': 'has_cats_allowed',
 'Hardwood Floors': 'has_hardwood_floors',
 'Dogs Allowed': 'has_dogs_allowed',
 'Doorman': 'has_doorman',
 'Dishwasher': 'has_dishwasher',
 'No Fee': 'has_no_fee',
 'Laundry in Building': 'has_laundry_in_building',
 'Fitness Center': 'has_fitness_center',
 'Pre-War': 'has_pre-war',
 'Laundry in Unit': 'has_laundry_in_unit',
 'Roof Deck': 'has_roof_deck',
 'Outdoor Space': 'has_outdoor_space',
 'Dining Room': 'has_dining_room',
 'High Speed Internet': 'has_high_speed_internet',
 'Balcony': 'has_balcony',
 'Swimming Pool': 'has_swimming_pool',
 'Laundry In Building': 'has_laundry_in_building',
 'New Construction': 'has_new_construction',
 'Terrace': 'has_terrace',
 'Exclusive': 'has_exclusive',
 'Loft': 'has_loft',
 'Garden/Patio': 'has_garden/patio',
 'Wheelchair Access': 'has_wheelchair_access',
 'Common Outdoor Space': 'has_common_outdoor_space'}


def process_home_features(df):
    # Add columns for popular features
    for key, val in home_features_dict.items():
        df[val] = 0

    def update_popular_feature_cols(row):
        features = row['features']
        for feature in features:
            if feature in home_features_dict:
                row[home_features_dict[feature]] = 1

        return row

    df = df.apply(update_popular_feature_cols, axis=1)
    return df


def price_per_total_room(row):
    rooms = row['total_rooms']
    if rooms == 0:
        price_per_total_rooms = 0
    else:
        price_per_total_rooms = row['price'] * 1.00 / rooms
    row['price_per_total_rooms'] = price_per_total_rooms
    return row


def price_per_bedroom(row):
    bedrooms = row['bedrooms']
    if bedrooms == 0:
        price_per_bedroom = 0
    else:
        price_per_bedroom = row['price'] * 1.00 / bedrooms
    row['price_per_bedroom'] = price_per_bedroom
    return row


def apply_manager_scores(row):
    manager_id = row['manager_id']

    if manager_id in manager_scores:
        row['manager_score'] = sum(manager_scores[manager_id])/len(manager_scores[manager_id])
    else:
        row['manager_score'] = 0

    return row


def remove_outliers(df):
    # standard deviation threshold
    sd_threshold = 1

    # Remove price outliers
    df = df[(df.price <= 15000) & (df.price >= 1000)]

    # Remove dist from city centre outliers
    # apporimate radius from city centre
    NYC_RADIUS = 20
    df = df[(df.dist_from_ctr <= 20)]
    return df
