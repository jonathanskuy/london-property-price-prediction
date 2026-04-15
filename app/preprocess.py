import pandas as pd
import numpy as np
from sklearn.neighbors import BallTree
import geopandas as gpd

df_onspd = pd.read_csv('../data/processed/onspd_clean.csv')
df_stops = pd.read_csv('../data/processed/stops_clean.csv')
df_metro = df_stops[df_stops['stoptype'] == 'MET']
df_rail = df_stops[df_stops['stoptype'] == 'RLY']
df_schools = pd.read_csv('../data/processed/schools_clean.csv')
df_good_schools = df_schools[df_schools['Overall effectiveness'].isin([1, 2])]
df_crime = pd.read_csv('../data/processed/crime_clean.csv')
df_crime['geometry'] = gpd.points_from_xy(df_crime.longitude, df_crime.latitude)

CENTRE_LAT = 51.50734
CENTRE_LONG = -0.12765
EARTH_RADIUS = 6371
DISTANCE_RADIUS = 1
DISTANCE_RADIUS_RAD = DISTANCE_RADIUS / EARTH_RADIUS

def haversine_distance(lat1, long1, lat2, long2, radius):
    lat1, long1, lat2, long2 = map(np.radians, [lat1, long1, lat2, long2]) # Converting degrees into radians

    lat_distance = lat2 - lat1
    long_distance = long2 - long1

    angular_separation = np.sin(lat_distance / 2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(long_distance / 2)**2 # Calculating the haversine formula component measuring the angular separation between two points on a spherical surface

    central_angle = 2 * np.arcsin(np.sqrt(angular_separation)) # Calculating the central angle between the two points

    return radius * central_angle

def standardise_postcodes(df, col):
    df[col] = (df[col].astype(str).str.upper().str.strip().str.replace(r'[^A-Z0-9]', '', regex=True))
    df = df[df[col].notna()]
    df = df[df[col] != '']

    postcode_regex = r'^[A-Z]{1,2}[0-9][0-9A-Z]?\s?[0-9][A-Z]{2}'
    df = df[df[col].str.match(postcode_regex)]

    df[col] = df[col].str[:-3] + ' ' + df[col].str[-3:]

    return df

def crime_feature(df, df_crime, crime_category, radius):
    property_coordinates = np.radians(df[['latitude', 'longitude']].values)

    df_crime_category = df_crime[df_crime['crime_type'].isin(crime_category)]

    crime_category_coordinates = np.radians(df_crime_category[['latitude', 'longitude']].values)

    crime_category_tree = BallTree(crime_category_coordinates, metric='haversine')

    radius_rad = radius / EARTH_RADIUS

    return crime_category_tree.query_radius(property_coordinates, r=radius_rad, count_only=True)

def add_lat_long(df):
    df = standardise_postcodes(df, 'postcode')

    df = df.merge(df_onspd[['postcode', 'latitude', 'longitude']], left_on='postcode', right_on='postcode', how='left')

    df = df.drop(columns=['postcode'])

    return df

def add_property_features(df):
    energy_map = {
        'A': 7,
        'B': 6,
        'C': 5,
        'D': 4,
        'E': 3,
        'F': 2,
        'G': 1
    }
    
    df['total_rooms'] = df['bedrooms'] + df['bathrooms'] + df['livingRooms']
    df['bath_bed_ratio'] = df['bathrooms'] / df['bedrooms'].replace(0, 1)
    df['area_per_room_sqm'] = df['floorAreaSqM'] / df['total_rooms']
    df['is_freehold'] = (df['tenure'] == 'Freehold').astype(int)
    df['energy_score'] = df['currentEnergyRating'].map(energy_map)

    df = pd.get_dummies(df, columns=['tenure', 'propertyType', 'currentEnergyRating'], drop_first=True)

    return df

def add_transport_features(df):
    property_coordinates = np.radians(df[['latitude', 'longitude']].values)
    metro_coordinates = np.radians(df_metro[['latitude', 'longitude']].values)
    rail_coordinates = np.radians(df_rail[['latitude', 'longitude']].values)

    metro_tree = BallTree(metro_coordinates, metric='haversine')
    rail_tree = BallTree(rail_coordinates, metric='haversine')
    
    metro_distances, _ = metro_tree.query(property_coordinates, k=1)
    rail_distances, _ = rail_tree.query(property_coordinates, k=1)
    
    df['distance_to_nearest_metro_km'] = metro_distances.flatten() * EARTH_RADIUS
    df['distance_to_nearest_rail_km'] = rail_distances.flatten() * EARTH_RADIUS
    df['total_metro_within_1km'] = metro_tree.query_radius(property_coordinates, r=DISTANCE_RADIUS_RAD, count_only=True)

    return df

def add_school_features(df):
    property_coordinates = np.radians(df[['latitude', 'longitude']].values)
    school_coordinates = np.radians(df_schools[['latitude', 'longitude']].values)
    good_school_coordinates = np.radians(df_good_schools[['latitude', 'longitude']].values)

    school_tree = BallTree(school_coordinates, metric='haversine')
    good_school_tree = BallTree(good_school_coordinates, metric='haversine')

    good_school_distances, _ = good_school_tree.query(property_coordinates, k=1)
    
    indices = school_tree.query_radius(property_coordinates, r=DISTANCE_RADIUS_RAD)
    
    average_ofsted = []

    for i in indices:
        if len(i) == 0:
            average_ofsted.append(np.nan)
        else:
            average_ofsted.append(df_schools.iloc[i]['Overall effectiveness'].mean())

    df['distance_to_nearest_good_school_km'] = good_school_distances.flatten() * EARTH_RADIUS
    df['total_good_Schools_within_1km'] = good_school_tree.query_radius(property_coordinates, r=DISTANCE_RADIUS_RAD, count_only=True)
    df['average_ofsted_score_within_1km'] = average_ofsted

    return df

def add_crime_features(df):
    property_coordinates = np.radians(df[['latitude', 'longitude']].values)
    crime_coordinates = np.radians(df_crime[['latitude', 'longitude']].values)

    crime_tree = BallTree(crime_coordinates, metric='haversine')

    df['total_crime_within_1km'] = crime_tree.query_radius(property_coordinates, r=DISTANCE_RADIUS_RAD, count_only=True)
    
    violent_crimes = ['Violence and sexual offences', 'Robbery', 'Possession of weapons']
    property_crimes = ['Vehicle crime', 'Criminal damage and arson', 'Burglary']
    theft_crimes = ['Other theft', 'Shoplifting', 'Bicycle theft', 'Theft from the person']
    disorder_crimes = ['Anti-social behaviour', 'Public order', 'Drugs']

    df['total_violent_crimes_within_1km'] = crime_feature(df, df_crime, violent_crimes, DISTANCE_RADIUS)
    df['total_property_crimes_within_1km'] = crime_feature(df, df_crime, property_crimes, DISTANCE_RADIUS)
    df['total_theft_crimes_within_1km'] = crime_feature(df, df_crime, theft_crimes, DISTANCE_RADIUS)
    df['total_disorder_crimes_within_1km'] = crime_feature(df, df_crime, disorder_crimes, DISTANCE_RADIUS)

    crime_features = ['total_crime_within_1km', 'total_violent_crimes_within_1km', 'total_property_crimes_within_1km', 'total_theft_crimes_within_1km', 'total_disorder_crimes_within_1km']

    for feature in crime_features:
        df[feature] = np.log1p(df[feature])

    df[crime_features]

    return df

def preprocess_input(df):
    df = add_lat_long(df)
    df = add_property_features(df)
    df = add_transport_features(df)
    df = add_school_features(df)
    df = add_crime_features(df)

    df['distance_to_centre_km'] = haversine_distance(df['latitude'], df['longitude'], CENTRE_LAT, CENTRE_LONG, EARTH_RADIUS)

    df = df.drop(columns=['latitude', 'longitude'])

    return df