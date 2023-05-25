import numpy as np
import pandas as pd
import math

# 사분위수 이상치 제거
# def remove_outliers(df, column_name):
#     Q1 = df[column_name].quantile(0.25)
#     Q3 = df[column_name].quantile(0.75)
    
#     IQR = Q3 - Q1
    
#     lower_bound = Q1 - 1.5 * IQR
#     upper_bound = Q3 + 1.5 * IQR
    
#     removed_df = df[ (df[column_name] >= lower_bound) & (df[column_name] <= upper_bound) ]
#     return removed_df


def remove_invalid_data(df):
    # Lat
    top = 33.567186
    bottom = 33.112476
        
    # Long
    left = 126.143480
    right = 126.973814

    removed_df = df[ ( (df['longitude'] > left) & (df['longitude'] < right) ) &
                     ( (df['latitude'] > bottom) & (df['latitude'] < top) )                    
                    ]
    return removed_df

# Haversine 공식 : 위도, 경로 간 거리 구하기
def haversine_distance(lon1, lat1, lon2, lat2):
    R = 6371  # 지구의 반지름 (단위: km)

    lon1_rad = math.radians(lon1)
    lat1_rad = math.radians(lat1)
    lon2_rad = math.radians(lon2)
    lat2_rad = math.radians(lat2)

    diff_lon = lon2_rad - lon1_rad
    diff_lat = lat2_rad - lat1_rad

    a = math.sin(diff_lon/2)**2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(diff_lat/2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    distance = R * c

    return distance

# Haversine 공식을 이용한 Trajectory 생성
# 관심지점 - GPS 데이터 간 거리 1km 이내
def calculate_path(rows, locations):
    path = []
    for row in rows:
        longitude = row['longitude']
        latitude = row['latitude']

        # 관심지점과의 거리 계산
        for location, coords in locations.items():
            distance = haversine_distance(longitude, latitude, coords[0], coords[1])
            if distance <= 1:  # 1km 이내인 경우
                path.append(location)
                break
          
    return path


if __name__ == "__main__":
    # 관심지점
    loc_df = pd.read_csv("2021_07_50_transformed.csv")
    loc_df = loc_df[["lon", "lat"]]

    locations = dict()

    index = 0
    for lon, lat in zip(loc_df['lon'], loc_df['lat']):
        locations["POI"+str(index)] = [lon, lat]
        index += 1

    
    # GPS 데이터
    df = pd.read_csv("2021_07.csv")
    
    # collection_dt를 "yyyy-mm-dd HH:MM:SS" 형식으로 변경
    df['collection_dt'] = pd.to_datetime(df['collection_dt'].astype(str), format='%Y%m%d%H%M%S%f')
    df['collection_dt'] = df['collection_dt'].dt.strftime('%Y-%m-%d %H:%M:%S')
    
    # 결측치 제거
    df.dropna()
    
    # # 이상치 제거
    # df = remove_outliers(df, "longitude")
    # df = remove_outliers(df, "latitude")

    df = remove_invalid_data(df)
    
    
    # 결측치, 이상치 제거한 GPS데이터 경로 생성
    data_dict = {}
    for _, row in df.iterrows():
        oid = row['oid']
        if oid not in data_dict:
            data_dict[oid] = []
        data_dict[oid].append(row.to_dict())
        
    
    paths = {}
    for oid, rows in data_dict.items():
        paths[oid] = calculate_path(rows, locations)
        
        
    # 계속 반복되는 지점을 하나로
    for oid, path in paths.items():
        new_path = []
        prev_location = None
        
        for location in path:
            if location != prev_location:
                new_path.append(location)
                prev_location = location
        paths[oid] = new_path
    
    
    # trajectory_df = pd.DataFrame(paths) 
    # print(trajectory_df)
        
    for oid, path in paths.items():
        if path == []:
            continue
        print(oid+":",path)   
    df.to_csv("processed_data.csv",index=False)
