import numpy as np
import pandas as pd
import math

# CSV파일에서 GPS데이터 load
def _GPS_data_load(path):
    # GPS 데이터
    df = pd.read_csv(path)
    df.set_index('oid', inplace=True)
    
    # collection_dt를 "yyyy-mm-dd HH:MM:SS" 형식으로 변경
    df['collection_dt'] = pd.to_datetime(df['collection_dt'].astype(str), format='%Y%m%d%H%M%S%f')
    df['collection_dt'] = df['collection_dt'].dt.strftime('%Y-%m-%d %H:%M:%S')
    
    # 결측치 제거(결측치 없다)
    df = df.dropna()
    
    # # 이상치 제거
    # 사분위수 사용 x
    # df = remove_outliers(df, "longitude")
    # df = remove_outliers(df, "latitude")

    df = _remove_invalid_data(df)
    
    return df

# K-Means Clustering으로 설정한 POI
def _POI_load(path):
    locations = pd.read_csv(path)

    locations = locations[["lon", "lat"]]
    locations.index = [f'POI{i}' for i in range(len(locations))]
    
    return locations

def _make_trajectory(df, locations):
     # 결측치, 이상치 제거한 GPS데이터 경로 생성
    data_dict = {}

    # 그룹화한 데이터프레임에서 oid별로 데이터를 추출하여 딕셔너리에 추가
    grouped = df.groupby('oid')
    for oid, group in grouped:
        data_dict[oid] = group
    
    trajectories = {}
    for oid, rows in data_dict.items():
        trajectories[oid] = _calculate_path(rows, locations)

    # 계속 반복되는 지점을 하나로
    for oid, trajectory in trajectories.items():
        new_trajectory = []
        prev_location = None
        
        for location in trajectory:
            if location != prev_location:
                new_trajectory.append(location)
                prev_location = location
        trajectories[oid] = new_trajectory
        
    return trajectories
        
def _remove_invalid_data(df):
    # Lat
    top = 33.567186
    bottom = 33.112476
        
    # Long
    left = 126.143480
    right = 126.973814

    df = df[(df['longitude'] > left) & 
            (df['longitude'] < right) &
            (df['latitude'] > bottom) & 
            (df['latitude'] < top)]
    return df

# Haversine 공식을 이용한 Trajectory 생성
# 관심지점 - GPS 데이터 간 거리 1km 이내
def _calculate_path(rows, locations):
    # rows : data_dict.value의type 은 dataframe
    path = []
    for _, row in rows.iterrows():
        longitude = row['longitude']
        latitude = row['latitude']
        for location, row in locations.iterrows():
            distance = _haversine_distance(longitude, latitude, row['lon'], row['lat'])
            if distance <= 1:
                path.append(location)
                break
    return path
    
# Haversine 공식 : 위도, 경로 간 거리 구하기
def _haversine_distance(lon1, lat1, lon2, lat2):
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



def _print_trajectory(trajectories):
    for oid, trajectory in trajectories.items():
        if trajectory == []:
            continue
        print(oid,":",trajectory)   

def trajectory_csv():
    # GPS 데이터
    df = _GPS_data_load("../data/2021_07.csv")
    df.to_csv("../data/2021_07_processed_data.csv",index=False)
    df = df[:50000]
    # 관심지점
    locations = _POI_load("../data/2021_07_POI_50_transformed.csv")
    
    # 경로 생성
    trajectory = _make_trajectory(df, locations)
    pd.DataFrame({'car_id':trajectory.keys(), 'trajectory':trajectory.values()}).to_csv("../data/trajectory.csv", index=False)
    pass


# if __name__ == "__main__":
    
#     # GPS 데이터
#     df = GPS_data_load("data/2021_07.csv")
#     df.to_csv("data/processed_data.csv",index=False)
#     df = df[:50000]
#     # 관심지점
#     locations = POI_load("data/2021_07_POI_50_transformed.csv")
    
#     # 경로 생성
#     trajectory = make_trajectory(df, locations)
#     trajectory_to_csv(trajectory)
#     # print_trajectory(trajectory)

    








# # 벡터화시켜서 계산?
# def calculate_path(rows, locations):
#     distances = haversine_distance(rows['longitude'], rows['latitude'], locations['lon'], locations['lat'])
#     within_range = distances <= 1
#     path = locations[within_range].index.tolist()
#     return path

# def haversine_distance(lon1, lat1, lon2, lat2):
#     R = 6371  # 지구의 반지름 (단위: km)
#     lon1_rad = np.radians(lon1)
#     lat1_rad = np.radians(lat1)
#     lon2_rad = np.radians(lon2)
#     lat2_rad = np.radians(lat2)

#     diff_lon = lon2_rad - lon1_rad
#     diff_lat = lat2_rad - lat1_rad

#     a = np.sin(diff_lat / 2) ** 2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(diff_lon / 2) ** 2
#     c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
#     distance = R * c

#     return distance

# def process_group(group):
#     oid = group['oid'].iloc[0]  # oid 값 추출
#     data = group.to_dict('records')  # 그룹의 데이터를 딕셔너리로 변환
#     if oid not in data_dict:
#         data_dict[oid] = []
#     data_dict[oid].extend(data)