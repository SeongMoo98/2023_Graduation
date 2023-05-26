import numpy as np
import pandas as pd
import math
import csv

# 제주도 상하좌우 위경도(Google Maps에서 측정)
def remove_invalid_data(df):
    # Lat
    top = 33.567186
    bottom = 33.112476
        
    # Long
    left = 126.143480
    right = 126.973814

    df = df[ ( (df['longitude'] > left) & (df['longitude'] < right) ) &
                     ( (df['latitude'] > bottom) & (df['latitude'] < top) )                    
                    ]
    return df

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

def calculate_path(rows, locations):
    # rows : data_dict.value type 은 dataframe
    path = []
    for _, row in rows.iterrows():
        longitude = row['longitude']
        latitude = row['latitude']
        for location, row in locations.iterrows():
            distance = haversine_distance(longitude, latitude, row['lon'], row['lat'])
            if distance <= 1:
                path.append(location)
                break
    return path

if __name__ == "__main__":
    # 관심지점
    # location(DataFrame)
    locations = pd.read_csv("K_Means_Clustering/2021_07_50_transformed.csv")

    locations = locations[["lon", "lat"]]
    locations.index = [f'POI{i}' for i in range(len(locations))]

    df = pd.read_csv("2021_07.csv",)

    # 데이터 타입 설정 # 
    df['oid'] = df['oid'].apply(str)
    df['collection_dt'] = df['collection_dt'].apply(str)
    df['longitude'] = df['longitude'].apply(float)
    df['latitude'] = df['latitude'].apply(float)

    df.set_index('oid', inplace=True)

    # collection_dt를 "yyyy-mm-dd HH:MM:SS" 형식으로 변경
    # 나중에 날짜별로 그룹핑 해야할 수도 있어서 일단 형식만 맞춰 놓을게요 
    df['collection_dt'] = pd.to_datetime(df['collection_dt'].astype(str), format='%Y%m%d%H%M%S%f')
    df['collection_dt'] = df['collection_dt'].dt.strftime('%Y-%m-%d %H:%M:%S')

     # 결측치 제거
     # df.info()로 확인해보면 결측치는 없지만
     # 다른 엑셀파일에는 있을수도 있어서 결측치 제거해줬습니다.
    df = df.dropna()

    df = remove_invalid_data(df)

    # data_dict 구조 : {'oid' : rows(DataFrame), ...}
    data_dict = {}

    # 그룹화한 데이터프레임에서 oid별로 데이터를 추출하여 딕셔너리에 추가
    grouped = df.groupby('oid')

    for oid, group in grouped:
        data_dict[oid] = group

    # ****************************************** # 
    # 여기가 오래걸려요
    # 100,000개로 돌렸을때 5분?
    # path 구조 : {'oid': [중복제거되지 않은 경로], ...}
    paths = {}

    for oid, rows in data_dict.items():
        paths[oid] = calculate_path(rows, locations)

    # ****************************************** #

    # path = {'oid' : ['POI1', 'POI1' ... ,'POI3','POI3', 'POI3', ...] , ...}
    # ==> path = {'oid' : ['POI1','POI3', ...] , ...}
    # 계속 중복되는 경로 제거
    for oid, path in paths.items():
        new_path = [location for i, location in enumerate(path) 
                    if i == 0 or location != path[i-1]]
        paths[oid] = new_path

    # # 출력코드입니다
    # for oid, path in paths.items():
    #     if path == []:
    #         continue
    #     print(oid+":",path)

    # 결과를 csv파일로 저장
    with open('trajectory.csv', 'w') as f:  
        writer = csv.writer(f)
        for key, value in paths.items():
            key = str(key)
            writer.writerow([key, value])