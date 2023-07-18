"""
MySQL에 넣을 Data processing
<oid, 출발지, 도착치, 전체경로, 운행상태(정지/이동), 시간대(새벽, 오전, 오후, 심야)>

※ 각 Column 데이터 구조 설명
oid : 봄, 여름, 가을, 겨울 각각 id 부여
    grouping : (년도, 월, 일, oid)
출발지, 도착지
전체경로 : 출발지, 도착지를 포함한 전체경로
운행상태 : 전체경로에 대해 각각의 Point에 대한 운행상태
    운행상태 선정 기준
        POI에 들어왔을때와 나갔을때의 평균속도?
시간대 : 전체경로에 대해 각각의 Point에 대한 시간대

※Code 진행 흐름
1. month_data 불러오기
2. month_data datetime으로 만들기
3. grouping 하기
4. 각 id별로 dist랑 velocity 구하기
5. trajectory 구하기
"""

 
import pandas as pd
import numpy as np
import math

import pymysql
# from sqlalchemy import create_engine
# pymysql.install_as_MySQLdb()
# import MySQLdb


""" Public Function """


### 제주 데이터허브에서 따온 csv 이름을 2020_1, 2021_2 이런식으로 바꿔서 저장해놨어요
### 그래서 가지고있는 파일 이름으로 변경하거나 csv파일 이름을 바꿔서 2020년 2021년 raw data를 합치는 코드 입니다.


# GPS 데이터(2020년, 2021년)을 합치고 이상치(좌표가 안맞는 GPS, 월이 안맞는 collection_dt) 제거
# 이후 전처리에 필요한 csv 생성
def GPS_data_concat():
    for month in range(1, 13):
        GPS_2020 = pd.read_csv(f"data/2020_{month}.csv")
        GPS_2021 = pd.read_csv(f"data/2021_{month}.csv")
        df = pd.concat([GPS_2020, GPS_2021], axis=0)
        
        # 결측치 제거
        df = df.dropna(axis=0)
        
        # # 이상치(GPS가 벗어난 데이터, 월이 안맞는 데이터) 제거
        df = _remove_invalid_data(df, month)
        
        df.to_csv(f"month_{month}", index=False)
    pass


def data_preprocessing(season):
    months = []
    db_name = ""
    table_name = ""
    if season == '봄':
        months = [3,4,5]
        db_name = "spring_db"
        table_name = "spring_table"
    elif season == '여름':
        months = [6,7,8]
        db_name = "summer_db"
        table_name = "summer_table"
    elif season == '가을':
        months = [9,10,11]
        db_name = "fall_db"
        table_name = "fall_table"
    elif season == '겨울':
        months = [11,12,13]
        db_name = "winter_db"
        table_name = "winter_table"
    else:
        print("계절을 다시 입력해주세요(봄, 여름, 가을, 겨울)")
        return
    
    POIs = _POI_load(path='data/2021_07_POI_50_transformed.csv')
    
    # mysql workbench에 schema가 없어도 돌아갈지 안돌아갈지는 모르겠는데
    # 안돌아가면 schema 생성하셔서 돌려보세요!
    # 그리고 create schema 할 때 3번째 줄에 Charset/Collation은 utf-8/utf-8-bin 선택하시면 됩니다
    # 또, 테이블은 _save_to_mysql() 내부에서 없으면 생성하게 만들어둬서 안만드셔도 괜찮아요
    for month in months:
        
        df = _GPS_data_load(path=f"data/month_{month}.csv")
        
        trajectories = _make_trajectories(df, POIs)
        
        _save_to_mysql(trajectories, db_name, table_name)
        # _save_to_csv(trajectories, path=f"data/trajectory_{month}")
    
    pass
    
""" Private Function """
def _save_to_mysql(trajectories, db_name, table_name):
    
    trajectories = trajectories.astype(str)
    
    # DB 정보
    host = "localhost"
    user = "root"
    password = "0000"
    
    conn = pymysql.connect(host=host, user=user, password=password, db=db_name)
    curs = conn.cursor(pymysql.cursors.DictCursor)
    
    try:
        # 테이블이 이미 존재하는지 확인
        table_exists = False
        cursor = conn.cursor()
        cursor.execute(f"SHOW TABLES LIKE '{table_name}'")
        result = cursor.fetchone()
        
        if result:
            table_exists = True
    
        # 테이블이 존재하지 않는 경우에만 CREATE TABLE 문 실행
        if not table_exists:
            create_table_sql = f"""CREATE TABLE `{db_name}`.`{table_name}` (
                `trajectory_id` VARCHAR(500) NOT NULL,
                `start_point` VARCHAR(500) NULL,
                `end_point` VARCHAR(500) NULL,
                `path` VARCHAR(500) NULL,
                `time_period` VARCHAR(500) NULL,
                `status` VARCHAR(500) NULL,
                PRIMARY KEY (`trajectory_id`));"""
            cursor.execute(create_table_sql)
    
        # INSERT INTO 문 실행
        insert_sql = f"INSERT INTO {table_name} (trajectory_id, start_point, end_point, path, time_period, status) VALUES (%s, %s, %s, %s, %s, %s)"
        for idx in range(len(trajectories)):
            curs.execute(insert_sql, tuple(trajectories.values[idx]))
        
        # 커밋
        conn.commit()
    
    except Exception as e:
        print(f"Error: {e}")
    finally:
        curs.close()
        conn.close()


def _save_to_csv(trajectories, path):
    try:
        trajectories.to_csv(path, encoding='utf-8-sig', index = False)
    except Exception as e:
        print(f"error : {e}")
    else:
        print("Save success")
    
    
    
def _remove_invalid_data(df, month):
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
    
    # GPS 데이터에서 월이 잘못 들어있는 데이터 제거
    # ex) 2020_2.csv에 3월 데이터가 들어가 있다
    df['collection_dt'] = pd.to_datetime(df['collection_dt'].astype(str), format='%Y%m%d%H%M%S%f')
    df = df.drop(df.loc[df['collection_dt'].dt.month != month].index, axis=0)
    
    return df


def _make_trajectories(df, locations):
    
    # DataFrame을 (년, 월, 일, oid)로 grouping
    grouped = df.groupby([df['collection_dt'].dt.year, 
                          df['collection_dt'].dt.month, 
                          df['collection_dt'].dt.day, 
                          df['oid']])
    
    trajectories = pd.DataFrame(columns=['trajectory_id', 'start_point','end_point', 'path', 'time_period','status'])
    concat_row = pd.DataFrame(columns=['trajectory_id', 'start_point','end_point', 'path', 'time_period','status'])
    
    # 5개만 mysql에 넣어보려고 적은 코드입니다
    # 아래에 똑같이 둘러쌓인 코드 지우면 전체 mysql에 넣는 코드가 됩니다
    #********#
    index = 0
    #********#
    
    for group_key, rows in grouped:
        rows.drop_duplicates(subset=['collection_dt'], inplace=True)

        #********#
        if index == 5 :
            break
        #********#
        
        # 한 차에 대해 GPS 점의 개수가 100개 이하면 제거
        if (len(rows) <= 100):
            continue
        # rows
        # 타입 : dataframe
        # column : ['oid', 'collection_dt', 'longitude','latitude', 'distance', 'velocity']
        # value example : 769545(df_index), 0c0000fd, 2020-01-04 06:14:10, 126.241423, 33.394894, 0.009780, 29.072355 
        # 첫 row의 distance와 velocity는 NaN으로 할당
        cal_rows = _calculate_group_distance_and_velocity(rows)
        path = _calculate_path(cal_rows, locations)
        
        removed_path, path_time_period, path_driving_status = _remove_duplicated_path(path, cal_rows)
        
        # 경로가 너무 길거나 0이면 추가 x
        if (len(removed_path) == 0) or (len(removed_path) >= 50):
            continue

        # 타입 : dictionary 
        # Key 형식 example : (2020, 1, 4, '46100c11')
        # value 형식 example : ['POI1', 'POI3', 'POI5', 'POI47']
        start_point, end_point = removed_path[0], removed_path[-1]

        column_range = ['trajectory_id', 'start_point', 'end_point', 'path', 'time_period', 'status']
        concat_row.loc[0, column_range] = [ group_key, start_point, end_point, removed_path, path_time_period, path_driving_status]
        
        trajectories = pd.concat([trajectories, concat_row], ignore_index=True)
    
            
        #********#
        index += 1
        #********#
    return trajectories

# 불러온 month_1에서 이상치 처리하고 grouping한 각각에 group 데이터에 대해서
# 파생변수 distance, velocity, 시간대를 게산하는 함수입니다
def _calculate_group_distance_and_velocity(rows):
    # grouping한 데이터의 rows(데이터프레임)의 distance와 velocity 계산
    long  = rows['longitude'].values
    lat  = rows['latitude'].values
    time_delta = rows['collection_dt'].diff().dropna().values

    # 단위 : km
    dist = [_haversine_distance(long[i], lat[i], long[i+1],lat[i+1]) for i in range(len(rows) -1)]    
    # 단위 : km/h
    velo = [d / (t.astype(float) * 1e-9 / 3600) for d, t in zip(dist, time_delta)]


    # GPS의 첫 data는 이전 점과의 거리, 속도를 구할 수 없다
    dist.insert(0, np.NaN)
    velo.insert(0, np.NaN)

    rows['distance'] = dist
    rows['velocity'] = velo
    
    rows['time_period'] = rows['collection_dt'].dt.time.apply(_map_time_period)
    return rows 
    

# 각 group의 row들에 대해서 가장 가까운 POI점을 구한 path에 대해
# 계속 동일하게 나타나는 지점을 제거하는 함수입니다.
def _remove_duplicated_path(path, cal_rows):
    removed_path = []
    path_time_period = []
    path_driving_status = []

    prev_location = path[0]
    start_idx = 0
    end_idx = len(path)-1

    for idx, location in zip(range(len(path)), path):
        
        if location != prev_location:
            end_idx = idx

            same_POI_rows = cal_rows.iloc[start_idx:end_idx]
            
            mean_velocity = _map_velocity_state(same_POI_rows['velocity'].mean())
            # 시간대가 섞여있는 데이터가 있을 수도 있어서 가장 많이 나온 시간대를 선정했습니다.
            period_time = same_POI_rows['time_period'].value_counts().idxmax()
            prev_location = location

            removed_path.append(location)
            path_time_period.append(period_time)
            path_driving_status.append(mean_velocity)
            
            prev_location = location
            start_idx = idx
        
    return removed_path, path_time_period, path_driving_status


# Haversine 공식을 이용한 Trajectory 생성
# 관심지점 - GPS 데이터 간 거리 1km 이내
def _calculate_path(rows, locations):
    path = []
    for df_index, row in rows.iterrows():
        GPS_lon = row['longitude']
        GPS_lat = row['latitude']
        
        # POI_name : POIx
        # 모든 POI와의 거리를 계산하고 가장 거리가 가까운 POI 선정
        distance = [_haversine_distance(GPS_lon, GPS_lat, POI_lon, POI_lat) for POI_name, (POI_lon, POI_lat) in locations.iterrows()]
        nearest_idx = np.argmin(distance)
    
        path.append(locations.index[nearest_idx])
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


# 전처리 된 GPS 데이터(GPS_data_concat) load
# ex) path='data/month_1.csv'
def _GPS_data_load(path):
    # 전처리 된 GPS 데이터
    df = pd.read_csv(path)

    # csv파일로 불러온 시간(Type : str)을 datetime으로 변경(이후 코드에서 datetime 사용)
    df['collection_dt'] = pd.to_datetime(df['collection_dt'])
    df.sort_values(by='collection_dt', ascending=True)
    
    return df

# K-Means Clustering으로 설정한 POI
def _POI_load(path):
    locations = pd.read_csv(path)

    locations = locations[["lon", "lat"]]
    locations.index = [f'POI{i}' for i in range(len(locations))]
    
    return locations



def _map_time_period(time):
    if time < pd.to_datetime('06:00:00').time():
        return '새벽'
    elif time < pd.to_datetime('12:00:00').time():
        return '오전'
    elif time < pd.to_datetime('18:00:00').time():
        return '오후'
    else:
        return '저녁'
    
def _map_velocity_state(velocity):
    if velocity < 10:
        return '정지'
    else:
        return '이동'

