import pandas as pd
from pyproj import Transformer
def make_POI_csv():
    # 원래 좌표계 (EPSG:5179)와 변환할 좌표계 (EPSG:4326)의 Transformer 객체 생성
    transformer = Transformer.from_crs("EPSG:5179", "EPSG:4326", always_xy=True)

    # CSV 파일 로드
    df = pd.read_csv('../data/2021_07_POI_50.csv')

    # 좌표 변환
    df[['lon', 'lat']] = df.apply(lambda row: transformer.transform(row['x'], row['y']), axis=1).apply(pd.Series)

    # 변환된 좌표로 업데이트된 CSV 파일 저장
    df.to_csv('../data/2021_07_POI_50_transformed.csv', index=False)
    pass

    
