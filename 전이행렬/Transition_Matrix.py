import pandas as pd
import numpy as np

if __name__ == "__main__":
    # csv 파일 불러오기
    df_trajectory = pd.read_csv('2020_07_trajectory_100000.csv')
    pois_50 = pd.read_csv("K_Means_Clustering/2021_07_50_transformed.csv")

    # transition_matrix : transition 횟수 count 저장, transition_probability : transition 확률 저장
    transition_matrix = np.zeros((len(pois_50), len(pois_50)), dtype=int)
    transition_probability = np.zeros((len(pois_50), len(pois_50)))
    total = 0

    # transition 횟수 count
    for index, trajectory in df_trajectory.iterrows():
        sequence = trajectory[1]
        sequence = sequence.strip("[]")
        sequence = sequence.split(', ')
        for i in range(len(sequence) - 1):
            sequence[i] = sequence[i].strip("'")
            point = int(sequence[i][3:]) - 1
            sequence[i + 1] = sequence[i + 1].strip("'")
            next_point = int(sequence[i + 1][3:]) - 1
            transition_matrix[point, next_point] += 1
            total += 1

    # 열의 이름 생성
    columns = ["POI" + str(i) for i in range(0, len(pois_50))]

    # 행의 이름 생성
    index = ["POI" + str(i) for i in range(0, len(pois_50))]

    # DataFrame 생성
    df_transitions = pd.DataFrame(transition_matrix, index=index, columns=columns)

    # transition probability 계산
    for i in range(len(pois_50)):
        for j in range(len(pois_50)):
            transition_probability[i][j] = transition_matrix[i][j] / total
    df = pd.DataFrame(transition_probability, index=index, columns=columns)

    # transition matrix csv 파일 생성
    df_transitions.to_csv("2020_07_transition_matrix.csv")
    df.to_csv("2020_07_transition_probability_matrix.csv")