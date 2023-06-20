import GPS_preprocessing
import POI_clustering
import Transition_Matrix


import time
if __name__ == "__main__":
    start_time = time.time()
    POI_clustering.make_POI_csv()
    
    GPS_preprocessing.trajectory_csv()
    
    Transition_Matrix.make_transition_matrix_csv()
    end_time = time.time()
    
    print(f"실행시간 {round(end_time-start_time,2)} s")
    pass