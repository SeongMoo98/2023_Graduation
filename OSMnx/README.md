Web demo of FMM

To run this demo, you need to first install the python api of fmm.

fmm
numpy
flask
tornado: for Python 2.7, 5.1 version is required.
Install with pip

pip install --upgrade pip
pip install numpy flask tornado==5.1
Download routable shapefile

Download a routable network shapefile (with id, source, target fields) from OSM using osmnx following the tutorial osm map matching.

An example dataset is provided as data.tar.gz.

# Extract the dataset
tar -xvf data/data.tar.gz -C data
Run the fmm web demo

# Precompute UBODT file
ubodt_gen --network data/edges.shp --network_id fid --source u --target v \
--delta 0.03 -o data/ubodt.txt --use_omp
# Start the web app
python web_demo.py -c fmm_config.json
Run the stmatch web demo

# Start the web app
python web_demo.py -c stmatch_config.json
Visit http://localhost:5000/demo in your browser.

You should be able to see

demo