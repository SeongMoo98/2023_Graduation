# -*- coding: utf-8 -*-
import os
import tornado.wsgi
import tornado.httpserver
import time
import optparse
import logging
import flask
from flask import jsonify
from mapmatcher import MapMatcher

import numpy as np

app = flask.Flask(__name__, static_url_path='/static')
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0

@app.route('/')
def index():
    return flask.render_template('demo.html')

@app.route('/demo')
def demo():
    return flask.render_template('demo.html')

@app.route('/match_wkt', methods=['GET'])
def match_url():
    latitude = float(flask.request.args.get('latitude', ''))
    longitude = float(flask.request.args.get('longitude', ''))
    wkt = "POINT({} {})".format(longitude, latitude)  # 입력된 위도와 경도를 WKT(Point) 형식으로 변환
    starttime = time.time()
    result = app.mapmatcher.match_wkt(wkt)
    mgeom_wkt = ""
    if result.mgeom.get_num_points() > 0:
        mgeom_wkt = result.mgeom.export_wkt()
    endtime = time.time()
    if mgeom_wkt != "":
        response_json = {"wkt": mgeom_wkt, "state": 1}
        return jsonify(response_json)
    else:
        return jsonify({"state": 0})

def start_tornado(app, port=5001):
    http_server = tornado.httpserver.HTTPServer(tornado.wsgi.WSGIContainer(app))
    http_server.listen(port)
    print("Tornado server starting on port {}".format(port))
    print("Visit http://localhost:{} to check the demo".format(port))
    tornado.ioloop.IOLoop.instance().start()

def start_from_terminal(app):
    parser = optparse.OptionParser()
    parser.add_option('-d', '--debug', help="enable debug mode", action="store_true", default=False)
    parser.add_option('-p', '--port', help="which port to serve content on", action="store", dest="port", type='int', default=5001)
    parser.add_option('-c', '--config', help="the model configuration file", action="store", dest="config_file", type='string', default="config.json")
    opts, args = parser.parse_args()
    app.mapmatcher = MapMatcher(opts.config_file)
    if opts.debug:
        app.run(debug=True, host='0.0.0.0', port=opts.port)
    else:
        start_tornado(app, opts.port)

if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO)
    start_from_terminal(app)
