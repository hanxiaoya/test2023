# coding: UTF-8
# set FLASK_ENV=development
from flask import Flask, request, jsonify
import json
from gevent import pywsgi

app = Flask(__name__)

# @app.route("/")
# def hello_world():
#     return "<p>Hello, World!</p>"

# # GET接口测试
# @app.route('/api/get', methods=['GET', 'POST'])
# def get():
#     name = request.args.get('name', '')
#     if name == 'hxy':
#         age = 18
#     else:
#         age = 'valid name'
#     return jsonify(data={'name': name, 'age': age}, extra={'total': '120'})

# POST接口测试
@app.route('/api/post', methods=['POST'])
def post():
    data = request.get_json()
    print(data)
    return jsonify(data=json.dumps(data), extra={'message': 'success'})

if __name__ == '__main__':
    app.run(debug=True)
#     server = pywsgi.WSGIServer(('0.0.0.0', 12345), app)
#     server.serve_forever()