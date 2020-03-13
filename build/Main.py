import modules.VisionModule  as vision
from flask import Flask, render_template, request, Response,jsonify,redirect,url_for
import threading
import queue
import pickle
import cv2

floodQ = queue.Queue()
parameters={}


app = Flask(__name__)

#Inicialization
@app.route('/')
def index():
    global parameters
    vision.roiState = False
    parameters = vision.loadParameters()
    vision.startup = True
    return render_template("index.html",
                          bin=parameters["binarization"],
                          bright=parameters["brightness"],
                          minArea=parameters["minArea"],
                          maxArea=parameters["maxArea"],
                          eros=parameters["erosion"],
                          lineWidth=parameters["lineWidth"])


#Buttons,Sliders
@app.route("/param_change", methods=["GET","POST"])
def param_change():
    print("Parameters changed")
    global parameters
    if request.method == 'POST':
        parameters["binarization"] = int(request.form["binarization"])
        parameters["brightness"] = int(request.form["brightness"])
        parameters["minArea"] = int(request.form["minArea"])
        parameters["maxArea"] = int(request.form["maxArea"])
        parameters["erosion"] = int(request.form["erosion"])
        parameters["lineWidth"] = int(request.form["lineWidth"])
    return jsonify({'binarization': parameters["binarization"],
                    'brightness': parameters["brightness"],
                    'minArea': parameters["minArea"],
                    'maxArea': parameters["maxArea"],
                    'erosion': parameters["erosion"],
                    'lineWidth': parameters["lineWidth"]})


@app.route("/param_save", methods=["GET","POST"])
def param_save():
    global parameters
    parameters["roi"] = [vision.x,vision.y,vision.w, vision.h]
    if vision.saveParameters(parameters):
        print("Save done")
        return jsonify({'msg':'Parâmetros salvos com sucesso!'})
    else:
        return jsonify({'msg':'!!!Falha ao salvar parâmetros!!!'})

@app.route("/counter_reset",methods=["POST"])
def counter_reset():
    print("Counter reset route")
    vision.entradas = 0
    return jsonify({'msg':'Contador Resetado'})

@app.route("/select_roi",methods=["POST"])
def select_roi():
    vision.roiState = True
    return jsonify({})

@app.route("/crop_params",methods=["POST"])
def crop_params():
    vision.x = int(float(request.form["x"]))
    vision.y = int(float(request.form["y"]))
    vision.w = int(float(request.form["w"]))
    vision.h = int(float(request.form["h"]))
    return jsonify({})

#Streaming
@app.route('/video_feed')
def video_feed():
    global floodQ,parameters
    """Video streaming route. Put this in the src attribute of an img tag."""
    return Response(vision.countObjects(floodQ,parameters),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


def floodGen():
    while True:
        global floodQ
        floodedFrame = floodQ.get()
        yield (b'--frame\r\n'
            b'Content-Type: image/jpeg\r\n\r\n' + bytearray(floodedFrame) + b'\r\n')

@app.route('/flood')
def flood():
    return Response(floodGen(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


    

if __name__ == "__main__":
    app.run(threaded=True)