#!/usr/bin/env python3
import modules.VisionModule  as vision
from flask import Flask, render_template, request, Response,jsonify,redirect,url_for
from waitress import serve
import threading
import queue
import cv2
import os
import sys

floodQ = queue.Queue()
parametersGlobal={}

if getattr(sys, 'frozen', False):
    template_folder = os.path.join(sys._MEIPASS, 'templates')
    static_folder = os.path.join(sys._MEIPASS, 'static')
    app = Flask(__name__, template_folder=template_folder, static_folder=static_folder)
else:
    app = Flask(__name__)

#Template Routes
@app.route('/')
@app.route('/counter',methods=["GET","POST"])
def index():
    global parametersGlobal
    vision.roiState = False
    if not vision.startup:
        parametersGlobal = vision.loadParameters()
        vision.startup = True
    vision.mode = "counter"
    vision.x,vision.y,vision.w, vision.h = vision.cx,vision.cy,vision.cw, vision.ch
    return render_template("index.html",
                          activeMode=vision.mode.upper(),
                          bin=parametersGlobal["counter"]["binarization"],
                          bright=parametersGlobal["counter"]["brightness"],
                          minArea=parametersGlobal["counter"]["minArea"],
                          maxArea=parametersGlobal["counter"]["maxArea"],
                          eros=parametersGlobal["counter"]["erosion"],
                          lineWidth=parametersGlobal["counter"]["lineWidth"])


@app.route("/presence",methods=["GET","POST"])
def presence():
    vision.mode = "presence"
    vision.x,vision.y,vision.w, vision.h = vision.px,vision.py,vision.pw, vision.ph
    print(f"Mode changed:{vision.mode}")
    return render_template("presence.html",
                          activeMode=vision.mode.upper(),
                          bin=parametersGlobal["presence"]["binarization"],
                          bright=parametersGlobal["presence"]["brightness"],
                          maxDif=parametersGlobal["presence"]["maxDif"])


@app.route("/rgb",methods=["GET","POST"])
def rgb():
    vision.mode = "rgb"
    vision.x,vision.y,vision.w, vision.h = vision.rx,vision.ry,vision.rw, vision.rh
    print(f"Mode changed:{vision.mode}")
    return render_template("rgb.html",
                          activeMode=vision.mode.upper(),
                          bright=parametersGlobal["rgb"]["brightness"],
                          maxRGB=parametersGlobal["rgb"]["maxRGB"],
                          minRGB=parametersGlobal["rgb"]["minRGB"],
                          refRGB=parametersGlobal["rgb"]["refRGB"])


#Buttons,Sliders
@app.route("/param_change", methods=["GET","POST"])
def param_change():
    print("Parameters changed")
    global parametersGlobal
    if request.method == 'POST':
        if vision.mode == "counter":
            parametersGlobal["counter"]["binarization"] = int(request.form["binarization"])
            parametersGlobal["counter"]["brightness"] = int(request.form["brightness"])
            parametersGlobal["counter"]["minArea"] = int(request.form["minArea"])
            parametersGlobal["counter"]["maxArea"] = int(request.form["maxArea"])
            parametersGlobal["counter"]["erosion"] = int(request.form["erosion"])
            parametersGlobal["counter"]["lineWidth"] = int(request.form["lineWidth"])
            return jsonify({'binarization': parametersGlobal[vision.mode]["binarization"],
                            'brightness': parametersGlobal[vision.mode]["brightness"],
                            'minArea': parametersGlobal[vision.mode]["minArea"],
                            'maxArea': parametersGlobal[vision.mode]["maxArea"],
                            'erosion': parametersGlobal[vision.mode]["erosion"],
                            'lineWidth': parametersGlobal[vision.mode]["lineWidth"]})
        if vision.mode == "presence":
            parametersGlobal["presence"]["binarization"] = int(request.form["binarization"])
            parametersGlobal["presence"]["brightness"] = int(request.form["brightness"])
            parametersGlobal["presence"]["maxDif"] = int(request.form["maxDif"])
            return jsonify({'binarization': parametersGlobal["presence"]["binarization"],
                            'brightness': parametersGlobal["presence"]["brightness"],
                            'maxDif':parametersGlobal["presence"]["maxDif"]})
        if vision.mode == "rgb":
            parametersGlobal["rgb"]["brightness"] = int(request.form["brightness"])
            parametersGlobal["rgb"]["maxRGB"] = [int(n) for n in request.form["maxRGB"].split(",")]
            parametersGlobal["rgb"]["minRGB"] = [int(n) for n in request.form["minRGB"].split(",")]
            parametersGlobal["rgb"]["refRGB"] = [int(n) for n in request.form["refRGB"].split(",")]
            return jsonify({'brightness': parametersGlobal["rgb"]["brightness"],
                            'maxRGB':",".join([str(n) for n in parametersGlobal["rgb"]["maxRGB"]]),
                            'minRGB':",".join([str(n) for n in parametersGlobal["rgb"]["minRGB"]]),
                            'refRGB':",".join([str(n) for n in parametersGlobal["rgb"]["refRGB"]])
                            })



@app.route("/param_save", methods=["GET","POST"])
def param_save():
    global parametersGlobal
    parametersGlobal[vision.mode]["roi"] = [vision.x,vision.y,vision.w, vision.h]
    if vision.saveParameters(parametersGlobal):
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

@app.route("/trigger",methods=["POST"])
def trigger():
    if vision.mode == "presence":
        vision.trigger = True
    return jsonify({})

@app.route("/crop_params",methods=["POST"])
def crop_params():
    if vision.mode=="counter":
        vision.cx = int(float(request.form["x"]))
        vision.cy = int(float(request.form["y"]))
        vision.cw = int(float(request.form["w"]))
        vision.ch = int(float(request.form["h"]))
    elif vision.mode=="presence":
        vision.px = int(float(request.form["x"]))
        vision.py = int(float(request.form["y"]))
        vision.pw = int(float(request.form["w"]))
        vision.ph = int(float(request.form["h"]))
    elif vision.mode=="rgb":
        vision.rx = int(float(request.form["x"]))
        vision.ry = int(float(request.form["y"]))
        vision.rw = int(float(request.form["w"]))
        vision.rh = int(float(request.form["h"]))
    return jsonify({})


#Data Aquisition
@app.route('/presenceDif',methods=["POST"])
def presenceDif():
    value = f"{vision.presenceDifPercentage:.2f}"
    return jsonify({'value':value,#str(vision.presenceDifPercentage),
                    'result':"APROVADO" if vision.approved else "REPROVADO"})

@app.route('/counterEntrys',methods=["POST"])
def counterEntrys():
    return jsonify({'value':str(vision.entradas)})

@app.route('/currentRGB',methods=["POST"])
def currentRGB():
    redValue = f"{vision.redValue:.0f}"
    greenValue = f"{vision.greenValue:.0f}"
    blueValue = f"{vision.blueValue:.0f}"
    return jsonify({'r':redValue,
                    'g':greenValue,
                    'b':blueValue,
                    'result':"APROVADO" if vision.approved else "REPROVADO"})


#Streaming
@app.route('/video_feed')
def video_feed():
    global floodQ,parametersGlobal
    if vision.mode == "counter":
        return Response(vision.countObjects(floodQ,parametersGlobal["counter"]),
                        mimetype='multipart/x-mixed-replace; boundary=frame')
    elif vision.mode == "presence":
        return Response(vision.checkPresence(floodQ,parametersGlobal["presence"],device=1),
                mimetype='multipart/x-mixed-replace; boundary=frame')
    elif vision.mode == "rgb":
        return Response(vision.rgbValueCheck(floodQ,parametersGlobal["rgb"],device=1),
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
    #app.run(threaded=True)
    serve(app, host='0.0.0.0', port=5555,threads=8)