import cv2
import numpy as np
import json
import os
import platform
import threading
import queue

entradas = 0
roiState = False
mode = "counter"
trigger = False
presenceDifPercentage = 0
x=0
y=0
w=200
h=200
startup = False
    
def loadParameters(device=0):
    if platform.system() == 'Linux':
        #path = os.path.join("","home","pi","Documents","MVCounter")
        path = os.path.join("","home",os.getlogin(),"Documents","MVCounter")
    elif platform.system() == 'Windows':
        path = os.path.join("","C:","Users","Public","Documents","MVCounter")  
    if not os.path.exists(path):
        print("MVCounter folder don't exist")
        os.mkdir(path)
        print("Created MVCounter folder")
    if os.path.exists(path):
        print("MVCounter folder already exists")
        if not os.path.exists(os.path.join(path,"parameters.json")):
            print("Parameters file don't exist")
            parameters = {
                "binarization":0,
                "brightness":0,
                "minArea":0,
                "maxArea":0,
                "erosion":0,
                "lineWidth":2,
                "savePath":f"{path}MVcounter/parameters.json",
                "roi":[50,50,200,200]
            }
            file = open(os.path.join(path,"parameters.json"),"w+",encoding="utf-8")
            print("Created parameters file")
            json.dump(parameters,file)
            file.close()
            path = os.path.join(path,"parameters.json")
        else:
            print("Parameters file already exists")
            path = os.path.join(path,"parameters.json")
    
    if os.path.exists(path):
        json_file =  open(path,"r",encoding="utf-8")
        parameters = json.load(json_file)
        json_file.close()
        global x,y,w,h,startup
        if not startup:
            x,y,w,h = parameters["roi"]

    print("Load Done")
    return parameters
    

def saveParameters(parameters):
    try:
        file = open(parameters["savePath"],"w+", encoding="utf-8")
        json.dump(parameters, file)
        file.close()
        return True
    except:
        return False

def countObjects(outQ,parameters,device=0):
    print("OBJECT COUNT ON")
    top = False
    bottom = False
    
    video_capture = cv2.VideoCapture(device)
    video_capture.set(cv2.CAP_PROP_FPS, 60)

    primeiroFrame = None

    oldCenter = []
    global entradas,roiState,x,y,w,h,mode

    while not roiState and mode=="counter":
        binarization = parameters["binarization"]
        brightness = parameters["brightness"]
        minArea = parameters["minArea"]
        maxArea = parameters["maxArea"]
        erosion = parameters["erosion"]
        lineWidth = parameters["lineWidth"]

        ret, image = video_capture.read()
        if not ret:
            break

        video_capture.set(10, brightness)  # brightness
        # both opencv and numpy are "row-major", so y goes first
        cropImg = image[y:y+h, x:x+w]
        cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
        CoordenadaYLinhaEntrada = int((h / 2))

        gray_img = cv2.cvtColor(cropImg, cv2.COLOR_BGR2GRAY)
        gray_img = cv2.GaussianBlur(gray_img, (15, 15), 0)

        if primeiroFrame is None:
            primeiroFrame = gray_img
            continue

        frame_delta = cv2.absdiff(primeiroFrame, gray_img)
        ret, thresh = cv2.threshold(frame_delta, binarization, 255, cv2.THRESH_BINARY)
        mask = thresh.copy()
        alt, larg = thresh.shape[:2]
        floodMask = np.zeros((alt+2, larg+2), np.uint8)
        maskedImg = cv2.bitwise_and(thresh, thresh, mask=mask)
        floodImg = maskedImg.copy()
        cv2.floodFill(floodImg, floodMask, (0, 0), 255)
        floodImg_Inv = cv2.bitwise_not(floodImg)
        floodedImg = maskedImg | floodImg_Inv
        mask2 = floodedImg.copy()
        kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (5, 5))
        mask2 = cv2.erode(mask2, kernel, iterations=erosion)
        floodedImg = cv2.bitwise_and(floodedImg, floodedImg, mask=mask2)

        cv2.line(cropImg,
                (0, CoordenadaYLinhaEntrada),
                (w, CoordenadaYLinhaEntrada), 
                (0, 0, 255), 
                lineWidth)

        contours, hierarchy = cv2.findContours(floodedImg, 
                                               cv2.RETR_EXTERNAL, 
                                               cv2.CHAIN_APPROX_TC89_L1)

        newCenter = []

        for c in contours:
            if cv2.contourArea(c) < minArea or cv2.contourArea(c) > maxArea:
                continue
            M = cv2.moments(c)
            (xc, yc, wc, hc) = cv2.boundingRect(c)

            cv2.rectangle(cropImg, 
                            (xc, yc),
                            (xc + wc, yc + hc), 
                            (0, 255, 0), 2)

            centroidX = 0
            centroidY = 0
            centroid = (centroidX, centroidY)

            if M["m00"] != 0:
                centroidX = int(M["m10"] / M["m00"])
                centroidY = int(M["m01"] / M["m00"])
                centroid = (centroidX, centroidY)
            cv2.circle(cropImg, centroid, 1, (255, 0, 0), 3)
            newCenter = newCenter + [centroidY]

        zipList = zip(newCenter, oldCenter)
        offSet = int(lineWidth/2)
        for a, b in zipList:
            top = False
            bottom = False
            if b <= CoordenadaYLinhaEntrada:
                top = True
            if a >= CoordenadaYLinhaEntrada + offSet:
                bottom = True
            if bottom and top:
                entradas += 1
        oldCenter = newCenter
        cv2.putText(image, "Entradas: {}".format(str(entradas)),
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (250, 0, 1), 2)
        
        #image = cv2.resize(image,(420,336))
        #floodedImg = cv2.resize(floodedImg,(420,336))
        (flag,encodedImg) = cv2.imencode(".jpg", image)
        (_,floodFrameEncoded) = cv2.imencode(".jpg", floodedImg)
        outQ.put(floodFrameEncoded)
        if not flag:
            continue
        yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + bytearray(encodedImg) + b'\r\n')
    if mode == "counter":
        print("CHANGING ROI")
        (_,image) = video_capture.read()
        #image = cv2.resize(image,(420,336))
        (_,encodedImg) = cv2.imencode(".jpg", image)
        yield (b'--frame\r\n'
            b'Content-Type: image/jpeg\r\n\r\n' + bytearray(encodedImg) + b'\r\n')
        video_capture.release()
        roiState = False
    else:
        print(f"MODE CHANGED!NOW {mode.upper()}")
        video_capture.release()


def checkPresence(outQ,parameters,device=0):
    print("PRESENCE CHECK ON")
    global roiState,x,y,w,h,mode,trigger,presenceDifPercentage

    video_capture = cv2.VideoCapture(device)
    video_capture.set(cv2.CAP_PROP_FPS, 60)

    primeiroFrame = None

    while not roiState and mode=="presence":
        binarization = parameters["binarization"]
        brightness = parameters["brightness"]

        ret, image = video_capture.read()
        if not ret:
            break

        video_capture.set(10, brightness)  # brightness
        # both opencv and numpy are "row-major", so y goes first
        cropImg = image[y:y+h, x:x+w]
        cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
        gray_img = cv2.cvtColor(cropImg, cv2.COLOR_BGR2GRAY)
        blur_img = cv2.GaussianBlur(gray_img, (15, 15), 0)
        ret, blur_img = cv2.threshold(blur_img, binarization, 255, cv2.THRESH_BINARY)
        if primeiroFrame is None:
            for i in range(10):
                primeiroFrame = blur_img
            continue
        
        if trigger:
            primeiroFrame = blur_img
            trigger = False
        difImg = cv2.absdiff(primeiroFrame,blur_img)
        meanFirstImgValue = np.mean(primeiroFrame.flatten())
        meanBlurValue = np.mean(blur_img.flatten())
        meanDifImgValue = np.mean(difImg)
        presenceDifPercentage = (meanDifImgValue*100)/255
        #print(f"First Mean: {meanFirstImgValue} Blur Mean:{meanBlurValue} Dif Mean: {meanDifImgValue} / {presenceDifPercentage}%")


        (flag,encodedImg) = cv2.imencode(".jpg", image)
        (_,processedImgEncoded) = cv2.imencode(".jpg",  difImg)
        outQ.put(processedImgEncoded)
        if not flag:
            continue
        yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + bytearray(encodedImg) + b'\r\n')
    if mode == "presence":
        print("CHANGING ROI")
        (_,image) = video_capture.read()
        (_,encodedImg) = cv2.imencode(".jpg", image)
        yield (b'--frame\r\n'
            b'Content-Type: image/jpeg\r\n\r\n' + bytearray(encodedImg) + b'\r\n')
        video_capture.release()
        roiState = False
    else:
        print(f"MODE CHANGED!NOW {mode.upper()}")
        video_capture.release()

