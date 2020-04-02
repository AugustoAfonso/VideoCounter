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
approved = None
trigger = False
presenceDifPercentage = 0

x=0
y=0
w=200
h=200

cx=0
cy=0
cw=0
ch=0

px=0
py=0
pw=0
ph=0

rx=0
ry=0
rw=0
rh=0
startup = False
    
def loadParameters():
    if platform.system() == 'Linux':
        #path = os.path.join("","home","pi","Documents","MVCounter")
        path = os.path.join("",os.path.expanduser("~"),"Documents","MVCounter")
    elif platform.system() == 'Windows':
        path = os.path.join("","C:\\","Users","Public","Documents","MVCounter")  
    if not os.path.exists(path):
        print("MVCounter folder don't exist")
        os.mkdir(path)
        print("Created MVCounter folder")
    if os.path.exists(path):
        print(f"MVCounter folder already exists at {path}")
        if not os.path.exists(os.path.join(path,"parameters.json")):
            print("Parameters file don't exist")
            parametersCounter = {
                "binarization":0,
                "brightness":0,
                "minArea":0,
                "maxArea":0,
                "erosion":0,
                "lineWidth":2,
                "savePath":os.path.join(path,"parameters.json"),
                "roi":[50,50,200,200]
            }
            parametersPresence = {
                "binarization":0,
                "brightness":0,
                "savePath":os.path.join(path,"parameters.json"),
                "maxDif":0,
                "roi":[50,50,200,200]
            }
            parametersRGB = {
                "binarization":0,
                "brightness":0,
                "maxRGB":"[0,0,0]",
                "savePath":os.path.join(path,"parameters.json"),
                "roi":[50,50,200,200]
            }
            parametersGlobal = {
                "counter":parametersCounter,
                "presence":parametersPresence,
                "rgb":parametersRGB
            }
            file = open(os.path.join(path,"parameters.json"),"w+",encoding="utf-8")
            json.dump(parametersGlobal,file)
            file.close()
            path = os.path.join(path,"parameters.json")
            print(f"Created parameters file at {path}")
        else:
            path = os.path.join(path,"parameters.json")
            print(f"Parameters file already exists at {path}")
    
    if os.path.exists(path):
        json_file =  open(path,"r",encoding="utf-8")
        parametersGlobal = json.load(json_file)
        json_file.close()
        
        global mode,cx,cy,cw,ch,px,py,ph,pw,rx,ry,rw,rh
        
        cx,cy,cw,ch = parametersGlobal["counter"]["roi"]
        px,py,pw,ph = parametersGlobal["presence"]["roi"]
        rx,ry,rw,rh = parametersGlobal["rgb"]["roi"]
        

    print(f"Load Done at path:{path}")
    return parametersGlobal
    

def saveParameters(parametersGlobal):
    global mode
    try:
        file = open(parametersGlobal[mode]["savePath"],"w+", encoding="utf-8")
        json.dump(parametersGlobal, file)
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
        try:
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
        except cv2.error as e:
            print(f"CV2 Exception Caught!!!")
            break
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
        video_capture.release()
        print(f"MODE CHANGED!NOW {mode.upper()}")


def checkPresence(outQ,parameters,device=0):
    print("PRESENCE CHECK ON")
    global roiState,x,y,w,h,mode,trigger,presenceDifPercentage,approved

    video_capture = cv2.VideoCapture(device)
    video_capture.set(cv2.CAP_PROP_FPS, 60)

    primeiroFrame = None

    while not roiState and mode=="presence":
        binarization = parameters["binarization"]
        brightness = parameters["brightness"]
        maxDifference = parameters["maxDif"]
        try:
            ret, image = video_capture.read()
            if not ret:
                break

            video_capture.set(10, brightness) # brightness
            video_capture.set(cv2.CAP_PROP_AUTOFOCUS, 1) #auto-focus 
            # both opencv and numpy are "row-major", so y goes first
            cropImg = image[y:y+h, x:x+w]
            cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
            gray_img = cv2.cvtColor(cropImg, cv2.COLOR_BGR2GRAY)
            blur_img = cv2.GaussianBlur(gray_img, (15, 15), 0)
            _,thres_img = cv2.threshold(blur_img, binarization, 255, cv2.THRESH_BINARY)
            if primeiroFrame is None:
                for i in range(10):
                    primeiroFrame = thres_img
                continue
            
            if trigger:
                primeiroFrame = thres_img
                trigger = False
            difImg = cv2.absdiff(primeiroFrame,thres_img)
            '''degub
            meanFirstImgValue = np.mean(primeiroFrame.flatten())
            meanBlurValue = np.mean(blur_img.flatten())
            '''
            meanDifImgValue = np.mean(difImg.flatten())
            presenceDifPercentage = (meanDifImgValue*100)/255
            if presenceDifPercentage >= maxDifference:
                approved = False
            elif presenceDifPercentage < maxDifference:
                approved = True
            

            (flag,encodedImg) = cv2.imencode(".jpg", image)
            (_,processedImgEncoded) = cv2.imencode(".jpg",  difImg)
            outQ.put(processedImgEncoded)
            if not flag:
                continue
            yield (b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + bytearray(encodedImg) + b'\r\n')
        except cv2.error as e:
            print("CV2 Exception Caught!!!")
            break
    if mode == "presence":
        print("CHANGING ROI")
        (_,image) = video_capture.read()
        (_,encodedImg) = cv2.imencode(".jpg", image)
        yield (b'--frame\r\n'
            b'Content-Type: image/jpeg\r\n\r\n' + bytearray(encodedImg) + b'\r\n')
        video_capture.release()
        roiState = False
    else:
        video_capture.release()
        print(f"MODE CHANGED!NOW {mode.upper()}")
        

def rgbValueCheck(outQ,parameters,device=0):
    print("RGB CHECK ON")
    global roiState,x,y,w,h,mode,trigger

    video_capture = cv2.VideoCapture(device)
    video_capture.set(cv2.CAP_PROP_FPS, 60)

    primeiroFrame = None

    while not roiState and mode=="rgb":
        binarization = parameters["binarization"]
        brightness = parameters["brightness"]
        try:
            ret, image = video_capture.read()
            if not ret:
                break

            video_capture.set(10, brightness)  # brightness
            # both opencv and numpy are "row-major", so y goes first
            cropImg = image[y:y+h, x:x+w]
            cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)

            if primeiroFrame is None:
                for i in range(10):
                    primeiroFrame = cropImg
                continue
            
            meanRGBImg = np.zeros(cropImg.shape,dtype='uint8')          
            #B:0 G:1 R:2          
            meanRGBImg[:,:,0] = np.mean(cropImg[:,:,0])
            meanRGBImg[:,:,1] = np.mean(cropImg[:,:,1])
            meanRGBImg[:,:,2] = np.mean(cropImg[:,:,2])
            blueValue = np.mean(meanRGBImg[:,:,0].flatten())
            #blueValue = np.mean(cropImg[:,:,0].flatten())
            greenValue = np.mean(meanRGBImg[:,:,1].flatten())
            #greenValue = np.mean(cropImg[:,:,1].flatten())
            redValue = np.mean(meanRGBImg[:,:,2].flatten())
            #redValue = np.mean(cropImg[:,:,2].flatten())
            #print(f"BGR:{blueValue:.0f},{greenValue:.0f},{redValue:.0f}")
            (flag,encodedImg) = cv2.imencode(".jpg", image)
            (_,processedImgEncoded) = cv2.imencode(".jpg",  meanRGBImg)
            outQ.put(processedImgEncoded)
            if not flag:
                continue
            yield (b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + bytearray(encodedImg) + b'\r\n')
        except cv2.error as e:
            print("CV2 Exception Caught!!!")
            break
    if mode == "rgb":
        print("CHANGING ROI")
        (_,image) = video_capture.read()
        (_,encodedImg) = cv2.imencode(".jpg", image)
        yield (b'--frame\r\n'
            b'Content-Type: image/jpeg\r\n\r\n' + bytearray(encodedImg) + b'\r\n')
        video_capture.release()
        roiState = False
    else:
        video_capture.release()
        print(f"MODE CHANGED!NOW {mode.upper()}")
        
