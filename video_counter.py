# -*- coding: utf-8 -*-
import cv2
import numpy as np
import cvui
import json
import os.path
import time

if not os.path.exists("C:/Users/Public/Documents/parametros.txt"):
    par = {}
    par["lim"] = 0
    par["brilho"] = 0
    par["minArea"] = 0
    par["maxArea"] = 0
    par["ero"] = 0
    par["dil"] = 0
    par["lineLarg"] = 1
    file = open("C:/Users/Public/Documents/parametros.txt","w",encoding="utf-8")
    json.dump(par,file)
    file.close()

json_file =  open("C:/Users/Public/Documents/parametros.txt","r",encoding="utf-8")
parameters = json.load(json_file)
json_file.close()

top = False
bottom = False

status=[]

entradas = 0
lim = [parameters["lim"]]
brilho = [parameters["brilho"]]
minArea = [parameters["minArea"]]
maxArea = [parameters["maxArea"]]
ero = [parameters["ero"]]
dil = [parameters["dil"]]
lineLarg = [parameters["lineLarg"]]

print("Contador de peças por visão de máquina - SetSystem - 2019")
print("Pressione a tecla 'Q' para sair")
device=input("Insira o ID da câmera deseja utilizar ou o caminho para o arquivo de vídeo: ")
if device == '':
    device = 0
elif device.isdigit() :
    device = int(device)

video_capture = cv2.VideoCapture(device)
video_capture.set(cv2.CAP_PROP_FPS, 60)

primeiroFrame = None

for i in range(0,20):
    ret,image = video_capture.read()

(x,y,w,h) = cv2.selectROI('Frame', image)
oldCenter = []

WINDOW_NAME	= 'Parametros'
frame = np.zeros((520, 300, 3), np.uint8)
cvui.init(WINDOW_NAME)

while True:
    frame[:] = (49, 52, 49)
    cvui.text(frame, 80, 450, 'Contagem:')
    if cvui.button(frame, 155, 442, 'Zerar'):
        entradas = 0
        status = []
    if cvui.button(frame,78,480,'Salvar Parametros'):
        par = {}
        par["lim"] = int(lim[0])
        par["brilho"] = int(brilho[0])
        par["minArea"] = int(minArea[0])
        par["maxArea"] = int(maxArea[0])
        par["ero"] = int(ero[0])
        par["dil"] = int(dil[0])
        par["lineLarg"] = int(lineLarg[0])
        file = open("C:/Users/Public/Documents/parametros.txt","w",encoding="utf-8")
        json.dump(par,file)
        file.close()
    cvui.text(frame, 120, 20, 'Binarizacao')
    cvui.trackbar(frame, 50, 30, 200, lim, 0, 255,1, '%.0Lf')#frame,x,y,largura,var,min,max,divisões,decimal
    cvui.text(frame, 130, 80, 'Brilho')
    cvui.trackbar(frame, 50, 90, 200, brilho, 0, 255,1, '%.0Lf')#frame,x,y,largura,var,min,max,divisões,decimal
    cvui.text(frame, 130, 130, 'A.Min.')
    cvui.trackbar(frame, 50, 140, 200, minArea, 0, 3000,1, '%.0Lf')#frame,x,y,largura,var,min,max,divisões,decimal
    cvui.text(frame, 130, 190, 'A.Max.')
    cvui.trackbar(frame, 50, 200, 200, maxArea, 0, 100000,1, '%.0Lf')#frame,x,y,largura,var,min,max,divisões,decimal
    cvui.text(frame, 130, 250, 'Erosao')
    cvui.trackbar(frame, 50, 260, 200, ero, 0, 20,1, '%.0Lf',cvui.TRACKBAR_DISCRETE, 1)#frame,x,y,largura,var,min,max,divisões,decimal
    #cvui.text(frame, 130, 310, 'Dilatacao')
    #cvui.trackbar(frame, 50, 320, 200, dil, 0, 20,1, '%.0Lf',cvui.TRACKBAR_DISCRETE, 1)#frame,x,y,largura,var,min,max,divisões,decimal
    cvui.text(frame,50,370,'Larg. Linha')
    cvui.trackbar(frame,50,380,200,lineLarg,1,20,1,'%.0Lf',cvui.TRACKBAR_DISCRETE, 1)
    cvui.imshow(WINDOW_NAME, frame)

    ret,image = video_capture.read()
    if not ret:
        break

    video_capture.set(10, int(brilho[0]))#brightness
    cropImg = image[y:y+h , x:x+w] # both opencv and numpy are "row-major", so y goes first
    cv2.rectangle(image,(x,y),(x+w,y+h),(0,255,0),2)
    CoordenadaYLinhaEntrada = int((h / 2))

    gray_img = cv2.cvtColor(cropImg,cv2.COLOR_BGR2GRAY)
    gray_img = cv2.GaussianBlur(gray_img,(15,15),0)

    if primeiroFrame is None:
        primeiroFrame = gray_img
        continue

    frame_delta = cv2.absdiff(primeiroFrame,gray_img)
    ret,thresh = cv2.threshold(frame_delta,int(lim[0]),255,cv2.THRESH_BINARY)
    mask = thresh.copy()
    alt,larg = thresh.shape[:2]
    floodMask = np.zeros((alt+2,larg+2),np.uint8)
    maskedImg = cv2.bitwise_and(thresh, thresh, mask=mask)
    floodImg = maskedImg.copy()
    cv2.floodFill(floodImg,floodMask,(0,0),255)
    floodImg_Inv = cv2.bitwise_not(floodImg)
    floodedImg = maskedImg | floodImg_Inv
    mask2 = floodedImg.copy()
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS,(5,5))
    mask2 = cv2.erode(mask2, kernel, iterations=int(ero[0]))
    floodedImg = cv2.bitwise_and(floodedImg, floodedImg, mask=mask2)

    cv2.line(cropImg, (0,CoordenadaYLinhaEntrada), (w,CoordenadaYLinhaEntrada), (0, 0, 255), int(lineLarg[0]))

    cv2.destroyWindow("Frame")
    cv2.imshow("Masked",maskedImg)
    cv2.imshow("Flooded",floodedImg)

    contours, hierarchy = cv2.findContours(floodedImg,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_L1)
    newCenter = []
    for c in contours:
        if cv2.contourArea(c) < int(minArea[0]) or cv2.contourArea(c) > int(maxArea[0]):
            continue
        M = cv2.moments(c)
        (xc,yc,wc,hc) = cv2.boundingRect(c)
        cv2.rectangle(cropImg, (xc, yc), (xc + wc, yc + hc), (0, 255, 0), 2)
        centroidX = 0
        centroidY = 0
        centroid = (centroidX,centroidY)
        if M["m00"] != 0:
            centroidX = int(M["m10"] / M["m00"])
            centroidY = int(M["m01"] / M["m00"])
            centroid = (centroidX,centroidY)
        cv2.circle(cropImg, centroid, 1, (255, 0, 0), 3)
        newCenter = newCenter + [centroidY]
    zipList = zip(newCenter,oldCenter)
    offSet = int(lineLarg[0]/2)
    for a,b in zipList:
        top = False
        bottom = False
        if b <= CoordenadaYLinhaEntrada:
            top = True
        if a >= CoordenadaYLinhaEntrada + offSet:
            bottom = True
        if bottom and top:
            status = status + ['x']
        print("Top: {}".format(str(top)))
        print("Bottom: {}".format(str(bottom)))
    print(status)
    entradas = status.count('x')
    oldCenter = newCenter
    cv2.putText(image, "Entradas: {}".format(str(entradas)), (10, 25),cv2.FONT_HERSHEY_SIMPLEX, 1, (250, 0, 1), 2)
    cv2.imshow("Contador de Pecas",image)
    key = cv2.waitKey(20) & 0xFF
    if  key == ord('q') or key == ord('Q'):
        break
    if key == ord('c'):
        entradas = 0
        status = []
    if key == ord('o'):
        lim[0]+=1
    if key == ord('i'):
        lim[0]-=1
    if key == ord('l'):
        brilho[0]+=1
    if key == ord('k'):
        brilho[0]-=1
    if key == ord('m'):
        minArea[0]+=1
    if key == ord('n'):
        minArea[0]-=1
    if key == ord('j'):
        maxArea[0]+=1
    if key == ord('h'):
        maxArea[0]-=1

video_capture.release()
cv2.destroyAllWindows()
