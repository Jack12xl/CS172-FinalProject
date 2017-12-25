import numpy as np
import cv2
import pygame
import random
import math
from pygame.locals import *
from sklearn.externals import joblib
from sklearn.preprocessing import Imputer
import random

import functions

cap = cv2.VideoCapture(1)
clf = joblib.load("./mode/svm.m")
kmeans=joblib.load("./mode/kmeans.pkl")

def timeDisplay(countDown):
    mins=int(countDown/60)
    sec=countDown-60*mins
    if sec < 10:
        return str(mins)+'    0'+str(sec)
    else:
        return str(mins)+'    '+str(sec)


def blankDisplay(blank,count):
    for i in range(count):
        screen.blit(blank,(668+(400-50*count)/2+50*i,200))
    return

def answerDisplay(ansfont,cnt,Word,WordList):
    for i in range(cnt):
        ans=ansfont.render(WordList[Word[i]],True,(217,35,35))
        screen.blit(ans,(667+(400-50*len(Word))/2+50*i+int(50-ans.get_width())/2,210))

def button(msg,x,y,w,h,ic,ac,gameDisplay,action=None):
    mouse=pygame.mouse.get_pos()
    click=pygame.mouse.get_pressed()
    if x+w > mouse[0] > x and y+h > mouse[1] > y:
        textRect=pygame.draw.rect(gameDisplay,ac,(x,y,w,h))
        if click[0] == 1 and action != None:
            action()
    else:
        textRect=pygame.draw.rect(gameDisplay,ic,(x,y,w,h))

    smallText=pygame.font.SysFont("comicsansms",40)
    show=smallText.render(msg,True,(255,255,255))
    textRect.center=((x+w/2),(y+h/2))
    gameDisplay.blit(show,(x+25,y+10))
    onbutton = (click == (1,0,0)) and (x+w > mouse[0] > x) and (y+h > mouse[1] > y)
    return onbutton

def countDown(gameDisplay,bg,itime):
    time=(pygame.time.get_ticks()-itime)/1000
    font=pygame.font.SysFont("ubuntu",100)
    font.set_bold(True)
    while time < 3:
        gameDisplay.blit(bg,(0,0))
        time=int((pygame.time.get_ticks()-itime)/1000)
        if 3-time == 0:
            break
        show=font.render(str(3-time),True,(0,0,0))
        gameDisplay.blit(show,(500,300))
        pygame.display.flip()
    gameDisplay.blit(bg,(0,0))
    show=font.render('GO!',True,(0,0,0))
    gameDisplay.blit(show,(450,300))
    pygame.display.flip()
    return

def game_intro(gameDisplay):
    intro = True
    while intro:
        pygame.display.flip()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
        continu=button("GO!",480,600,100,50,(25,48,164),(67,83,186),gameDisplay)
        if continu:
            break
    return

def hintPrint(hintfont,Explanation,screen, widthThres, heightThres,ws,color):
    hint=hintfont.render('['+Explanation[1:-1]+']',True,color)
    if hint.get_width() > widthThres:
        cut=int(hint.get_width()/widthThres)+1
        unit=int(len(Explanation[1:])/cut)
        begin=1
        nextBegin=0
        for i in range(cut):
            # print(i,begin)
            nextBegin+=unit
            if i == cut-1:
                toPrint=Explanation[begin:-1]+']'
            else:
                while Explanation[nextBegin-1] != ' ':
                    nextBegin+=1
                toPrint=Explanation[begin:nextBegin]
                if i == 0:
                    toPrint='['+toPrint
            begin=nextBegin
            hint1=hintfont.render(toPrint,True,color)
            screen.blit(hint1,(ws+(400-hint1.get_width())//2,300+i*30))
    else:
        screen.blit(hint,(ws+int((400-hint.get_width())/2), heightThres))
    return

def moreHint(cnt,Word,WordList,screen):
    hintfont=pygame.font.SysFont("comicansms",40)
    mouse=pygame.mouse.get_pos()
    # print(mouse)
    if cnt < len(Word):
        Explanation=' Next letter is ' +WordList[Word[cnt]] + '!\n'
        if 509 < mouse[0]< 700 and  500 < mouse[1]:
            hintPrint(hintfont,Explanation,screen, 300, 580, 180,(255,255,255))
    return

def cvimage_to_pygame(image):
    """Convert cvimage into a pygame image"""
    return pygame.image.frombuffer(image.tostring(), image.shape[:2],"RGB")

def judgeAnswer(inp,check):
    return (inp==check)

def scoreDisplay(score,screen):
    font=pygame.font.SysFont("arial",40)
    show=font.render("SCORE:"+str(int(score)),True,(0,0,0))
    screen.blit(show,(670+(400-show.get_width())/2,130))

def gameEnd(score,endType):
    if endType == 1:
        print("game clear")
        screen.blit(clear,(0,0))
    else:
        print("time up")
        screen.blit(timeup,(0,0))
    font=pygame.font.SysFont("arial",45)
    show=font.render(str(int(score)),True,(255,255,255))
    screen.blit(show,((1067-show.get_width())/2+50,105+(800-show.get_height())/2))
    pygame.display.flip()
    pygame.time.wait(3000)
    exit(0)
    return

#### Generate vocabularies and their corresponding meaning ####
WordList = ['A','B','C','D','H','I','L','V','Y']
Vocabulary = []
Meaning = []
Input = open('Game')
for line in Input :
	Object = line.split(':')
	Word = []
	for i in Object[0]:
		Index = WordList.index(i)
		Word.append(Index)
	Vocabulary.append(Word)
	Meaning.append(Object[1])

# OrderOfCorrectWord = 16

oneSetTime=180
oneSet=5
pygame.init()
instr=pygame.image.load('resources/1067800instr.png')
bg=pygame.image.load('resources/1067800pre.png')
blank=pygame.image.load('resources/blank50.png')
timeup=pygame.image.load('resources/1067800timeup.png')
clear=pygame.image.load('resources/1067800clear.png')
bgblank=pygame.image.load('resources/1067800blank.png')
screen=pygame.display.set_mode((1067,800))
font=pygame.font.SysFont("ubuntu",35)
hintfont=pygame.font.SysFont("calibri",30)
ansfont=pygame.font.SysFont("comicansms",50)
screen.blit(instr,(0,0))
WordIndexChosen = random.sample(range(len(Vocabulary)),oneSet)
pygame.mouse.set_visible(True)


##### GAME PHASE #####
game_intro(screen)
time_past=pygame.time.get_ticks()
countDown(screen,bgblank,time_past)
# pygame.time.wait(500)
screen.blit(bg,(0,0))
pygame.display.flip()
time_past=pygame.time.get_ticks()
score=0
for NumPlay in range(oneSet):
    answer=0
    cnt=0
    OrderOfCorrectWord = WordIndexChosen[NumPlay]
    Word = Vocabulary[OrderOfCorrectWord]
    Explanation=Meaning[OrderOfCorrectWord]
    print(Meaning[OrderOfCorrectWord])
    unitScore=20/len(Word)
    while True:
        ret,img=cap.read()
        img=img[0:300,300:600]
        image=cvimage_to_pygame(img)
        cv2.imshow('test',img)
        screen.blit(bg,(0,0))
        screen.blit(image,(728,455))
        if cnt >= len(Word):
            break
        for event in pygame.event.get():
            if event.type==pygame.QUIT:
                exit(0)
        if oneSetTime-int((pygame.time.get_ticks()-time_past)/1000) <= 0:
            gameEnd(score,0)
        time=timeDisplay(oneSetTime-int((pygame.time.get_ticks()-time_past)/1000))
        font.set_bold(True)
        if (pygame.time.get_ticks()-time_past)/1000 <= oneSetTime-30:
            timeShow=font.render(time,True,(48,44,105))
        else:
            timeShow=font.render(time,True,(217,35,35))
        screen.blit(timeShow,(865,37))
        hintPrint(hintfont,Explanation,screen,350,350,668,(217,35,35))
        blankDisplay(blank,len(Word))
        ###### FOR CV CODE #####
        mul,thresh=functions.skinDetection(img)
        pic=functions.roiExtraction(img,thresh,mul)
        if np.count_nonzero(pic) == 0:
            continue
        descriptor,labels=functions.SIFTtest(pic,kmeans)
        if labels is None:
            continue
        BoWVector=functions.getTestBoWVector(labels)
        InputNumber = clf.predict(BoWVector)
        print(InputNumber)
        isPoseCorrect=judgeAnswer(InputNumber,Word[cnt])
        answer+=isPoseCorrect
        print('answer:',answer)
        if Word[cnt] == 5:
            if answer >= 15:
                cnt+=1
                score+=unitScore
                answer=0
            elif isPoseCorrect:
                answer+=1
            else:
                answer=0
        else:
            if answer >= 6:
                cnt+=1
                score+=unitScore
                answer=0
            elif isPoseCorrect:
                answer+=1
            else:
                answer=0
        if cnt <= len(Word):
            answerDisplay(ansfont,cnt,Word,WordList)
            # print("cnt:":cnt)
            moreHint(cnt,Word,WordList,screen)
        scoreDisplay(score,screen)
        pygame.display.flip()
    pygame.time.wait(1000)
gameEnd(100,1)


cap.release()
cv2.destroyAllWindows()
