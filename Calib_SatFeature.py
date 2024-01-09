import os
import cv2 
import numpy as np
import argparse

R = 114
ESC_KEY = 27
SPACE = 32

def checkReduction(image):
    reduction = 1
    if image.shape[0]>2000 or image.shape[1]>2000:
        image = cv2.resize(image, (0,0), fx=0.5, fy=0.5)
        reduction = 2
    if image.shape[0]>1500 or image.shape[1]>1500:
        reduction = 2
    return reduction


''''''''''''''' Background Extractor '''''''''''''''
def calcBackgound(VideoPath, reduce, Save=None):
    cap = cv2.VideoCapture(VideoPath)
    _, f = cap.read()
    f= cv2.resize(f, (f.shape[1]// reduce , f.shape[0] // reduce))
    img_bkgd = np.float32(f)
    # reduce = checkReduction(img_bkgd)
    print('<< When you feel the background is sufficiently clear, press SPACE to end and save the background.')
    while True:
        ret, f = cap.read()
        if not ret: break
        f= cv2.resize(f, (f.shape[1]// reduce , f.shape[0] // reduce))
        cv2.imshow('Main Video', f)
        cv2.accumulateWeighted(f, img_bkgd, 0.01)
        res2 = cv2.convertScaleAbs(img_bkgd)
        cv2.imshow('<< When you feel the background is sufficiently clear, press SPACE to end and save the background.', res2)
        k = cv2.waitKey(20)
        if k == SPACE: break
    if not Save is None: cv2.imwrite(Save, res2)
    cv2.destroyAllWindows()
    cap.release()
    return res2


''''''''''''''' Region of Interest '''''''''''''''
def _getROI(image, Save=None):
    while True:
        roi, coords , roiImage = getROI('<< Select a Region of Interst for calibration | Actions: SPACE = Complete,  R = Retry |', image).run()
        zeroDim = False
        for i in roi.shape:
            if i ==0: zeroDim = True
        if zeroDim: continue
        cv2.imshow(':: Region of Interest | Actions: SPACE = Complete,  R = Retry |', roi)
        k = cv2.waitKey(0)
        if k%256 == R: cv2.destroyAllWindows(); continue
        elif k%256 == SPACE: cv2.destroyAllWindows(); break
    if not Save is None: cv2.imwrite(Save, roiImage)
    return roi, coords

def applyROI(coord, roiCoord, reverse=False):
    x1, y1, x2, y2 = coord
    [sX, sY], [eX, eY] = roiCoord
    if reverse:
        return x1 + sX, y1 + sY, x2 + sX, y2 + sY
    else:
        return x1 - sX, y1 - sY, x2 - sX, y2 - sY

def applyROIxy(coord, roiCoord, reverse=False):
    x, y = coord
    [sX, sY], [eX, eY] = roiCoord
    if reverse:
        return x + sX, y + sY
    else:
        return x - sX, y - sY


def putROI(image, roiCoord):
    [sX, sY], [eX, eY] = roiCoord
    if len(image.shape) > 2:
        roi = image[sY:eY, sX:eX,:]
    else:
        roi = image[sY:eY, sX:eX]
    return roi

def ShowROI(image, roiCoord):
    mask = getMask(image, roiCoord)
    [sX, sY], [eX, eY] = roiCoord
    Show = cv2.addWeighted(image, 0.5, np.where(mask > 1, image, 0), 1 - 0.5, 0)
    cv2.rectangle(Show,(sX, sY),(eX, eY), (255,255,255),2)
    cv2.putText(Show, 'Region Of Interest', (sX, sY-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
    return Show


''''''''''''''' Bird's Eye View '''''''''''''''
class birds_eye:
    def __init__(self, image, cordinates, size=None):
        self.original = image.copy()
        self.image =  image
        self.c, self.r = image.shape[0:2]
        if size:self.bc, self.br = size
        else:self.bc, self.br = self.c, self.r
        pst2 = np.float32(cordinates)
        pst1 = np.float32([[0,0], [self.r,0], [0,self.c], [self.r,self.c]])
        self.transferI2B = cv2.getPerspectiveTransform(pst1, pst2)
        self.transferB2I = cv2.getPerspectiveTransform(pst2, pst1)
        self.bird = self.img2bird()
    def img2bird(self):
        self.bird = cv2.warpPerspective(self.image, self.transferI2B, (self.br, self.bc))
        return self.bird
    def bird2img(self):
        self.image = cv2.warpPerspective(self.bird, self.transferB2I, (self.r, self.c))
        return self.image
    def setImage(self, img):
        self.image = img
    def setBird(self, bird):
        self.bird = bird
    def convrt2Bird(self, img):
        return cv2.warpPerspective(img, self.transferI2B, (self.bird.shape[1], self.bird.shape[0]))
    def convrt2Image(self, bird):
        return cv2.warpPerspective(bird, self.transferB2I, (self.image.shape[1], self.image.shape[0]))
    def projection_on_bird(self, p, float_type=False):
        M = self.transferI2B
        px = (M[0][0]*p[0] + M[0][1]*p[1] + M[0][2]) / ((M[2][0]*p[0] + M[2][1]*p[1] + M[2][2]))
        py = (M[1][0]*p[0] + M[1][1]*p[1] + M[1][2]) / ((M[2][0]*p[0] + M[2][1]*p[1] + M[2][2]))
        if float_type: return px, py
        return int(px), int(py)
    def projection_on_image(self, p, float_type=False):
        M = self.transferB2I
        px = (M[0][0]*p[0] + M[0][1]*p[1] + M[0][2]) / ((M[2][0]*p[0] + M[2][1]*p[1] + M[2][2]))
        py = (M[1][0]*p[0] + M[1][1]*p[1] + M[1][2]) / ((M[2][0]*p[0] + M[2][1]*p[1] + M[2][2]))
        if float_type: return px, py
        return int(px), int(py)
def project(M, p):
    px = (M[0][0]*p[0] + M[0][1]*p[1] + M[0][2]) / ((M[2][0]*p[0] + M[2][1]*p[1] + M[2][2]))
    py = (M[1][0]*p[0] + M[1][1]*p[1] + M[1][2]) / ((M[2][0]*p[0] + M[2][1]*p[1] + M[2][2]))
    return int(px), int(py)


colorPool = {1: (165,42,42), 2:(137,43,226), 3:(210, 105, 30), 4:(175,175,0), 5:(22,175,100), 6:(100,10,0), 7:(100,200,0), 8:(100,10,200)}
navigattorSize = 80

def Semi(Camera, Satellite, Save=None):
    Hsapace = 20
    Vspace = 10
    reduceC = checkReduction(Camera)
    reduceS = checkReduction(Satellite)

    while True:
        fp1 = getFeaturePoint(Satellite, '<< Choose a minimum of 4 random points for satellite matching | Actions: SPACE = Complete, R = Retry |' , colorPool=colorPool, navigattorSize=navigattorSize, reduction=2)
        seeds1, img1, seedsimg1 = fp1.run()
        k = cv2.waitKey(0)
        if k%256 == R: cv2.destroyAllWindows(); continue

        fp2 = getFeaturePoint(Camera, '<< Choose a minimum of 4 points corresponding to the chosen satellite points | Actions: SPACE = Complete, R = = Retry |', colorPool=colorPool, navigattorSize=navigattorSize, reduction=checkReduction(Camera))
        seeds2, img2, seedsimg2 = fp2.run()
        k = cv2.waitKey(0)
        if k%256 == R: cv2.destroyAllWindows(); continue
        
        if len(seeds1) != len(seeds2): 
            print('::: The number of Seeds must be Equal!')
            cv2.destroyAllWindows(); continue
        if len(seeds1) < 4: 
            print('::: Please select at least 4 points')
            cv2.destroyAllWindows(); continue


        matchPart = Hsapace*3 + navigattorSize*2
        w = img1.shape[1] + img2.shape[1] + matchPart
        h = max(img1.shape[0] , img2.shape[0], navigattorSize*len(seeds1) + 10 * len(seeds1))
        alls = np.ones((h,w,3), dtype=np.uint8) * 255
        alls[0:img1.shape[0], 0:img1.shape[1], :] = img1
        alls[0:img2.shape[0], img1.shape[1] + matchPart:w, :] = img2
        for i in range(len(seeds1)):
            try: 
                pt1 = seeds1[i][0] // reduceS, seeds1[i][1] // reduceS
                pt2 = img1.shape[1] + Hsapace , navigattorSize//2 + Vspace
                color = colorPool[i+1]
                cv2.line(alls, pt1, pt2, color, 2)

                topLeft = img1.shape[1] + Hsapace,  0 + Vspace
                downRight = img1.shape[1] + Hsapace + navigattorSize, navigattorSize + Vspace
                alls[topLeft[1] : downRight[1],  topLeft[0] : downRight[0], :] = seedsimg1[i]
                cv2.rectangle(alls ,topLeft,downRight, color, 2)

                topLeft = img1.shape[1] + Hsapace*2 + navigattorSize,  0 + Vspace
                downRight = img1.shape[1] + Hsapace*2 + navigattorSize + navigattorSize , navigattorSize + Vspace
                alls[topLeft[1] : downRight[1],  topLeft[0] : downRight[0], :] = seedsimg2[i]
                cv2.rectangle(alls ,topLeft,downRight, color, 2)
                
                pt1 = img1.shape[1] + matchPart + seeds2[i][0] // reduceC, seeds2[i][1] // reduceC
                pt2 = img1.shape[1] + Hsapace*2 + navigattorSize*2, navigattorSize //2 + Vspace
                cv2.line(alls, pt1, pt2, color, 2)
            except Exception as e:
                print(">> Warning: The preview image of matched points is not properly prepared due to the selected seed points being too close to the image border. \n' +\
                    >> Info: Nevertheless, there is no need to worry as the calibration is progressing well.")
            Vspace += navigattorSize + 10 
        
        reduce = checkReduction(alls)
        alls_resized= cv2.resize(alls, (alls.shape[1]// reduce , alls.shape[0] // reduce))
        cv2.imshow('Summary of your matched points | Actions: SPACE = Complete, R = = Retry |' , alls_resized)
        k = cv2.waitKey(0)
        if k%256 == R: cv2.destroyAllWindows(); continue

        onBird, _ = cv2.findHomography(np.float32(seeds2), np.float32(seeds1), cv2.RANSAC,5.0)
        mapped = cv2.warpPerspective(Camera, onBird, (Satellite.shape[1],Satellite.shape[0]))

        Transparency = 0.5; mapped_results = cv2.addWeighted(Satellite, Transparency, mapped, 1 - Transparency, 0)
        
        reduce = checkReduction(mapped_results)
        mapped_results_resized= cv2.resize(mapped_results, (alls.shape[1]// reduce , alls.shape[0] // reduce))
        cv2.imshow('Camera image Mapped on Satellite image', mapped_results_resized)

        s = Camera.shape
        tl = project(onBird, (0,0))
        tr = project(onBird, (s[1],0))
        dl = project(onBird, (0,s[0]))
        dr = project(onBird, (s[1],s[0]))
        BEVcoord = [[tl[0], tl[1]], [tr[0], tr[1]], [dl[0], dl[1]], [dr[0], dr[1]]]
        e = birds_eye(Camera, BEVcoord, size=(Satellite.shape[1], Satellite.shape[0]))
        cv2.imshow('Bird\'s Eye View', e.bird)

        if not Save is None: 
            cv2.imwrite(Save, mapped_results)
            cv2.imwrite(Save+'.match.bmp', alls)
        k = cv2.waitKey(0)
        if k%256 == SPACE: break
        elif k%256 == R: cv2.destroyAllWindows()
    cv2.destroyAllWindows()
    return BEVcoord, e.bird.shape[:2]

def getMask(image, coords, Save=None):
    [sX, sY], [eX, eY] = coords
    mask = np.zeros_like(image)
    mask[sY:eY, sX:eX, :] = 255
    if not Save is None: cv2.imwrite(Save, mask)
    return mask



class getPixelUnitPoints:
    def __init__(self, windowName, image, e,hAxis=30, number=3):
        self.img = image.copy()
        self.num = number
        self.pnt = []
        self.wn =windowName
        self.Done = False
        self.e = e
        self.ha =hAxis
        self.sit = False
    def run(self):
        cv2.imshow(self.wn, self.img)
        cv2.setMouseCallback(self.wn, self.EventManager)
        k = cv2.waitKey(0)
        if k%256 == SPACE:
            if len(self.pnt) == 3:
                cv2.destroyAllWindows()
                return self.pnt, self.img
            else:
                print(': Uncomplete action!')
                return None, None
        else: return None, None
    def EventManager(self, event, x, y, flags, param):  
        imShow = self.img.copy()
        if not self.sit:
            bx, by = self.e.projection_on_bird((x, y))
            vx, vy = self.e.projection_on_image((bx, by - self.ha))
            cv2.line(imShow, (vx, vy), (x,y), (255,100,100), 3)
            hx, hy = self.e.projection_on_image((bx- self.ha, by))
            cv2.line(imShow, (hx, hy), (x,y), (100,100,255), 3)
        else:
            c = 0
            if not self.Done:
                if len(self.pnt) == 1:  cv2.line(imShow, self.pnt[0], (x,y), (255,0,0), 2)
                if len(self.pnt) == 2:  cv2.line(imShow, self.pnt[0], (x,y), (0,0,255), 2)
                c = np.sqrt((self.pnt[0][0] - x)**2 + (self.pnt[0][1] - y)**2)
                cv2.putText(imShow, str(round(c))+' pixel', (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
        cv2.imshow(self.wn, imShow)
        if event == cv2.EVENT_LBUTTONDOWN:
            if not self.sit: 
                self.sit=True
                cv2.circle(imShow, (x, y), 5, (255, 255, 255), -1)
                self.img = imShow.copy()
                self.pnt.append((x, y))
            else:
                if len(self.pnt) < self.num:
                    self.pnt.append((x, y))
                    cv2.line(self.img, self.pnt[0], self.pnt[-1], (255,0,0), 2)
                    cv2.circle(self.img, (x, y), 5, (255, 255, 255), -1)
                    cv2.putText(self.img, str(round(c))+' pixel', (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
                if len(self.pnt) == self.num and not self.Done:
                    cv2.line(self.img, self.pnt[0], self.pnt[-1], (0,0,255), 2)
                    cv2.circle(self.img, (x, y), 5, (255, 255, 255), -1)
                    cv2.putText(self.img, str(round(c))+' pixel', self.pnt[-1], cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
                    self.Done = True

def getPixelUnit(frame, realDistance, Coordinate, BEV_size, Save=None):
    e = birds_eye(frame, Coordinate, size=BEV_size)
    while True:
        p ,img= getPixelUnitPoints(f'<< Select two points on the horizontal and vertical axes that are {realDistance} meters apart | Actions: SPACE = Complete, R = Retry |', frame, e).run()
        if not p: continue
        else: break
    if Save: cv2.imwrite(Save, img)

    p1 = e.projection_on_bird(p[0])
    p2 = e.projection_on_bird(p[1])
    disv = np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

    p1 = e.projection_on_bird(p[0])
    p2 = e.projection_on_bird(p[2])
    dish = np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
    
    return (realDistance*100 / disv), (realDistance*100 / dish)


R = 114
ESC_KEY = 27
SPACE = 32

class getROI:
    def __init__(self, windowName, image):
        self.img = image.copy()
        self.w = image.shape[1]
        self.h = image.shape[0]
        self.wn =windowName
        self.coord = [[0,0],[self.w, self.h]]
        self.ix, self.iy = 0,0
        self.drawing = False
        self.roi = self.img
        self.mask = np.ones_like(self.img)
        self.Show = self.img
    def run(self):
        cv2.imshow(self.wn, self.img)
        cv2.setMouseCallback(self.wn, self.EventManager)
        cv2.waitKey(0)
        return self.roi, self.coord, self.Show
    def EventManager(self, event, x, y, flags, param):  
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.ix, self.iy = x,y
        elif event == cv2.EVENT_MOUSEMOVE:
            self.imShow = self.img.copy()
            if self.drawing == True:
                sX = min(self.ix, x)
                sY = min(self.iy, y)
                eX = max(self.ix, x)
                eY = max(self.iy, y)
                cv2.rectangle(self.imShow,(sX, sY),(eX, eY), (255,255,255),2)
                self.coord = [sX, sY], [eX, eY]
                cv2.imshow(self.wn, self.imShow)
        elif event == cv2.EVENT_LBUTTONUP:
            self.drawing = False
            [sX, sY], [eX, eY] = self.coord
            self.roi = self.img[sY:eY, sX:eX,:]
            self.mask = np.zeros_like(self.imShow)
            self.mask[sY:eY, sX:eX, :] = 1
            self.Show = cv2.addWeighted(self.imShow, 0.5, np.where(self.mask == 1, self.img, 0), 1 - 0.5, 0)
            cv2.putText(self.Show, f'ROI = {self.roi.shape}', (sX, sY-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
            cv2.imshow(self.wn, self.Show)


class getPoint:
    def __init__(self, windowName, image, number):
        self.img = image.copy()
        self.num = number
        self.pnt = []
        self.wn =windowName
        self.oneceFlag = False
        cv2.imshow(windowName, self.img)
        cv2.setMouseCallback(windowName, self.EventManager)
        
    def run(self):
        return self.pnt, self.img
    def EventManager(self, event, x, y, flags, param):  
        imShow = self.img.copy()
        c = 0
        if len(self.pnt) == 1:
            cv2.line(imShow, self.pnt[-1], (x,y), (255,255,255), 2)
            c = np.sqrt((self.pnt[-1][0] - x)**2 + (self.pnt[-1][1] - y)**2)
        cv2.putText(imShow, str(round(c))+' pixel', (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
        cv2.imshow(self.wn, imShow)
        if event == cv2.EVENT_LBUTTONDOWN:
            if len(self.pnt) < self.num:
                self.pnt.append((x, y))
                cv2.circle(self.img, (x, y), 5, (0, 0, 255), 5)
            if len(self.pnt) == self.num and not self.oneceFlag:
                cv2.line(self.img, self.pnt[-2], self.pnt[-1], (0,0,255), 2)
                cv2.putText(self.img, str(round(c))+' pixel', self.pnt[-1], cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
                self.oneceFlag = True


class getFeaturePoint:
    def __init__(self, image, windowName , colorPool,  reduction=2, navigattorSize = 80, DrawLine=False, showNaro=True, totoalSeed=8):
        self.image = image.copy()
        self.imgSize = image.shape
        self.reduction = reduction
        self.showSize = (self.imgSize[1] // reduction , self.imgSize[0] // reduction)
        self.showImg = cv2.resize(image, self.showSize)
        self.totoalSeed = totoalSeed
        cv2.imshow(windowName, self.showImg)
        self.navigBoxSize = (navigattorSize,  navigattorSize)
        self.windowName = windowName
        cv2.setMouseCallback(windowName, self.EventManager)
        self.seed = []
        self.imgSeed = []
        self.colorPool = colorPool
        self.DrawLine = DrawLine
        self.showNaro = False if reduction == 1 else showNaro
    def run(self):
        return self.seed, cv2.resize(self.image , self.showSize), self.imgSeed
    def EventManager(self, event, x, y, flags, param):
        showImg = cv2.resize(self.image , self.showSize)
        if x < self.navigBoxSize[0]:
                naviQuart = 4 if y < self.navigBoxSize[1] else 1
        else:
            if x < self.showSize[0] - self.navigBoxSize[0]:
                naviQuart = 4 if y < self.navigBoxSize[1] else 1
            else:
                naviQuart = 2 if y > self.showSize[1] - self.navigBoxSize[1] else 3    
        if naviQuart == 1: cofX = +1 ; cofY = -1
        if naviQuart == 2: cofX = -1 ; cofY = -1
        if naviQuart == 3: cofX = -1 ; cofY = +1
        if naviQuart == 4: cofX = +1 ; cofY = +1
        cx = x + cofX*self.navigBoxSize[0]//2
        cy = y + cofY*self.navigBoxSize[1]//2
        center = cx, cy
        topLeft =  min(cx + cofX*self.navigBoxSize[0]//2,cx - cofX*self.navigBoxSize[0]//2) , min(cy + cofY*self.navigBoxSize[1]//2, cy - cofY*self.navigBoxSize[1]//2)
        downRight= max(cx - cofX*self.navigBoxSize[0]//2,cx + cofX*self.navigBoxSize[0]//2) , max(cy + cofY*self.navigBoxSize[1]//2, cy - cofY*self.navigBoxSize[1]//2)
        orgCenter = x * self.reduction , y * self.reduction
        self.tmp = showImg.copy()
        try:
            roi = self.image[orgCenter[1]- self.navigBoxSize[1]//2: orgCenter[1]+ self.navigBoxSize[1]//2, 
                    orgCenter[0] - self.navigBoxSize[0]//2: orgCenter[0]+ self.navigBoxSize[0]//2]
            if self.showNaro:
                self.tmp[topLeft[1]: downRight[1], topLeft[0]: downRight[0]] = roi
                cv2.rectangle(self.tmp,topLeft,downRight, (0, 255, 0), 2)
                pointColor =(255,255,255) 
                cv2.circle(self.tmp, center, 1, pointColor, -1)
        except Exception as e:
            print(">> Error:", e)

        if event == cv2.EVENT_LBUTTONDOWN:
            if self.DrawLine:
                if len(self.seed) < 4:
                    self.seed.append(orgCenter)
                    self.imgSeed.append(roi)
                    if len(self.seed) >= 2:
                        cv2.line(self.image, self.seed[-2], self.seed[-1], self.colorPool[2], 2)
                        if len(self.seed)==4:
                            cv2.line(self.image, self.seed[0], self.seed[-1], self.colorPool[2], 2)
                            a = self.seed[-1]
                            self.seed[-1] = self.seed[-2]
                            self.seed[-2] = a
                    cv2.circle(self.image, orgCenter, 10, self.colorPool[2], -1)
            else:
                if len(self.seed) < self.totoalSeed:
                    self.seed.append(orgCenter)
                    self.imgSeed.append(roi)
                    cv2.circle(self.image, orgCenter, 14, self.colorPool[len(self.seed)], -1)
                    cv2.putText(self.image, str(len(self.seed)), (orgCenter[0]-8, orgCenter[1]+8), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)

        cv2.imshow(self.windowName, self.tmp)






def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--src', type=str, help='Video file', default='Leeds.mp4')
    parser.add_argument('--sat', type=str, help='Satellite image', default='Satellite.png')
    opt = parser.parse_args()
    return opt

def main(opt):
    
    Input = r"C:\Users\tsmaz\OneDrive - University of Leeds\Research\Repositories\BEV-Calibrator\1 06-01-2024 - ver 1.0 [GitHub]\Leeds.mp4" # opt.src
    Sat = r"C:\Users\tsmaz\OneDrive - University of Leeds\Research\Repositories\BEV-Calibrator\1 06-01-2024 - ver 1.0 [GitHub]\Satellite.png" #opt.sat
    
    if not os.path.isfile(Input):
        print(f'>> File Error: The video file {Input} is not found!')
        return
    if not os.path.isfile(Sat):
        print(f'>> File Error: The satellite file {Sat} is not found!')
        return


    root = os.path.dirname(os.path.abspath(Input))
    filename = os.path.basename(Input).split('.')[0]

    output_path = f'{root}\\{filename}'
    if os.path.exists(output_path):
        print(f'>> Warning: The directory {output_path} already exists! The new configuration will replace it!' )
    else:
        os.mkdir(output_path)

    Satellite = cv2.imread(Sat)
    Background = calcBackgound(Input, reduce=2, Save=f'{output_path}\\Background.bmp')
    ROI, ROI_Coords = _getROI(Background, Save=f'{output_path}\\Region of Interest.bmp')
    BEV_Coordinate, BEV_size = Semi(ROI, Satellite, Save=f'{output_path}\\Bird Eye View.bmp')
    Pixel_Unie = getPixelUnit(ROI, realDistance=1, Coordinate=BEV_Coordinate, BEV_size=BEV_size, Save=f'{output_path}\\Pixel Unit.bmp')

    output_string =   'Calibration: Region of Interest: '+ str(ROI_Coords)+'\n' \
                    + 'Calibration: Coordinate: '+ str(BEV_Coordinate)+'\n' \
                    + 'Calibration: BEV size: '+ str(BEV_size)+'\n'\
                    + 'Calibration: Pixel_Unie: '+ str(Pixel_Unie)+'\n'\

    print(output_string)

    f = open(f"{output_path}\\config.txt", 'w')
    f.write(output_string)
    f.close()

    print(':: Done.')

if __name__ == "__main__":
    try:
        opt = parse_opt()
        main(opt)
    except Exception as e:
        print(":: Ops:", e)
