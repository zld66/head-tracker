"""
人头追踪工具 v5.0
"""
import cv2
import pyautogui
import numpy as np
from PIL import ImageGrab
import json
import os

pyautogui.FAILSAFE = True
pyautogui.PAUSE = 0.01

COLOR_RANGES = {
    'red': {'name': '红色', 'ranges': [{'lower': np.array([0,100,100]), 'upper': np.array([10,255,255])}, {'lower': np.array([156,100,100]), 'upper': np.array([180,255,255])}], 'display': (0,0,255)},
    'yellow': {'name': '黄色', 'ranges': [{'lower': np.array([20,100,100]), 'upper': np.array([35,255,255])}], 'display': (0,255,255)},
    'purple': {'name': '紫色', 'ranges': [{'lower': np.array([125,100,100]), 'upper': np.array([155,255,255])}], 'display': (255,0,255)}
}

class KeyConfig:
    DEFAULT_KEYS = {'quit': 'q', 'mouse_toggle': 'm', 'color_cycle': 'c', 'mode_cycle': 'd'}
    def __init__(self):
        self.keys = self.DEFAULT_KEYS.copy()
        self.config_file = 'key_config.json'
        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    self.keys.update(json.load(f))
            except: pass
    def save(self):
        with open(self.config_file, 'w', encoding='utf-8') as f:
            json.dump(self.keys, f)
    def get_key(self, a): return self.keys.get(a, 'q')
    def get_code(self, a): return ord(self.get_key(a))

class App:
    def __init__(self):
        self.face = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.hog = cv2.HOGDescriptor()
        self.hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
        self.kc = KeyConfig()
        self.running = False
        self.mouse_on = True
        self.color = None
        self.mode = 'both'
        self.sw, self.sh = pyautogui.size()
        self.lx, self.ly = None, None

    def detect_bodies(self, f):
        h, w = f.shape[:2]
        s = 640/w if w>640 else 1
        r = cv2.resize(f, (int(w*s), int(h*s)))
        try:
            boxes, _ = self.hog.detectMultiScale(r, winStride=(8,8), padding=(8,8), scale=1.05)
            return [(int(x/s), int(y/s), int(ww/s), int(hh/s)) for x,y,ww,hh in boxes]
        except: return []

    def detect_colors(self, f):
        if not self.color: return []
        hsv = cv2.cvtColor(f, cv2.COLOR_BGR2HSV)
        cfg = COLOR_RANGES[self.color]
        m = None
        for r in cfg['ranges']:
            t = cv2.inRange(hsv, r['lower'], r['upper'])
            m = t if m is None else cv2.bitwise_or(m, t)
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((5,5), np.uint8))
        cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return [c for c in cnts if cv2.contourArea(c) > 500]

    def handle_key(self, k):
        if k == self.kc.get_code('quit'): return False
        elif k == self.kc.get_code('mouse_toggle'): self.mouse_on = not self.mouse_on; print(f"鼠标:{'开' if self.mouse_on else '关'}")
        elif k == self.kc.get_code('color_cycle'):
            cs = [None,'red','yellow','purple']
            self.color = cs[(cs.index(self.color) if self.color in cs else 0)+1 if self.color in cs else 0] if self.color else 'red'
            if self.color not in cs: self.color = None
            idx = cs.index(self.color) if self.color in cs else 0
            self.color = cs[(idx+1)%4]
            print(f"颜色:{COLOR_RANGES[self.color]['name'] if self.color else '关'}")
        elif k == self.kc.get_code('mode_cycle'):
            ms = ['both','face','body']
            self.mode = ms[(ms.index(self.mode)+1)%3]
            print(f"模式:{'人脸+人体' if self.mode=='both' else '仅人脸' if self.mode=='face' else '仅人体'}")
        return True

    def run(self, cam=0):
        self.running = True
        cap = cv2.VideoCapture(cam)
        if not cap.isOpened(): print("无法打开摄像头"); return
        cap.set(3, 640); cap.set(4, 480)
        print(f"\n摄像头已启动 | Q=退出 M=鼠标 C=颜色 D=模式")
        while self.running:
            ret, f = cap.read()
            if not ret: continue
            f = cv2.flip(f, 1)
            gray = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
            fh, fw = f.shape[:2]
            targets = []
            if self.mode in ['face','both']:
                for x,y,w,h in self.face.detectMultiScale(gray, 1.1, 5, minSize=(30,30)):
                    cv2.rectangle(f, (x,y), (x+w,y+h), (0,255,0), 2)
                    cx, cy = x+w//2, y+h//2
                    cv2.circle(f, (cx,cy), 5, (0,255,0), -1)
                    targets.append((cx,cy))
            if self.mode in ['body','both'] and not targets:
                for x,y,w,h in self.detect_bodies(f):
                    if w<50 or h<100: continue
                    cv2.rectangle(f, (x,y), (x+w,y+h), (255,165,0), 2)
                    hx, hy = x+w//2, y+h//6
                    cv2.circle(f, (hx,hy), 8, (255,0,0), -1)
                    targets.append((hx,hy))
            if self.color and not targets:
                for c in self.detect_colors(f):
                    M = cv2.moments(c)
                    if M['m00']>0:
                        cx, cy = int(M['m10']/M['m00']), int(M['m01']/M['m00'])
                        cv2.circle(f, (cx,cy), 10, (255,255,255), -1)
                        targets.append((cx,cy))
            if targets and self.mouse_on:
                tx, ty = targets[0]
                sx, sy = int((tx/fw)*self.sw), int((ty/fh)*self.sh)
                if self.lx: sx, sy = int(self.lx+(sx-self.lx)*0.3), int(self.ly+(sy-self.ly)*0.3)
                pyautogui.moveTo(sx, sy)
                self.lx, self.ly = sx, sy
            cv2.imshow('Head Tracker', f)
            if not self.handle_key(cv2.waitKey(1)&0xFF): break
        cap.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    print("\n" + "="*40 + "\n   人头追踪工具 v5.0\n" + "="*40)
    app = App()
    print("\n检测模式: 1.人脸+人体 2.仅人脸 3.仅人体")
    try: m = input("选择(默认1): ").strip() or "1"
    except: m = "1"
    app.mode = {'1':'both','2':'face','3':'body'}.get(m,'both')
    print("\n颜色: 1.关 2.红 3.黄 4.紫")
    try: c = input("选择(默认1): ").strip() or "1"
    except: c = "1"
    app.color = {'1':None,'2':'red','3':'yellow','4':'purple'}.get(c)
    try: cam = int(input("摄像头索引(默认0): ").strip() or "0")
    except: cam = 0
    try: app.run(cam)
    except KeyboardInterrupt: pass
