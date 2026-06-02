"""
人头追踪工具 v6.0 - 屏幕捕获模式
"""

import cv2
import pyautogui
import numpy as np
from PIL import ImageGrab
import json
import os
import sys
import ctypes

pyautogui.FAILSAFE = True
pyautogui.PAUSE = 0.01

try:
    user32 = ctypes.windll.user32
    user32.SetProcessDPIAware()
    SCREEN_WIDTH, SCREEN_HEIGHT = user32.GetSystemMetrics(0), user32.GetSystemMetrics(1)
except:
    SCREEN_WIDTH, SCREEN_HEIGHT = pyautogui.size()

COLOR_RANGES = {
    'red': {'name': '红色',ranges': [{'lower': np.array([0,100,100]), 'upper': np.array([10,255,255])}, {'lower': np.array([156,100,100]), 'upper': np.array([180255,255])}], 'display': (0,0,255)},
    'yellow': {'name': '黄色', 'ranges': [{'lower': np.array([20,100,100]), 'upper': np.array([35,255,255])}], 'display': (0,255,255)},
    'purple': {'name': '紫色', 'ranges': [{'lower': np.array([125,100,100]), 'upper': np.array([155,255,255])}], 'display': (255,0,255)}
}

class KeyConfig:
    DEFAULT_KEYS = {'quit': 'q', 'mouse_toggle': 'm', 'color_cycle': 'c', 'mode_cycle': 'd'}
    def __init__(self):
        self.keys = self.DEFAULT_KEYS.copy()
        self.config_file = os.path.join(os.path.dirname(sys.executable if getattr(sys, 'frozen', False) else __file__), 'key_config.json')
        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    self.keys(json.load(f))
            except: pass
    def save(self):
        try:
            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(self.keys, f)
        except: pass
    def get_key(self, a): return self.get(a, 'q')
    def get_code(self, a): return ord(self.get_key(a))

class HeadTracker:
    def __init__(self):
        self.face = None
        try:
            self.face = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        except: pass
        self.hog = cv2.HOGDescriptor()
        self.hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
        self.running = False
        self.mouse_on = True
        self.color = None
        self.mode = 'body  # 默认人体检测
        self.lx, self.ly = None, None

    def detect_bodies(self, f):
        h, w = f.shape[:2]
        s = 640/w if w>640 else 1
        r = cv2.resize(f, (int(w*s), int(h*s)))
        try:
            boxes, _ = self.hog.detectMultiScale(r, winStride=(8,8), padding=(8,8 scale=1.05)
            return [(int(x/s), int(y/s), int(bw/s), int(bh/s)) for x,y,bw,bh in boxes if bw/s>50 and bh/s>100]
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
        cnt, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return [c for c in cnts if cv2.contourArea(c) > 500]

    def handle_key(self, k, kc):
        if k == kc.get_code('quit'): return False
        elif k == kc.get_code('mouse_toggle'): self.mouse_on = not self.mouse_on; print(f"鼠标:{'开' if self.mouse_on else '关'}")
        elif k == kc.get_code('color_cycle'):
            cs = [None,'red','yellow','purple']
            self.color = cs[(cs.index(self.color)+1 if self.color in cs else 0) % 4]
            print(f"颜色:{COLOR_RANGES[self.color]['name'] if self.color else '关'}")
        elif k == kc.get_code('mode_cycle'):
            ms = ['body','face','both']
            self.mode = ms[(ms.index(self.mode)+1)%3]
            print(f"模式:{'人体' if self.mode=='body' else '人脸' if self.mode=='face' else '人脸+人体'}")
        return True

    def run(self, kc):
        self.running = True
        print(f"\n屏幕捕获已启动 | {SCREEN_WIDTH}x{SCREEN_HEIGHT}")
        print(f"键位: Q=退出 M=鼠标 C=颜色 D=模式")
        
        while self.running:
            screenshot = ImageGrab.grab()
            f = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)
            fh, fw = f.shape[:2]
            gray = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
            targets = []
            
            if self.mode in ['face','both']:
                if self.face is not None:
                    for x,y,w,h in self.face.detectMultiScale(gray, 1.1, 5, minSize=(30,30)):
                        cv2.rectangle(f, (x,y), (x+w,y+h), (0,255,0), 2)
                        cx, cy = x+w//2, y+h//2
                        cv2.circle(f, (cx,cy), 5, (0,255,0), -1)
                        targets.append((cx,cy))
            
            if self.mode in ['body','both'] and not targets:
                for x,y,w,h in self.detect_bodies(f):
                    cv2.rectangle(f, (x,y), (x+w,y+h), (255,165,0), 2)
                    hx, hy = x+w//2, y+h//6
                    cv2.circle(f, (hx,hy), 10, (255,0,0), -1)
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
                sx, sy = int((tx/fw)*SCREEN_WIDTH),((ty/fh)*SCREEN_HEIGHT)
                if self.lx: sx, sy = int(self.lx+(sx-self.lx)*0.3), int(self.ly+(sy-self.ly)*0.3)
                pyautogui.moveTo(sx, sy)
                self.lx, self.ly = sx, sy
            
            df = cv2.resize(f, (fw//2, fh//2))
            cv2.putText(df, f"Target:{len(targets)} Mouse:{'ON' if self.mouse_on else 'OFF'}", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
            cv2.imshow('Screen Tracker', df)
            if not self.handle_key(cv2.waitKey(1)&0xFF, kc): break
        
        cv2.destroyAllWindows()

if __name__ == '__main__':
    print("\n" + "="*40 + "\n   屏幕追踪工具 v6.0\n" + "="*40)
    kc = KeyConfig()
    
    print("\n1.启动  2.改键位  3.看键位")
    try: c = input("选择(默认1): ").strip() or "1"
    except: c = "1"
    
    if c == '2':
        print("\n自定义键位（输入字母，回车跳过）")
        for a,d in [('quit','退出'),('mouse_toggle','鼠标'),('color_cycle','颜色'),('mode_cycle','模式')]:
            try:
                n = input(f"{d}[{kc.get_key(a).upper()}]: ").strip().lower()
                if len(n)==1 and n.isalpha(): kc.keys[a]=n
            except: pass
        kc.save()
    elif c == '3':
        print(f"退出:{kc.get_key('quit').upper()} 鼠标:{kc.get_key('mouse_toggle').upper()} 颜色:{kc.get_key('color_cycle').upper()} 模式:{kc.get_key('mode_cycle').upper()}")
        input("按回车退出...")
        exit()
    
    print("\n模式: 1.人体 2.人脸 3.人脸+人体")
    try: m = input("选择(默认1): ").strip() or "1"
    except: m = "1"
    
    print("\n颜色: 1.关 2.红 3.黄 4.紫")
    try: cl = input("选择(默认1): ").strip() or "1"
    except: cl = "1"
    
    app = HeadTracker()
    app.mode = {'1':'body','2':'face','3':'both'}.get(m,'body')
    app.color = {'1':None,'2':'red','3':'yellow','4':'purple'}.get(cl)
    
    try: app.run(kc)
    except KeyboardInterrupt: pass

