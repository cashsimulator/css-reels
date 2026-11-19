"""Weeks, not months — CFO reel builder. Usage: python3 build.py test|full [appclip.mp4]"""
import sys, subprocess, numpy as np, cv2, soundfile as sf
from PIL import Image, ImageDraw, ImageFont

R = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
W, H, FPS, SR = 1080, 1920, 30, 24000
FB = f"{R}/assets/fonts/Poppins-Bold.ttf"
FXB = f"{R}/assets/fonts/Poppins-Bold.ttf"
import os
if not os.path.exists(FXB): FXB = FB

# ---------- voice lines ----------
def load_trim(key):
    a, sr = sf.read(f"{R}/audio/{key}.wav"); assert sr == SR
    idx = np.where(np.abs(a) > 0.012)[0]
    pad = int(0.04 * SR)
    return a[max(0, idx[0] - pad): idx[-1] + pad].astype(np.float32)
V = {k: load_trim(k) for k in ["c1","c2","o1","c3","c4","c5","c6","c7","c8","c9","c10"]}
dur = {k: len(v) / SR for k, v in V.items()}
T = {}
T["c1"] = 0.05
T["c2"] = T["c1"] + dur["c1"] + 0.45
T["o1"] = T["c2"] + dur["c2"] + 0.12
T["c3"] = T["o1"] + dur["o1"] + 0.25
T["c4"] = T["c3"] + 2.0
T["c5"] = T["c4"] + 2.0
T["c6"] = T["c5"] + max(dur["c5"] + 0.5, 2.6)
APP1 = T["c6"] + max(dur["c6"] + 0.5, 2.6)          # app: typing numbers, line goes red
RED = APP1 + 2.6                           # the moment it turns red
T["c7"] = RED + 0.15
APP2 = max(T["c7"] + dur["c7"] + 0.1, APP1 + 4.6)   # app: slider dragged
SLIDE = APP2 + 0.7
T["c8"] = APP2 + 0.25
T["c9"] = T["c8"] + dur["c8"] + 0.08
APP_END = min(APP2 + 3.2, APP1 + 8.0)
T["c10"] = T["c9"] + dur["c9"] + 0.3
END = T["c10"] + dur["c10"] + 0.3          # end card starts
TOTAL = END + 1.5
SPEAKER = {k: "cfo" for k in V}; SPEAKER["o1"] = "owner"
CAP = {"c1": "You think your money lasts a few months.", "c2": "It probably lasts a few weeks.", "o1": "No way.",
       "c3": "Rent.", "c4": "Payroll.", "c5": "Suppliers you pay before you get paid.", "c6": "And clients who always pay late.",
       "c7": "There. That's the week you're in trouble.", "c8": "Get paid faster and look.", "c9": "You just bought yourself months.",
       "c10": "Know your number before it surprises you."}
N = int(round(TOTAL * FPS))

# ---------- mouth openness per frame, per speaker ----------
def env_track(speaker):
    a = np.zeros(int((TOTAL + 1) * SR), np.float32)
    for k, v in V.items():
        if SPEAKER[k] == speaker:
            s = int(T[k] * SR); a[s:s + len(v)] += v
    e = np.array([np.sqrt(np.mean(a[int(i / FPS * SR):int((i + 1) / FPS * SR)] ** 2)) for i in range(N)])
    e = e / (np.percentile(e[e > 0], 95) + 1e-9)
    op = np.clip((e - 0.08) / 0.8, 0, 1)
    return np.convolve(op, [0.25, 0.5, 0.25], mode="same")
OPEN = {"cfo": env_track("cfo"), "owner": env_track("owner")}

# ---------- source images with jaw warp ----------
class Face:
    def __init__(self, path, up, mouth_xy, halfw, top, bot0, bot1, close_px, open_px):
        img = cv2.imread(path)
        self.img = cv2.resize(img, None, fx=up, fy=up, interpolation=cv2.INTER_LANCZOS4)
        blur = cv2.GaussianBlur(self.img, (0, 0), 1.2 * up)
        self.img = cv2.addWeighted(self.img, 1.45, blur, -0.45, 0)   # light unsharp
        h, w = self.img.shape[:2]; self.up = up
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        mx, my = mouth_xy[0] * up, mouth_xy[1] * up
        wx = np.clip(1 - (np.abs(xx - mx) / (halfw * up)) ** 2, 0, 1)
        t0, t1, b0, b1 = (my - 3 * up), (my + 1 * up), (my + bot0 * up), (my + bot1 * up)
        wy = np.where(yy < t0, 0, np.where(yy < t1, (yy - t0) / (t1 - t0), np.where(yy < b0, 1, np.where(yy < b1, 1 - (yy - b0) / (b1 - b0), 0))))
        self.wmap = (wx * wy).astype(np.float32); self.xx, self.yy = xx, yy
        self.close, self.open = close_px * up, open_px * up
        self.cache = {}
    def at(self, o):
        q = round(float(o) * 20) / 20
        if q not in self.cache:
            sh = np.float32(self.close * (1 - q) - self.open * q)
            self.cache[q] = cv2.remap(self.img, self.xx, self.yy + self.wmap * sh, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        return self.cache[q]
CFO = Face(f"{R}/src/cfo_office.png", 2, (518, 229), 34, 0, 22, 46, 6.5, 2.0)
OWN = Face(f"{R}/src/owner.png", 1.25, (452, 576), 48, 0, 30, 70, 9.0, 3.0)

sys.path.insert(0, R); import broll_anim
ANIM = broll_anim.build(f"{R}/src")
BROLL = {k: cv2.resize(cv2.imread(f"{R}/src/broll_{n}.png"), (W, H), interpolation=cv2.INTER_LANCZOS4)
         for k, n in [("c3", "rent"), ("c4", "payroll"), ("c5", "suppliers"), ("c6", "late_clients")]}

def cam(face, o, cx, cy, h_src, dx=0, dy=0, rot=0):
    img = face.at(o); up = face.up
    hs = h_src * up; ws = hs * 9 / 16
    ih, iw = img.shape[:2]
    cx = np.clip(cx * up, ws / 2, iw - ws / 2); cy = np.clip(cy * up, hs / 2, ih - hs / 2)
    s = H / hs
    M = cv2.getRotationMatrix2D((float(cx), float(cy)), rot, s)
    M[0, 2] += W / 2 - cx + dx; M[1, 2] += H / 2 - cy + dy
    return cv2.warpAffine(img, M, (W, H), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

def ease(x): x = min(max(x, 0), 1); return 1 - (1 - x) ** 3
def lerp(a, b, x): return a + (b - a) * x

# ---------- look: dark grade + faint scanlines ----------
SCAN = np.ones((H, 1, 1), np.float32); SCAN[::3] = 0.93
VIG = None
def grade(f):
    global VIG
    if VIG is None:
        yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
        r = np.sqrt(((xx - W / 2) / (W / 2)) ** 2 + ((yy - H / 2) / (H / 2)) ** 2)
        VIG = (1 - 0.38 * np.clip(r - 0.45, 0, 1) ** 1.4)[..., None].astype(np.float32)
    return np.clip(f.astype(np.float32) * VIG * SCAN, 0, 255).astype(np.uint8)

# ---------- text overlays (RGBA, cached) ----------
def rgba_text_block(lines, font, fill, stroke, box=None, pad=(36, 18), radius=28, maxw=960):
    d0 = ImageDraw.Draw(Image.new("RGBA", (10, 10)))
    sw = 0 if box else 10
    sizes = [d0.textbbox((0, 0), l, font=font, stroke_width=sw) for l in lines]
    lh = int(font.size * (1.42 if not box else 1.12))
    tw = max(b[2] - b[0] for b in sizes); th = lh * len(lines)
    bw, bh = tw + (pad[0] * 2 if box else 40), th + (pad[1] * 2 if box else 40)
    im = Image.new("RGBA", (bw, bh), (0, 0, 0, 0)); d = ImageDraw.Draw(im)
    if box: d.rounded_rectangle((0, 0, bw - 1, bh - 1), radius=radius, fill=box)
    for i, l in enumerate(lines):
        b = d.textbbox((0, 0), l, font=font, stroke_width=sw)
        x = (bw - (b[2] - b[0])) // 2 - b[0]
        y = (pad[1] if box else 20) + i * lh - int(font.size * 0.08)
        d.text((x, y), l, font=font, fill=fill, stroke_width=sw, stroke_fill=stroke)
    return np.array(im)

def wrap(text, font, maxw):
    d0 = ImageDraw.Draw(Image.new("RGBA", (10, 10))); words = text.split(); lines = []; cur = ""
    for w_ in words:
        t = (cur + " " + w_).strip()
        if d0.textlength(t, font=font) <= maxw: cur = t
        else: lines.append(cur); cur = w_
    lines.append(cur); return lines

FCAP = ImageFont.truetype(FXB, 78)
CAPIMG = {k: rgba_text_block(wrap(v, FCAP, 900), FCAP, (255, 255, 255), (0, 0, 0)) for k, v in CAP.items()}
FHOOK = ImageFont.truetype(FXB, 70)
HOOK = rgba_text_block(["How long would your", "money really last?"], FHOOK, (10, 10, 10), None, box=(255, 229, 0, 255))
FCHIP = ImageFont.truetype(FXB, 64)
CHIPS = [("Rent", "c3"), ("Payroll", "c4"), ("Suppliers", "c5"), ("Late clients", "c6")]
CHIPIMG = {n: rgba_text_block([n], FCHIP, (10, 10, 10), None, box=(255, 255, 255, 255), pad=(40, 14), radius=40) for n, _ in CHIPS}
FEND1 = ImageFont.truetype(FXB, 86); FEND2 = ImageFont.truetype(FB, 60)

def paste(frame, rgba, cx, cy, scale=1.0, alpha=1.0):
    if scale != 1.0:
        rgba = cv2.resize(rgba, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
    h, w = rgba.shape[:2]; x0, y0 = int(cx - w / 2), int(cy - h / 2)
    xa, ya, xb, yb = max(x0, 0), max(y0, 0), min(x0 + w, W), min(y0 + h, H)
    if xa >= xb or ya >= yb: return
    sub = rgba[ya - y0:yb - y0, xa - x0:xb - x0].astype(np.float32)
    a = sub[..., 3:4] / 255 * alpha
    roi = frame[ya:yb, xa:xb].astype(np.float32)
    frame[ya:yb, xa:xb] = (roi * (1 - a) + sub[..., 2::-1] * a).astype(np.uint8)   # RGBA->BGR

def pop(t, t0):  # pop-in scale
    x = (t - t0) / 0.18
    if x < 0: return 0
    if x >= 1: return 1.0
    return 0.6 + 0.55 * np.sin(x * np.pi * 0.75)

# ---------- app footage ----------
APPCLIP = sys.argv[2] if len(sys.argv) > 2 else None
def placeholder(t, which):
    f = np.full((H, W, 3), (24, 20, 18), np.uint8)
    cv2.rectangle(f, (90, 420), (W - 90, H - 520), (90, 90, 90), 4)
    lines = ["APP RECORDING", "typing numbers -> cash line" if which == 1 else "drag 'clients pay faster'", "goes RED" if which == 1 else "red point moves right"]
    for i, l in enumerate(lines):
        cv2.putText(f, l, (150, 760 + i * 110), cv2.FONT_HERSHEY_DUPLEX, 1.6, (200, 200, 200), 3, cv2.LINE_AA)
    if which == 1 and t >= RED:
        k = max(0, 1 - (t - RED) / 0.6)
        f = cv2.addWeighted(f, 1, np.full_like(f, (40, 40, 230)), 0.45 * k + 0.12, 0)
    return f
app_cap = None
def app_frame(t, which):
    return placeholder(t, which)

# ---------- shot list ----------
def shot(i):
    t = i / FPS
    oc, oo = OPEN["cfo"][i], OPEN["owner"][i]
    if t < T["c2"]:                                   # S1 hook close-up, slow push
        f = cam(CFO, oc, 515, 235, 400)
    elif t < T["o1"]:                                 # S2 snap zoom-in
        f = cam(CFO, oc, 515, 235, 400)
    elif t < T["c3"] - 0.1:                           # S3 owner
        x = (t - T["o1"]) / 0.8
        shake = np.sin(t * 60) * 6 * max(0, 1 - x * 2)
        f = cam(OWN, oo, 440, 640, 860)
    elif t < APP1:                                    # b-roll: rent / payroll / suppliers / late clients
        k = "c3" if t < T["c4"] else "c4" if t < T["c5"] else "c5" if t < T["c6"] else "c6"
        f = ANIM[{"c3": "rent", "c4": "payroll", "c5": "suppliers", "c6": "late_clients"}[k]].frame(t - T[k])
    elif t < APP2:
        f = app_frame(t, 1)
        if t >= RED:  # punch-in zoom on the red moment
            z = 1.0
            f = cv2.warpAffine(f, np.float32([[z, 0, (1 - z) * W / 2], [0, z, (1 - z) * H * 0.55]]), (W, H))
    elif t < APP_END:
        f = app_frame(t, 2)
    elif t < T["c10"]:                                # S9 CFO straight to camera
        f = cam(CFO, oc, 515, 235, 400)
    elif t < END:                                     # S10 tighter for the last line
        f = cam(CFO, oc, 515, 235, 400)
    else:
        f = None
    return f

def end_card(clean):
    f = np.full((H, W, 3), (14, 12, 11), np.uint8)
    if clean: return f
    a = rgba_text_block(["Cash Scenario", "Simulator"], FEND1, (255, 255, 255), (0, 0, 0))
    b = rgba_text_block(["cashsimulator.app"], FEND2, (10, 10, 10), None, box=(255, 229, 0, 255), pad=(40, 16))
    paste(f, a, W / 2, H / 2 - 110); paste(f, b, W / 2, H / 2 + 120)
    return f

def overlays(f, t):
    if t < T["o1"]:
        paste(f, HOOK, W / 2, 330)
    for k in CAP:
        if T[k] - 0.03 <= t < T[k] + dur[k] + 0.2:
            paste(f, CAPIMG[k], W / 2, H * 0.74)
    if T["c3"] <= t < APP1:
        cur = [(n, k) for n, k in CHIPS if t >= T[k]][-1]
        paste(f, CHIPIMG[cur[0]], W / 2, 190, scale=pop(t, T[cur[1]]))
    return f

def frame_pair(i):
    t = i / FPS
    if t >= END:
        return end_card(False), end_card(True)
    base = grade(shot(i))
    return overlays(base.copy(), t), base

# ---------- audio ----------
def synth():
    n = int(TOTAL * SR); mix = np.zeros(n, np.float32); tt = np.arange(n) / SR
    vo = np.zeros(n, np.float32)
    for k, v in V.items():
        s = int(T[k] * SR); e = min(n, s + len(v)); vo[s:e] += v[:e - s] * 1.1
    # tense bed: 92 bpm kick + low pulse drone + soft hats, ducked under the voice
    bed = np.zeros(n, np.float32); beat = 60 / 92
    kick_t = np.arange(0, 0.35, 1 / SR); kick = np.sin(2 * np.pi * (50 + 90 * np.exp(-kick_t * 30)) * kick_t) * np.exp(-kick_t * 9)
    hat_t = np.arange(0, 0.05, 1 / SR); rng = np.random.default_rng(1)
    for b in np.arange(0, TOTAL, beat):
        s = int(b * SR); e = min(n, s + len(kick)); bed[s:e] += kick[:e - s] * 0.55
        h = int((b + beat / 2) * SR)
        if h + len(hat_t) < n: bed[h:h + len(hat_t)] += rng.normal(0, 1, len(hat_t)).astype(np.float32) * np.exp(-hat_t * 90) * 0.05
    drone = (np.sin(2 * np.pi * 55 * tt) + 0.5 * np.sin(2 * np.pi * 82.4 * tt)) * (0.55 + 0.45 * np.sin(2 * np.pi * tt / beat / 2) ** 2)
    bed += drone.astype(np.float32) * 0.07
    venv = np.convolve(np.abs(vo), np.ones(int(0.12 * SR)) / int(0.12 * SR), mode="same")
    duck = 1 - 0.55 * np.clip(venv / 0.03, 0, 1)
    endfade = np.clip((TOTAL - tt) / 0.4, 0, 1)
    mix += bed * duck.astype(np.float32) * 0.42 * endfade.astype(np.float32)
    def add(s_t, sig, g=1.0):
        s = int(s_t * SR); e = min(n, s + len(sig)); mix[s:e] += sig[:e - s] * g
    # tick for each chip
    tk = np.arange(0, 0.04, 1 / SR); tick = (np.sin(2 * np.pi * 2600 * tk) * np.exp(-tk * 160)).astype(np.float32)
    for _, k in CHIPS: add(T[k], tick, 0.5)
    # whoosh on the snap zoom and the slider
    def whoosh(d, f0, f1):
        x = np.arange(0, d, 1 / SR); noise = rng.normal(0, 1, len(x)).astype(np.float32)
        out = np.zeros_like(noise); fc = np.linspace(f0, f1, len(x)); y = 0.0
        a = 1 - np.exp(-2 * np.pi * fc / SR)
        for j in range(len(x)): y += a[j] * (noise[j] - y); out[j] = y
        return out * np.sin(np.pi * x / d) ** 2 * 1.2
    add(SLIDE - 0.1, whoosh(0.55, 300, 4000), 0.5)
    # bass drop + alarm at the red moment
    bd_t = np.arange(0, 1.0, 1 / SR); bd = np.tanh(2.5 * np.sin(2 * np.pi * (110 * np.exp(-bd_t * 2.2) + 32) * bd_t)) * np.exp(-bd_t * 2.5)
    add(RED, bd.astype(np.float32), 0.55)
    al_t = np.arange(0, 0.5, 1 / SR); al = np.sign(np.sin(2 * np.pi * np.where((al_t % 0.25) < 0.125, 880, 660) * al_t)) * 0.25 * (al_t < 0.48)
    add(RED + 0.05, al.astype(np.float32), 0.22)
    # soft hit on end card
    add(END, (bd[:int(0.6 * SR)] * 0.6).astype(np.float32), 0.4)
    mix += vo
    peak = np.max(np.abs(mix)); mix = mix / max(peak, 1e-6) * 0.89
    sf.write(f"{R}/out/audio.wav", mix, SR)

def ff(out):
    return subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
                             "-i", f"{R}/out/audio.wav", "-c:v", "libx264", "-crf", "18", "-preset", "veryfast", "-pix_fmt", "yuv420p",
                             "-c:a", "aac", "-b:a", "192k", "-shortest", "-movflags", "+faststart", out], stdin=subprocess.PIPE)

if __name__ == "__main__":
    os.makedirs(f"{R}/out", exist_ok=True)
    print({k: round(v, 2) for k, v in T.items()}, "APP1", round(APP1, 2), "RED", round(RED, 2), "APP2", round(APP2, 2), "APP_END", round(APP_END, 2), "END", round(END, 2), "TOTAL", round(TOTAL, 2))
    if sys.argv[1] == "test":
        for t in [0.3, T["c2"] + 0.5, T["o1"] + 0.3, T["c4"] + 0.2, T["c6"] + 0.5, RED + 0.3, T["c10"] + 1, END + 0.5]:
            a, _ = frame_pair(int(t * FPS)); cv2.imwrite(f"{R}/out/t_{t:.1f}.jpg", a[::2, ::2])
    else:
        synth()
        p1, p2 = ff(f"{R}/out/cap_tmp.mp4"), ff(f"{R}/out/clean_tmp.mp4")
        for i in range(N):
            a, b = frame_pair(i); p1.stdin.write(a.tobytes()); p2.stdin.write(b.tobytes())
        p1.stdin.close(); p2.stdin.close()
        assert p1.wait() == 0 and p2.wait() == 0
        os.replace(f"{R}/out/cap_tmp.mp4", f"{R}/out/cfo_reel_captions.mp4"); os.replace(f"{R}/out/clean_tmp.mp4", f"{R}/out/cfo_reel_clean.mp4")
        # cover frame
        c = grade(cam(CFO, 0.3, 515, 240, 270)); paste(c, rgba_text_block(["How long", "would your", "money really", "last?"], ImageFont.truetype(FXB, 120), (10, 10, 10), None, box=(255, 229, 0, 255), pad=(44, 24)), W / 2, 520)
        cv2.imwrite(f"{R}/out/cfo_reel_cover.png", c)
