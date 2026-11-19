"""Shared engine for the Cash Scenario Simulator story reels (9:16)."""
import sys, os, re, json, subprocess, numpy as np, cv2, soundfile as sf
from PIL import Image, ImageDraw, ImageFont
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "engine"))
import broll_anim as BA
from kokoro_onnx import Kokoro

R = f"{ROOT}/assets/chars"; V3 = os.environ.get("CSS_WORK", f"{ROOT}/work"); MODELS = os.environ.get("CSS_MODELS", os.path.expanduser("~/.cache/css-models"))
W, H, FPS, SR = 1080, 1920, 30, 24000
SHOTS = f"{ROOT}/assets/shots"
_K = None
BASE_SPEED = float(os.environ.get("CSS_VOICE_SPEED", "1.12"))   # ~1.2x: lively but every word clear

def squeeze_pauses(a, sr, keep=0.11, thr=0.008):
    """shorten any pause inside a line (commas, full stops) to `keep` seconds, with tiny fades so it never clicks"""
    hop = int(0.01 * sr); env = np.array([np.sqrt(np.mean(a[i:i + hop] ** 2)) for i in range(0, len(a), hop)])
    quiet = env < thr; out = []; i = 0; last = 0
    while i < len(quiet):
        if quiet[i]:
            j = i
            while j < len(quiet) and quiet[j]: j += 1
            if (j - i) * 0.01 > keep + 0.02 and i > 0 and j < len(quiet):
                cut0 = i * hop + int(keep / 2 * sr); cut1 = j * hop - int(keep / 2 * sr)
                seg = a[last:cut0].copy(); f = min(int(0.006 * sr), len(seg))
                if f: seg[-f:] *= np.linspace(1, 0, f)
                out.append(seg); last = cut1
                nxt = a[last:last + int(0.006 * sr)]
            i = j
        else: i += 1
    tail = a[last:].copy(); f = min(int(0.006 * sr), len(tail))
    if out and f: tail[:f] *= np.linspace(0, 1, f)
    out.append(tail); return np.concatenate(out).astype(np.float32)

def mouth_curve(a, fps=30, sr=24000, lead=0.04):
    """mouth openness per video frame. Speech band only, fast open, quick close, mouth leads the sound by ~40 ms."""
    from scipy.signal import butter, sosfilt
    sos = butter(4, [250, 3000], btype="band", fs=sr, output="sos"); b = sosfilt(sos, a)
    hop = int(0.01 * sr); env = np.array([np.sqrt(np.mean(b[i:i + hop] ** 2)) for i in range(0, len(b), hop)])
    if env.max() <= 0: return np.zeros(int(len(a) / sr * fps) + 2)
    env = env / (np.percentile(env[env > 0], 92) + 1e-9)
    o = np.clip((env - 0.10) / 0.75, 0, 1) ** 0.7
    sm = np.zeros_like(o); v = 0.0                       # attack 20 ms, release 60 ms
    for k, x in enumerate(o):
        v += (x - v) * (0.6 if x > v else 0.18); sm[k] = v
    n = int(len(a) / sr * fps) + 2; t = (np.arange(n) / fps + lead) * 100
    return np.interp(t, np.arange(len(sm)), sm, right=0.0)

# ---------------- voice ----------------
def tts(name, lines):
    """lines: list of dict(key, voice, text, speed). Caches per video."""
    global _K
    os.makedirs(f"{V3}/{name}/audio", exist_ok=True)
    for L in lines:
        p = f"{V3}/{name}/audio/{L['key']}.wav"
        sp = BASE_SPEED * L.get("speed", 1.0)
        meta = p + ".txt"; sig = f"v2|{L['voice']}|{sp:.3f}|{L['text']}"
        if os.path.exists(p) and os.path.exists(meta) and open(meta).read() == sig: continue
        if _K is None: _K = Kokoro(f"{MODELS}/kokoro-v1.0.onnx", f"{MODELS}/voices-v1.0.bin")
        s, sr = _K.create(L["text"], voice=L["voice"], speed=sp, lang="en-us")
        i = np.where(np.abs(s) > 0.012)[0]; pd = int(0.03 * sr); s = squeeze_pauses(s[max(0, i[0] - pd):i[-1] + pd], sr)
        sf.write(p, s, sr); open(meta, "w").write(sig)
    return {L["key"]: sf.read(f"{V3}/{name}/audio/{L['key']}.wav")[0].astype(np.float32) for L in lines}

# ---------------- look ----------------
SCAN = np.ones((H, 1, 1), np.float32); SCAN[::3] = 0.93
_yy, _xx = np.mgrid[0:H, 0:W].astype(np.float32)
VIG = (1 - 0.38 * np.clip(np.sqrt(((_xx - W / 2) / (W / 2)) ** 2 + ((_yy - H / 2) / (H / 2)) ** 2) - 0.45, 0, 1) ** 1.4)[..., None].astype(np.float32)
def pixelate(f, factor=4, levels=28):
    """PS2 / pixel look: chunky pixels and a slightly reduced palette"""
    h, w = f.shape[:2]; sm = cv2.resize(f, (w // factor, h // factor), interpolation=cv2.INTER_AREA)
    q = 256 // levels; sm = np.clip(sm.astype(np.int32) // q * q + q // 2, 0, 255).astype(np.uint8)
    return cv2.resize(sm, (w, h), interpolation=cv2.INTER_NEAREST)
def grade(f): return np.clip(f.astype(np.float32) * VIG * SCAN, 0, 255).astype(np.uint8)

# ---------------- talking faces ----------------
class Face:
    """864x1536 PS2 portrait, mouth closed in the source. Speaking pulls the lower lip/jaw down and shows a dark mouth gap."""
    def __init__(self, path, mouth, halfw, open_px=7, jaw=60):
        img = cv2.imread(path); assert img.shape[:2] == (1536, 864), path
        self.img = cv2.resize(img, (W, H), interpolation=cv2.INTER_LANCZOS4); s = W / 864
        mx, my = mouth[0] * s, mouth[1] * s; hw = halfw * s
        wx = np.clip(1 - (np.abs(_xx - mx) / hw) ** 2, 0, 1)
        jaw_ = jaw * s
        wy = np.where(_yy < my, 0, np.where(_yy < my + jaw_ * 0.5, 1, np.clip(1 - (_yy - my - jaw_ * 0.5) / (jaw_ * 0.5), 0, 1)))
        self.wmap = (wx * wy).astype(np.float32); self.open = open_px * s
        self.mx, self.my, self.hw = mx, my, hw; self.cache = {}
    def at(self, o):
        q = round(float(o) * 20) / 20
        if q not in self.cache:
            sh = np.float32(-self.open * q)
            f = cv2.remap(self.img, _xx, _yy + self.wmap * sh, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
            if q > 0:
                m = np.zeros((H, W), np.float32)
                cv2.ellipse(m, (int(self.mx), int(self.my + self.open * q * 0.45)), (int(self.hw * 0.62), max(1, int(self.open * q * 0.5))), 0, 0, 360, 1.0, -1)
                m = cv2.GaussianBlur(m, (0, 0), 2.2)[..., None]
                f = (f.astype(np.float32) * (1 - 0.85 * m) + np.array([28, 22, 40], np.float32) * 0.85 * m).astype(np.uint8)
            self.cache[q] = f
        return self.cache[q]
    def framed(self, o, zoom=1.0, cy=None):
        f = self.at(o)
        if zoom == 1.0: return f
        cy = self.my if cy is None else cy * W / 864
        M = np.float32([[zoom, 0, (1 - zoom) * W / 2], [0, zoom, (1 - zoom) * cy]])
        return cv2.warpAffine(f, M, (W, H), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

# the CFO from the first reel (different source size, own warp)
_src = open(f"{ROOT}/engine/cfo_build_ref.py").read()
exec(_src[_src.index("class Face:"):_src.index("CFO = Face(")].replace("class Face:", "class CFOFace:")); exec(_src[_src.index("def cam("):_src.index("def ease(x)")])
_CFO = None
def cfo(o):
    global _CFO
    if _CFO is None: _CFO = CFOFace(f"{R}/cfo_office.png", 2, (518, 229), 34, 0, 22, 46, 6.5, 2.0)
    return cam(_CFO, o, 515, 235, 400)

# ---------------- extra particles ----------------
class Snow:
    def __init__(self, box, n=90, seed=1, speed=(40, 90)):
        rng = np.random.default_rng(seed); s = W / 864
        self.box = [v * s for v in box]; x0, y0, x1, y1 = self.box
        self.x = rng.uniform(x0, x1, n); self.y = rng.uniform(y0, y1, n); self.v = rng.uniform(*speed, n) * s
        self.ph = rng.uniform(0, 6.28, n); self.r = rng.uniform(1.5, 3.5, n)
    def draw(self, f, t):
        x0, y0, x1, y1 = self.box; L = np.zeros(f.shape[:2], np.float32)
        for x, y, v, p, r in zip(self.x, self.y, self.v, self.ph, self.r):
            yy = y0 + (y - y0 + v * t) % (y1 - y0); xx = x + 6 * np.sin(t * 1.3 + p)
            if x0 <= xx < x1: cv2.circle(L, (int(xx), int(yy)), int(r), 1.0, -1, cv2.LINE_AA)
        return np.clip(f.astype(np.float32) + L[..., None] * 200, 0, 255).astype(np.uint8)

def pixelize(path, crop=None):
    """tiny image -> crisp 8x pixel-art frame (used where only small renders exist, on purpose retro)."""
    im = cv2.imread(path)
    big = cv2.resize(im, (im.shape[1] * 8, im.shape[0] * 8), interpolation=cv2.INTER_NEAREST)
    y0 = (big.shape[0] - H) // 2; x0 = (big.shape[1] - W) // 2
    return big[y0:y0 + H, x0:x0 + W].copy()

# ---------------- app inserts ----------------
BG = (22, 17, 14)
_SH = {}
def shot(n):
    if n not in _SH: _SH[n] = cv2.imread(f"{SHOTS}/{n}.png")
    return _SH[n]
def panel(parts, gap=30, width=1000):
    ims = []
    for p in parts:
        if isinstance(p[0], list):                 # side by side pieces: ([(shot,x0,x1),...], y0, y1)
            pieces, y0, y1 = p
            row = [shot(n)[y0:y1, a:b] for n, a, b in pieces]
            im = np.hstack([np.hstack([r, np.full((r.shape[0], 24, 3), BG, np.uint8)]) for r in row])[:, :-24]
        else:
            n, x0, y0, x1, y1 = p; im = shot(n)[y0:y1, x0:x1]
        s = width / im.shape[1]
        ims.append(cv2.resize(im, (width, int(im.shape[0] * s)), interpolation=cv2.INTER_CUBIC if s > 1 else cv2.INTER_AREA))
    h = sum(i.shape[0] for i in ims) + gap * (len(ims) - 1)
    out = np.full((h, width, 3), BG, np.uint8); y = 0
    for i in ims: out[y:y + i.shape[0]] = i; y += i.shape[0] + gap
    return out
def framed(p, hl=None, cy=0.47):
    blur = cv2.resize(cv2.GaussianBlur(cv2.resize(p, (W // 10, max(1, int(p.shape[0] * W / p.shape[1] / 10)))), (0, 0), 2), (W, H))
    f = (blur.astype(np.float32) * 0.25 + np.array(BG, np.float32) * 0.75).astype(np.uint8)
    y0 = int(H * cy - p.shape[0] / 2); x0 = (W - p.shape[1]) // 2
    cv2.rectangle(f, (x0 - 4, y0 - 4), (x0 + p.shape[1] + 3, y0 + p.shape[0] + 3), (70, 62, 56), 2)
    f[y0:y0 + p.shape[0], x0:x0 + p.shape[1]] = p
    for a, b in (hl or []):
        cv2.rectangle(f, (x0 - 10, y0 + a), (x0 + p.shape[1] + 9, y0 + b), (58, 162, 247), 5)
    return f

# panels built from real app screenshots (demo business: services agency/cleaning, $50k/month invoices)
def P_DANGER_HEAD():   # EBITDA 12M +36,000 ... LOW -14,000 FEB + danger banner
    p = panel([([("04_thing_zoom", 0, 549)], 106, 184), ([("04_thing_zoom", 1134, 1430)], 106, 184), ("04_thing_zoom", 25, 290, 1100, 460)], gap=34)
    return framed(p)
def P_FEB():           # danger banner + month-end chart, -14,000 Feb
    return framed(panel([("05_feb", 10, 145, 560, 222), ("05_feb", 0, 265, 1300, 620)], gap=24))
def P_SAFE():          # safe banner + chart, low 33,000
    return framed(panel([("07_safe", 10, 145, 580, 222), ("07_safe", 0, 265, 1300, 620)], gap=24))
def P_WEEKLY():        # week by week: below zero Feb 26, lowest -15,667 Mar 10 when wages go out
    return framed(panel([("03_weekly_danger", 30, 200, 378, 233), ("03_weekly_danger", 378, 200, 756, 233),
                         ("03_weekly_danger", 40, 268, 640, 560)], gap=26))
def P_TERMS30():       # inputs: clients pay after 30 / we pay suppliers after 30
    return framed(panel([([("06b_30zoom", 20, 420), ("06b_30zoom", 900, 1260)], 670, 810)]))
def P_TERMS60():       # inputs: clients pay after 60 / suppliers 30
    p = panel([([("02_inputs60", 16, 250), ("02_inputs60", 494, 680)], 806, 916)])
    s = 1000 / (234 + 24 + 186)
    return framed(p, [(int((838 - 806) * s), int((914 - 806) * s))])

# ---------------- text ----------------
ANTON = f"{ROOT}/assets/fonts/Anton-Regular.ttf"; PIX = f"{ROOT}/assets/fonts/Silkscreen-Regular.ttf"
FCAP = ImageFont.truetype(ANTON, 84); FPIX = ImageFont.truetype(PIX, 92)
def rgba(im): a = np.array(im).astype(np.float32); return a[..., [2, 1, 0]], a[..., 3:4] / 255
def pill(txt):
    d = ImageDraw.Draw(Image.new("RGBA", (10, 10))); b = d.textbbox((0, 0), txt, font=FCAP)
    tw, th = b[2] - b[0], b[3] - b[1]; px, py = 34, 18
    im = Image.new("RGBA", (tw + 2 * px, th + 2 * py), (0, 0, 0, 0)); dd = ImageDraw.Draw(im)
    dd.rounded_rectangle((0, 0, im.width - 1, im.height - 1), radius=26, fill=(0, 0, 0, 225))
    dd.text((px - b[0], py - b[1]), txt, font=FCAP, fill=(255, 255, 255, 255)); return rgba(im)
def pixel_label(lines, size=92, color=(255, 255, 255)):
    f = ImageFont.truetype(PIX, size); d = ImageDraw.Draw(Image.new("RGBA", (10, 10)))
    sizes = [d.textbbox((0, 0), l, font=f) for l in lines]
    w = max(b[2] - b[0] for b in sizes) + 60; lh = int(size * 1.22); h = lh * len(lines) + 40
    im = Image.new("RGBA", (w, h), (0, 0, 0, 0)); dd = ImageDraw.Draw(im)
    for i, l in enumerate(lines):
        b = dd.textbbox((0, 0), l, font=f); x = (w - (b[2] - b[0])) // 2 - b[0]
        for dx in range(-6, 7, 2):
            for dy in range(-6, 7, 2): dd.text((x + dx, 20 + i * lh + dy), l, font=f, fill=(0, 0, 0, 150))
        dd.text((x, 20 + i * lh), l, font=f, fill=color + (255,))
    return rgba(im)
CAP_STYLE = os.environ.get("CSS_CAP_STYLE", "pixel")
MUSIC = os.environ.get("CSS_MUSIC", f"{ROOT}/assets/music/bed.mp3"); MUSIC_START = float(os.environ.get("CSS_MUSIC_START", "35"))
_FSUB = None
def pixel_sub(txt, col):
    """game-style subtitle: pixel font, speaker colour, hard black outline, no box"""
    global _FSUB
    if _FSUB is None: _FSUB = ImageFont.truetype(f"{ROOT}/assets/fonts/VT323.ttf", 96)
    d = ImageDraw.Draw(Image.new("RGBA", (10, 10))); words = txt.split(); lines = [""]
    for w_ in words:
        t_ = (lines[-1] + " " + w_).strip()
        if d.textlength(t_, font=_FSUB) <= 940: lines[-1] = t_
        else: lines.append(w_)
    lh = 92; im = Image.new("RGBA", (1000, lh * len(lines) + 30), (0, 0, 0, 0)); dd = ImageDraw.Draw(im)
    for i, l in enumerate(lines):
        dd.text((500, 15 + lh * i + lh // 2), l, font=_FSUB, fill=tuple(col) + (255,), anchor="mm", stroke_width=7, stroke_fill=(0, 0, 0, 255))
    return rgba(im)
def hud(items):
    """top-left game HUD. Items: "RICK", "TRY 1/3", or bars written as "PROFIT ##########" / "CASH ##........"
    ('#' = full cell, '.' = empty cell). Optional dict form: {"text": ..., "color": "red"}."""
    F = ImageFont.truetype(f"{ROOT}/assets/fonts/VT323.ttf", 58); lh = 60; cell = 26
    cols = {"white": (255, 255, 255), "red": (255, 80, 70), "green": (90, 230, 120), "yellow": (255, 229, 0), "orange": (255, 160, 80)}
    rows = [(it if isinstance(it, dict) else {"text": it}) for it in items]
    parsed = []
    for r in rows:
        t_ = r["text"]; i = min([t_.find(c) for c in "#." if c in t_] or [len(t_)])
        label, bar = t_[:i].rstrip(), t_[i:].replace(" ", "")
        parsed.append((label, bar, cols.get(r.get("color", ""), None)))
    lw = max(int(ImageDraw.Draw(Image.new("RGBA", (1, 1))).textlength(l, font=F)) for l, _, _ in parsed)
    bw = max(len(b) for _, b, _ in parsed) * (cell + 4)
    w = 36 + lw + (24 + bw if bw else 0)
    im = Image.new("RGBA", (w, lh * len(parsed) + 24), (0, 0, 0, 0)); d = ImageDraw.Draw(im)
    d.rectangle((0, 0, w - 1, im.height - 1), fill=(0, 0, 0, 160), outline=(255, 255, 255, 210), width=3)
    for i, (label, bar, col) in enumerate(parsed):
        y = 12 + i * lh
        full = bar.count("#") / max(1, len(bar)) if bar else 1
        c = col or ((90, 230, 120) if full > 0.6 else (255, 229, 0) if full > 0.3 else (255, 80, 70)) if bar else (col or (255, 255, 255))
        d.text((18, y + lh // 2), label, font=F, fill=(255, 255, 255, 255) if bar else c + (255,), anchor="lm")
        for j, ch in enumerate(bar):
            x0 = 18 + lw + 24 + j * (cell + 4); y0 = y + (lh - cell) // 2 + 2
            d.rectangle((x0, y0, x0 + cell, y0 + cell), fill=(c + (255,)) if ch == "#" else (60, 60, 60, 255), outline=(0, 0, 0, 255), width=2)
    return rgba(im)

def paste(frame, ra, cx, cy, k=1.0):
    rgb, al = ra; h, w = rgb.shape[:2]; x0, y0 = int(cx - w / 2), int(cy - h / 2)
    xa, ya, xb, yb = max(x0, 0), max(y0, 0), min(x0 + w, W), min(y0 + h, H)
    sub = rgb[ya - y0:yb - y0, xa - x0:xb - x0]; al = al[ya - y0:yb - y0, xa - x0:xb - x0] * k
    roi = frame[ya:yb, xa:xb].astype(np.float32); frame[ya:yb, xa:xb] = (roi * (1 - al) + sub * al).astype(np.uint8)

# ---------------- word timing -> captions ----------------
def _runs(a, min_gap=0.09, thr=0.006):
    fr = int(0.01 * SR); e = np.array([np.sqrt(np.mean(a[i:i + fr] ** 2)) for i in range(0, max(1, len(a) - fr), fr)])
    on = e > thr; out = []; i = 0
    while i < len(on):
        if on[i]:
            j = i
            while j < len(on) and on[j]: j += 1
            out.append([i * 0.01, j * 0.01]); i = j
        else: i += 1
    m = []
    for r in out:
        if m and r[0] - m[-1][1] < min_gap: m[-1][1] = r[1]
        else: m.append(r)
    return m
def _align(text, a, t0):
    phrases = [p.strip() for p in re.split(r"(?<=[,.?!])\s+", text) if p.strip()]
    rr = _runs(a)
    if len(rr) != len(phrases): rr = [[rr[0][0], rr[-1][1]]]; phrases = [text]
    out = []
    for ph, (s, e) in zip(phrases, rr):
        ws = ph.split(); wt = np.array([len(re.sub(r"[^\w']", "", x)) + 1.5 for x in ws], float)
        ed = s + (e - s) * np.concatenate([[0], np.cumsum(wt) / wt.sum()])
        out += [(ws[i], t0 + ed[i], t0 + ed[i + 1]) for i in range(len(ws))]
    return out
WEAK = {"a", "an", "the", "of", "in", "it", "your", "to", "for", "if", "is", "and", "but", "you", "we", "what", "how", "any", "before", "at", "on", "me", "my", "i", "so", "our", "by", "with"}
def captions_for(text, a, t0, line_end, upper=True):
    words = _align(text, a, t0); ph = []; chunks = []
    bare = lambda x: re.sub(r"[^\w'-]", "", x[0]).lower()
    for w in words + [("", None, None)]:
        if w[1] is not None: ph.append(w)
        if ph and (w[1] is None or re.search(r"[,.?!]$", w[0])):
            n = len(ph); mx = 3 if upper else 6; kk = 1 if n <= mx + 1 else -(-n // mx); sizes = [n // kk + (1 if i < n % kk else 0) for i in range(kk)]
            cuts = list(np.cumsum(sizes)[:-1])
            for ci in range(len(cuts)):
                cc = cuts[ci]; lo = cuts[ci - 1] if ci else 0; hi = cuts[ci + 1] if ci + 1 < len(cuts) else n
                if bare(ph[cc - 1]) in WEAK:
                    if cc + 1 < hi and cc + 1 - lo <= 3: cuts[ci] = cc + 1
                    elif cc - 1 > lo and bare(ph[cc - 2]) not in WEAK: cuts[ci] = cc - 1
            if len(cuts) and n - cuts[-1] == 1 and n - (cuts[-2] if len(cuts) > 1 else 0) <= 4: cuts = cuts[:-1]
            b = [0] + cuts + [n]; chunks += [ph[b[i]:b[i + 1]] for i in range(len(b) - 1)]; ph = []
    caps = []
    for i, ch in enumerate(chunks):
        txt = " ".join(x[0] for x in ch); txt = (txt.upper() if upper else txt).rstrip(",.").replace("!", "")
        s = ch[0][1] - 0.04 if i else t0
        e = chunks[i + 1][0][1] - 0.04 if i + 1 < len(chunks) else line_end
        caps.append((s, e, txt))
    return caps

# ---------------- end card ----------------
def end_card():
    FEND = ImageFont.truetype(ANTON, 210); FB = f"{ROOT}/assets/fonts/Poppins-Bold.ttf"
    F2, F3 = ImageFont.truetype(FB, 54), ImageFont.truetype(FB, 40)
    im = Image.new("RGB", (W, H), (16, 17, 22)); d = ImageDraw.Draw(im); cx, cy = W // 2, 560
    d.rounded_rectangle((cx - 90, cy - 90, cx + 90, cy + 90), radius=36, fill=(30, 32, 40), outline=(60, 62, 72), width=3)
    d.line((cx - 48, cy - 62, cx + 48, cy - 62), fill=(236, 232, 222), width=10); d.line((cx - 48, cy + 62, cx + 48, cy + 62), fill=(236, 232, 222), width=10)
    d.polygon([(cx - 40, cy - 54), (cx + 40, cy - 54), (cx + 6, cy - 2), (cx + 40, cy + 54), (cx - 40, cy + 54), (cx - 6, cy - 2)], outline=(236, 232, 222), width=8)
    d.polygon([(cx - 26, cy + 48), (cx + 26, cy + 48), (cx, cy + 18)], fill=(247, 162, 58))
    b1 = d.textbbox((0, 0), "DM", font=FEND); b2 = d.textbbox((0, 0), "BETA", font=FEND)
    w1, w2 = b1[2] - b1[0], b2[2] - b2[0]; gap = 40; padx = 34; x = (W - (w1 + gap + w2 + 2 * padx)) // 2; y = 760
    d.text((x - b1[0], y - b1[1]), "DM", font=FEND, fill=(255, 255, 255)); bx = x + w1 + gap; bh = b2[3] - b2[1]
    d.rounded_rectangle((bx, y - 26, bx + w2 + 2 * padx, y + bh + 26), radius=30, fill=(247, 162, 58))
    d.text((bx + padx - b2[0], y - b2[1]), "BETA", font=FEND, fill=(16, 17, 22))
    for txt, f, yy, col in [("for a private beta code", F2, 1080, (236, 232, 222)), ("Cash Scenario Simulator", F3, 1180, (160, 166, 178))]:
        b = d.textbbox((0, 0), txt, font=f); d.text(((W - (b[2] - b[0])) // 2 - b[0], yy), txt, font=f, fill=col)
    return cv2.cvtColor(np.array(im), cv2.COLOR_RGB2BGR)

def cover(base, lines, out, y=560, size=100):
    FXB = f"{ROOT}/assets/fonts/Poppins-Bold.ttf"
    f = ImageFont.truetype(FXB, size); d = ImageDraw.Draw(Image.new("RGBA", (10, 10)))
    lh = int(size * 1.12); tw = max(d.textbbox((0, 0), l, font=f)[2] for l in lines)
    im = Image.new("RGBA", (tw + 88, lh * len(lines) + 44), (0, 0, 0, 0)); dd = ImageDraw.Draw(im)
    dd.rounded_rectangle((0, 0, im.width - 1, im.height - 1), radius=28, fill=(255, 229, 0, 255))
    for i, l in enumerate(lines):
        b = dd.textbbox((0, 0), l, font=f); dd.text(((im.width - (b[2] - b[0])) // 2 - b[0], 22 + i * lh - int(size * 0.08)), l, font=f, fill=(10, 10, 10))
    c = base.copy(); paste(c, rgba(im), W / 2, y); cv2.imwrite(out, c)

# ---------------- the reel ----------------
class Reel:
    """segments: dict(key, voice, text, speed=1, vis=callable(t_local, mouth)->frame, speaker=True if vis is a talking face,
       gap=0.25, label=None|(lines), label_y=880, caption=True)"""
    def __init__(self, name, segs, mood_flip=None, tail=2.2):
        self.name = name; self.segs = segs
        self.A = tts(name, segs)
        t = 0.05; self.T = {}
        for s in segs:
            s["gap"] = 0.35 if s.get("beat") else min(s.get("gap", 0.1), 0.12)
            self.T[s["key"]] = t; t += len(self.A[s["key"]]) / SR + s["gap"]
        self.END = t; self.TOTAL = t + tail; self.N = int(round(self.TOTAL * FPS))
        self.mood_flip = self.T[mood_flip] if mood_flip else None
        self.open = {}
        for s in segs:                                    # mouth openness per talking segment
            self.open[s["key"]] = mouth_curve(self.A[s["key"]], FPS, SR)
        self.caps = []
        for i, s in enumerate(segs):
            if s.get("caption", True):
                k = s["key"]; le = self.T[k] + len(self.A[k]) / SR + s.get("gap", 0.25) - 0.02
                col = tuple(s.get("cap_color", (255, 255, 255)))
                self.caps += [(a_, b_, t_, col) for a_, b_, t_ in captions_for(s["text"], self.A[k], self.T[k], le, upper=(CAP_STYLE != "pixel"))]
        self.pills = {(c[2], c[3]): (pixel_sub(c[2], c[3]) if CAP_STYLE == "pixel" else pill(c[2])) for c in self.caps}
        self.huds = {s["key"]: hud(s["hud"]) for s in segs if s.get("hud")}
        self.labels = {s["key"]: pixel_label(s["label"], s.get("label_size", 92), s.get("label_color", (255, 255, 255))) for s in segs if s.get("label")}
        self.endcard = end_card()
    def seg_at(self, t):
        cur = None
        for s in self.segs:
            if t >= self.T[s["key"]]: cur = s
        return cur
    def frame(self, i):
        t = i / FPS
        if t >= self.END: return self.endcard.copy(), self.endcard
        s = self.seg_at(t) or self.segs[0]; k = s["key"]; tl = t - self.T[k]
        op = self.open[k]; o = op[int(tl * FPS)] if s.get("speaker") and int(tl * FPS) < len(op) else 0.0
        base = s["vis"](tl, o)
        if s.get("pixel"): base = pixelate(base, s.get("pixel_factor", 4))
        if s.get("grade", True): base = grade(base)
        f = base.copy()
        if k in self.labels: paste(f, self.labels[k], W / 2, s.get("label_y", 880) + (190 if k in self.huds and s.get("label_y", 880) < 500 else 0), min(1.0, tl / 0.08))
        if k in self.huds: paste(f, self.huds[k], self.huds[k][0].shape[1] / 2 + 36, self.huds[k][0].shape[0] / 2 + 150)
        for a, b, txt, col in self.caps:
            if a <= t < b: paste(f, self.pills[(txt, col)], W / 2, 1530 if CAP_STYLE == "pixel" else 1560, 1.0 if CAP_STYLE == "pixel" else min(1.0, (t - a) / 0.08)); break
        return f, base
    def audio(self):
        n = int(self.TOTAL * SR); tt = np.arange(n) / SR; vo = np.zeros(n, np.float32)
        for s in self.segs:
            a = self.A[s["key"]]; st = int(self.T[s["key"]] * SR); vo[st:st + len(a)] += a * 1.1
        rng = np.random.default_rng(1); beat = 60 / 92; bed = np.zeros(n, np.float32)
        kt = np.arange(0, 0.35, 1 / SR); kick = (np.sin(2 * np.pi * (50 + 90 * np.exp(-kt * 30)) * kt) * np.exp(-kt * 9)).astype(np.float32)
        ht = np.arange(0, 0.05, 1 / SR); flip = self.mood_flip if self.mood_flip is not None else 1e9
        for b in np.arange(0, self.TOTAL, beat):
            st = int(b * SR); e = min(n, st + len(kick)); bed[st:e] += kick[:e - st] * 0.55
            for off, g in ([(beat / 2, 0.05)] + ([(beat / 4, 0.03), (3 * beat / 4, 0.03)] if b >= flip else [])):
                h = int((b + off) * SR)
                if h + len(ht) < n: bed[h:h + len(ht)] += (rng.normal(0, 1, len(ht)) * np.exp(-ht * 90)).astype(np.float32) * g
        x = np.clip((tt - flip) / 0.6, 0, 1)
        dark = np.sin(2 * np.pi * 55 * tt) + 0.5 * np.sin(2 * np.pi * 82.4 * tt) + 0.35 * np.sin(2 * np.pi * 65.4 * tt)
        bright = np.sin(2 * np.pi * 55 * tt) + 0.5 * np.sin(2 * np.pi * 82.4 * tt) + 0.35 * np.sin(2 * np.pi * 69.3 * tt) + 0.2 * np.sin(2 * np.pi * 110 * tt)
        bed += (((1 - x) * dark + x * bright) * (0.55 + 0.45 * np.sin(2 * np.pi * tt / beat / 2) ** 2) * 0.07).astype(np.float32)
        if MUSIC and os.path.exists(MUSIC):
            m = subprocess.run(["ffmpeg", "-loglevel", "error", "-ss", str(MUSIC_START), "-i", MUSIC, "-ac", "1", "-ar", str(SR), "-f", "f32le", "-"], capture_output=True, check=True).stdout
            m = np.frombuffer(m, np.float32)
            if len(m) < n: m = np.tile(m, n // max(1, len(m)) + 1)
            m = m[:n] / (np.max(np.abs(m[:n])) + 1e-9)
            bed = (m * 0.11 * np.clip(tt / 0.15, 0, 1)).astype(np.float32)        # instant start, no slow fade-in
        venv = np.convolve(np.abs(vo), np.ones(int(0.12 * SR)) / int(0.12 * SR), mode="same")
        mix = vo + bed * (1 - 0.55 * np.clip(venv / 0.03, 0, 1)).astype(np.float32) * 0.42 * np.clip((self.TOTAL - tt) / 0.8, 0, 1).astype(np.float32)
        d = f"{V3}/{self.name}/out"; os.makedirs(d, exist_ok=True)
        sf.write(f"{d}/mix_raw.wav", mix / np.max(np.abs(mix)) * 0.95, SR)
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", f"{d}/mix_raw.wav", "-af", "highpass=f=40,loudnorm=I=-14:TP=-1.5:LRA=9", "-ar", "48000", f"{d}/mix.wav"], check=True)
    def sheet(self):
        d = f"{V3}/{self.name}/out"; os.makedirs(d, exist_ok=True)
        ts = [self.T[s["key"]] + min(0.9, len(self.A[s["key"]]) / SR * 0.5) for s in self.segs] + [self.END + 0.5]
        ims = [self.frame(int(x * FPS))[0][::5, ::5] for x in ts]
        while len(ims) % 6: ims.append(np.zeros_like(ims[0]))
        rows = [np.hstack(ims[i:i + 6]) for i in range(0, len(ims), 6)]
        cv2.imwrite(f"{d}/sheet.jpg", np.vstack(rows)); return f"{d}/sheet.jpg"
    def render(self):
        d = f"{V3}/{self.name}/out"; self.audio()
        def ff(o): return subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
                                            "-i", f"{d}/mix.wav", "-c:v", "libx264", "-crf", "19", "-preset", "veryfast", "-pix_fmt", "yuv420p",
                                            "-c:a", "aac", "-b:a", "192k", "-shortest", "-movflags", "+faststart", o], stdin=subprocess.PIPE)
        p1, p2 = ff(f"{d}/cap.tmp.mp4"), ff(f"{d}/clean.tmp.mp4")
        for i in range(self.N):
            a, b = self.frame(i); p1.stdin.write(a.tobytes()); p2.stdin.write(b.tobytes())
        p1.stdin.close(); p2.stdin.close(); assert p1.wait() == 0 and p2.wait() == 0
        os.replace(f"{d}/cap.tmp.mp4", f"{d}/{self.name}.mp4"); os.replace(f"{d}/clean.tmp.mp4", f"{d}/{self.name}_clean.mp4")
        def ts(x): return f"{int(x // 3600):02d}:{int(x % 3600 // 60):02d}:{int(x % 60):02d},{int(round((x % 1) * 1000)) % 1000:03d}"
        with open(f"{d}/{self.name}.srt", "w") as f:
            for n_, (a, b, txt, _c) in enumerate(self.caps, 1): f.write(f"{n_}\n{ts(a)} --> {ts(b)}\n{txt}\n\n")
        print(self.name, round(self.TOTAL, 1), "s")
