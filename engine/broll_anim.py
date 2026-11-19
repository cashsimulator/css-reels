"""Locked-camera motion for the b-roll stills: moving hands/cash, breathing, flickering lights, rain, grain."""
import numpy as np, cv2

W, H = 1080, 1920
S = W / 864.0                      # source images are 864x1536
YY, XX = np.mgrid[0:H, 0:W].astype(np.float32)

def ease(x): x = np.clip(x, 0, 1); return x * x * (3 - 2 * x)

def soft_box(x0, y0, x1, y1, feather=30):
    m = np.zeros((H, W), np.float32)
    cv2.rectangle(m, (int(x0 * S), int(y0 * S)), (int(x1 * S), int(y1 * S)), 1.0, -1)
    k = int(feather * S) | 1
    return cv2.GaussianBlur(m, (0, 0), feather * S / 2.5)

def radial(cx, cy, r):
    d = np.sqrt((XX - cx * S) ** 2 + (YY - cy * S) ** 2) / (r * S)
    return np.clip(1 - d, 0, 1) ** 2

class Scene:
    def __init__(self, path):
        self.img = cv2.resize(cv2.imread(path), (W, H), interpolation=cv2.INTER_LANCZOS4).astype(np.float32)
        self.moves, self.breaths, self.lights, self.rain = [], [], [], None
    def move(self, box, fx, fy, feather=30):          # fx, fy: t -> source px offset
        self.moves.append((soft_box(*box, feather), fx, fy)); return self
    def breathe(self, box, amp, rate=0.33, phase=0.0):
        self.breaths.append((soft_box(*box, 60), box[3] * S, amp, rate, phase)); return self
    def light(self, cx, cy, r, amt, seed=0):
        self.lights.append((radial(cx, cy, r), amt, seed)); return self
    def add_rain(self, box, n=140, seed=3):
        rng = np.random.default_rng(seed); x0, y0, x1, y1 = [v * S for v in box]
        self.rain = (soft_box(*box, 10), x0, y0, x1, y1, rng.uniform(x0, x1, n), rng.uniform(y0, y1, n), rng.uniform(900, 1400, n), rng.uniform(18, 40, n))
        return self
    def frame(self, t):
        mx = np.zeros((H, W), np.float32); my = np.zeros((H, W), np.float32)
        for m, fx, fy in self.moves:
            mx += m * np.float32(fx(t) * S); my += m * np.float32(fy(t) * S)
        mapx = XX - mx; mapy = YY - my
        for m, yb, amp, rate, ph in self.breaths:
            s = amp * (0.5 - 0.5 * np.cos(2 * np.pi * (rate * t + ph)))
            mapy = mapy + m * ((YY - yb) * (1 - 1 / (1 + s)))
        f = cv2.remap(self.img, mapx.astype(np.float32), mapy.astype(np.float32), cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        for m, amt, seed in self.lights:
            fl = 1 + amt * (0.6 * np.sin(2 * np.pi * 7.3 * t + seed) * np.sin(2 * np.pi * 1.7 * t + 2 * seed) + 0.4 * np.sin(2 * np.pi * 13.1 * t + seed))
            f = f * (1 + (fl - 1) * m[..., None])
        if self.rain is not None:
            m, x0, y0, x1, y1, xs, ys, sp, ln = self.rain
            layer = np.zeros((H, W), np.float32)
            for x, y, v, l in zip(xs, ys, sp, ln):
                yy = y0 + ((y - y0 + v * t) % (y1 - y0))
                cv2.line(layer, (int(x + 0.15 * l), int(yy - l)), (int(x), int(yy)), 1.0, 2, cv2.LINE_AA)
            f = f + (layer * m)[..., None] * np.array([70, 60, 45], np.float32)
        g = np.random.default_rng(int(abs(t) * 30)).normal(0, 3.2, (H // 2, W // 2)).astype(np.float32)
        f = f + cv2.resize(g, (W, H), interpolation=cv2.INTER_NEAREST)[..., None]
        return np.clip(f, 0, 255).astype(np.uint8)

def build(src):
    sc = {}
    # Rent: cash slides from the owner's hands into the landlord's hand, then the owner's hands pull back
    r = Scene(f"{src}/broll_rent.png")
    r.move((260, 950, 490, 1080), lambda t: 38 * ease(t / 1.1), lambda t: -4 * ease(t / 1.1), 26)       # cash
    r.move((410, 960, 580, 1130), lambda t: 22 * ease(t / 1.1) - 10 * ease((t - 1.2) / 0.6), lambda t: 0, 30)  # landlord hand
    r.move((170, 950, 330, 1110), lambda t: 14 * ease(t / 0.9) - 26 * ease((t - 1.1) / 0.7), lambda t: 10 * ease((t - 1.1) / 0.7), 30)  # owner hands
    r.breathe((0, 280, 430, 1350), 0.006, 0.4)
    r.breathe((560, 120, 864, 1536), 0.004, 0.3, 0.5)
    r.light(120, 150, 170, 0.12, 1).light(580, 295, 90, 0.18, 2)
    sc["rent"] = r
    # Payroll: owner hands a bill over, baristas shuffle their bills
    p = Scene(f"{src}/broll_payroll.png")
    p.move((365, 740, 525, 850), lambda t: -34 * ease(t / 1.0) + 20 * ease((t - 1.3) / 0.6), lambda t: -6 * ease(t / 1.0), 26)
    p.move((540, 880, 650, 950), lambda t: 3 * np.sin(2 * np.pi * 1.6 * t), lambda t: 2 * np.sin(2 * np.pi * 1.6 * t + 1), 18)
    p.move((165, 680, 290, 790), lambda t: 3 * np.sin(2 * np.pi * 2.2 * t), lambda t: 4 * np.sin(2 * np.pi * 2.2 * t + 0.7), 20)
    p.move((300, 615, 425, 700), lambda t: 4 * np.sin(2 * np.pi * 1.9 * t + 1.5), lambda t: 3 * np.sin(2 * np.pi * 1.9 * t), 20)
    p.breathe((620, 280, 864, 1300), 0.005, 0.38)
    p.breathe((0, 380, 430, 1200), 0.004, 0.42, 0.3)
    p.light(612, 85, 150, 0.12, 3)
    sc["payroll"] = p
    # Suppliers: owner counts bills with quick finger flicks, invoice flutters
    s = Scene(f"{src}/broll_suppliers.png")
    s.move((550, 590, 660, 690), lambda t: 5 * np.sin(2 * np.pi * 2.6 * t), lambda t: -3 * np.abs(np.sin(2 * np.pi * 2.6 * t)), 18)
    s.move((400, 600, 510, 680), lambda t: 1.5 * np.sin(2 * np.pi * 1.1 * t), lambda t: 3 * np.sin(2 * np.pi * 3.3 * t), 16)
    s.breathe((520, 230, 864, 820), 0.006, 0.45)
    s.breathe((0, 200, 420, 860), 0.004, 0.35, 0.6)
    s.light(478, 55, 140, 0.22, 4)
    sc["suppliers"] = s
    # Late clients: rain on the window, he rubs his head, phone glow pulses
    c = Scene(f"{src}/broll_late_clients.png")
    c.add_rain((30, 60, 425, 640))
    c.move((500, 560, 600, 720), lambda t: 0, lambda t: 6 * np.sin(2 * np.pi * 0.9 * t), 24)
    c.move((330, 470, 560, 620), lambda t: 2 * np.sin(2 * np.pi * 0.45 * t), lambda t: 0, 40)
    c.breathe((20, 420, 700, 1040), 0.007, 0.3)
    c.light(435, 900, 120, 0.15, 5).light(540, 50, 160, 0.1, 6)
    sc["late_clients"] = c
    return sc
