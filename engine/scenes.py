"""Everything a video spec can use: talking characters, moving b-roll scenes, app screens."""
import numpy as np, cv2
from engine import *

ease = BA.ease

# ---------- talking characters (mouth closed in source) ----------
CHARACTERS = {
    "cfo":   dict(desc="The CFO. Dark suit, wood office at night, calm and a little cold. Main narrator of every video.", voice="am_michael"),
    "owner": dict(desc="Coffee-shop owner, man ~30, grey hoodie, arms crossed, smug. Good for stating myths / wrong beliefs.", voice="am_puck",
                  file="owner_smug.png", mouth=(440, 463), halfw=50),
    "baker": dict(desc="Bakery owner, woman ~35, curly hair, navy apron, serious. Good for storytime in first person.", voice="bf_emma",
                  file="baker.png", mouth=(434, 522), halfw=48),
    "maya":  dict(desc="Young design-studio founder, woman ~27, black bob, glasses, mustard sweater, excited. Good for POV / first hire / growth stories.", voice="af_heart",
                  file="maya.png", mouth=(443, 630), halfw=54),
}
_faces = {}
def face(name):
    if name not in _faces:
        c = CHARACTERS[name]; _faces[name] = Face(f"{R}/{c['file']}", c["mouth"], c["halfw"], open_px=15, jaw=95)
    return _faces[name]

# ---------- moving b-roll ----------
def _scene(img): return BA.Scene(f"{R}/{img}.png")
def _built():
    sc = BA.build(R)                            # rent, payroll, suppliers, late_clients (coffee-shop owner)
    c = _scene("broll_client_leaves")
    c.move((620, 320, 800, 820), lambda t: 30 * ease(t / 2.4), lambda t: -10 * ease(t / 2.4), 30)
    c.move((330, 880, 470, 960), lambda t: 0, lambda t: -6 * ease(t / 1.2), 20)
    c.breathe((0, 200, 470, 1000), 0.006, 0.35); c.light(248, 125, 170, 0.14, 7).light(728, 228, 70, 0.2, 8)
    c.add_rain((480, 30, 864, 760), n=110, seed=9); sc["client_leaves"] = c
    st = _scene("bk_stock")
    st.move((600, 400, 864, 700), lambda t: -14 * ease(t / 1.6), lambda t: 6 * ease(t / 1.6), 30)
    st.move((270, 760, 400, 870), lambda t: 3 * np.sin(2 * np.pi * 2.4 * t), lambda t: 1.5 * np.sin(2 * np.pi * 4.8 * t), 16)
    st.breathe((30, 190, 400, 1150), 0.006, 0.35); sc["bakery_stock_delivery"] = st
    ru = _scene("bk_rush")
    ru.move((450, 650, 660, 800), lambda t: -22 * ease(t / 1.4), lambda t: 4 * ease(t / 1.4), 26)
    ru.breathe((0, 380, 450, 1536), 0.005, 0.4); ru.breathe((560, 360, 830, 900), 0.005, 0.33, 0.4)
    for i, (x, y) in enumerate([(20, 80), (95, 75), (250, 70), (320, 25), (450, 100), (570, 50), (680, 50), (810, 70)]): ru.light(x, y, 40, 0.5, i + 3)
    sc["bakery_busy_rush"] = ru
    ja = _scene("bk_jan")
    ja.move((590, 650, 720, 790), lambda t: 4 * np.sin(2 * np.pi * 0.6 * t), lambda t: 6 * np.sin(2 * np.pi * 0.6 * t), 26)
    ja.breathe((260, 550, 720, 1080), 0.007, 0.3); ja.light(270, 120, 90, 0.08, 4); sc["bakery_quiet_bills"] = ja
    hi = _scene("mx_hire")
    hi.move((310, 680, 460, 790), lambda t: 0, lambda t: 6 * np.sin(2 * np.pi * 2.0 * t) * (t < 1.6), 22)
    hi.breathe((40, 330, 330, 1350), 0.005, 0.35); hi.breathe((520, 250, 864, 1500), 0.005, 0.3, 0.5); sc["studio_handshake_hire"] = hi
    pa = _scene("mx_payroll")
    pa.move((380, 1120, 440, 1190), lambda t: 0, lambda t: 4 * max(0, np.sin(2 * np.pi * 1.5 * t)), 12)
    pa.move((720, 690, 800, 760), lambda t: 2 * np.sin(2 * np.pi * 6 * t), lambda t: 1.5 * np.sin(2 * np.pi * 5 * t), 12)
    pa.breathe((0, 520, 520, 1250), 0.006, 0.32); pa.light(745, 365, 90, 0.1, 5); sc["studio_paying_salary"] = pa
    ni = _scene("mx_night")
    ni.move((120, 500, 560, 880), lambda t: 3 * np.sin(2 * np.pi * 0.9 * t), lambda t: 2 * np.sin(2 * np.pi * 1.8 * t), 40)
    ni.breathe((0, 700, 660, 1200), 0.008, 0.5); ni.light(520, 950, 260, 0.22, 6); sc["studio_night_panic"] = ni
    return sc
BROLL_DESC = {
    "rent": "Coffee-shop owner hands a stack of cash to his landlord at the shop door.",
    "payroll": "Coffee-shop owner pays his two baristas, they hold banknotes.",
    "suppliers": "Coffee-shop owner counts cash into a delivery driver's hand, sacks of coffee, invoice.",
    "late_clients": "Coffee-shop owner alone at night with unpaid invoices, staring at his phone, rain.",
    "client_leaves": "A big client in a suit walks out of the coffee shop, owner watches worried.",
    "bakery_stock_delivery": "Bakery owner signs for a big delivery of flour sacks and boxes (stocking up, paying upfront).",
    "bakery_busy_rush": "Bakery packed with customers in December, lights, owner handing a box over the counter.",
    "bakery_quiet_bills": "Empty bakery in January, owner at a laptop with bills, hand on forehead.",
    "studio_handshake_hire": "Design-studio founder shakes hands with her first employee.",
    "studio_paying_salary": "Design-studio founder sends a payment on her laptop, calendar with circled paydays.",
    "studio_night_panic": "Design-studio founder alone at night, hands on head, laptop glowing red.",
}
_SC = None
def broll(name):
    global _SC
    if _SC is None: _SC = _built()
    return _SC[name]

# ---------- app screens (real screenshots of the demo business, numbers are fixed) ----------
APP_SCREENS = {
    "danger_head": ("Cash 60,000 / EBITDA 12M 36,000 / LOW -14,000 FEB. Profitable business that still goes negative.", P_DANGER_HEAD),
    "feb_chart":   ("Danger banner + month-end cash chart: account goes below zero in Feb, -14,000 (clients pay in 60 days).", P_FEB),
    "safe_chart":  ("Safe banner + chart: with clients paying in 30 days, lowest point is +33,000. Safe all year.", P_SAFE),
    "weekly":      ("Week-by-week view: below zero on Feb 26, lowest -15,667 on Mar 10 when wages go out.", P_WEEKLY),
    "terms_60":    ("Inputs highlighted: clients pay after 60 days, we pay suppliers after 30.", P_TERMS60),
    "terms_30":    ("Inputs: clients pay after 30, suppliers after 30.", P_TERMS30),
}
_AP = {}
def app(name):
    if name not in _AP: _AP[name] = APP_SCREENS[name][1]()
    return _AP[name]
