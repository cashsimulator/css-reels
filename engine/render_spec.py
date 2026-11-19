"""Render one video from a spec JSON.  python3 engine/render_spec.py queue/<id>/spec.json [--sheet]
Outputs next to the spec: video.mp4, video_clean.mp4, cover.jpg, captions.srt, sheet.jpg"""
import sys, os, json, shutil
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scenes import *

COLORS = {"white": (255, 255, 255), "yellow": (255, 229, 0), "red": (255, 90, 80), "green": (90, 230, 120), "orange": (255, 160, 80), "teal": (120, 230, 220)}

def vis_for(show, speaker):
    t = show["type"]
    if t == "cfo": return (lambda tl, o: cfo(o)), speaker == "cfo", True
    if t == "face":
        F = face(show["char"]); z = show.get("zoom", 1.0)
        return (lambda tl, o: F.framed(o, z)), speaker == show["char"], True
    if t == "broll":
        S = broll(show["scene"]); return (lambda tl, o: S.frame(tl)), False, True
    if t == "app":
        img = app(show["screen"]); return (lambda tl, o: img.copy()), False, False
    raise ValueError(f"unknown show type {t}")

def build(spec_path):
    spec = json.load(open(spec_path)); d = os.path.dirname(os.path.abspath(spec_path))
    segs = []
    for s in spec["segments"]:
        v, talking, gr = vis_for(s["show"], s["speaker"])
        seg = dict(key=s["key"], voice=CHARACTERS[s["speaker"]]["voice"], text=s["text"], speed=s.get("speed", 1.0),
                   vis=v, speaker=talking, grade=gr, gap=s.get("gap", 0.25), caption=s.get("caption", True))
        if s.get("label"):
            seg["label"] = s["label"]; seg["label_y"] = s.get("label_y", 320); seg["label_color"] = COLORS.get(s.get("label_color", "white"), (255, 255, 255))
        segs.append(seg)
    reel = Reel(spec["id"], segs, mood_flip=spec.get("mood_flip"))
    return spec, d, reel

if __name__ == "__main__":
    spec, d, reel = build(sys.argv[1])
    out = f"{V3}/{reel.name}/out"
    if "--sheet" in sys.argv:
        shutil.copy(reel.sheet(), f"{d}/sheet.jpg"); print(f"{d}/sheet.jpg", round(reel.TOTAL, 1)); sys.exit()
    reel.render()
    for src, dst in [(f"{out}/{reel.name}.mp4", f"{d}/video.mp4"), (f"{out}/{reel.name}_clean.mp4", f"{d}/video_clean.mp4")]:
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", src, "-c:v", "libx264", "-crf", "23", "-preset", "medium", "-maxrate", "6M",
                        "-bufsize", "12M", "-pix_fmt", "yuv420p", "-c:a", "copy", "-movflags", "+faststart", dst], check=True)
    shutil.copy(f"{out}/{reel.name}.srt", f"{d}/captions.srt")
    ch = spec.get("cover_char", "cfo")
    base = grade(cfo(0)) if ch == "cfo" else grade(face(ch).framed(0))
    cover(base, spec["cover_text"], f"{d}/cover.png", y=spec.get("cover_y", 1480))
    Image.open(f"{d}/cover.png").convert("RGB").save(f"{d}/cover.jpg", quality=90); os.remove(f"{d}/cover.png")
    shutil.copy(reel.sheet(), f"{d}/sheet.jpg")
    print("rendered", reel.name, round(reel.TOTAL, 1), "s")
