# css-reels: automatic Reels for Cash Scenario Simulator

This repo makes short 9:16 Reels for the Instagram account **@cashscenariosimulator** (Instagram account id `17841435392255864`, posted through the **Windsor.ai for Social Media** connector, connector `instagram`) and posts them **on request**. Each morning the owner tells Claude (in the setup chat) how many to post, the topic(s) and the times. Claude builds every video right away, pushes them here, and creates one scheduled POST run per video at the exact time asked. No previews are sent.

Every scheduled run starts in a fresh session. Clone this repo, `cd` into it, run `bash setup.sh`, then follow the section for your run type.

```
engine/        video engine (render_spec.py is the entry point, scenes.py lists everything you can use)
assets/        characters, scenes, app screenshots, fonts
ideas.md       the idea queue the owner fills in
queue/<id>/    one folder per video: spec.json, status.json, video.mp4, cover.jpg, captions.srt, sheet.jpg
```

Public file links (Instagram downloads from these):
`https://raw.githubusercontent.com/cashsimulator/css-reels/main/queue/<id>/video.mp4` and `.../cover.jpg`

---

## BUILD (done in the setup chat when the owner asks, e.g. "post 3 tomorrow about X")

Goal: one finished video for the next slot, pushed to the repo, and a preview sent to the owner.

1. Work out the slot: the next 13:00 or 19:00 (Africa/Casablanca) that has no `queue/*/status.json` with that `slot`. If one already exists with status `ready`, stop: nothing to do.
2. Pick the idea: the first line under **Next up** in `ideas.md`. If the list is empty, write a new idea from **Themes** that is not close to anything under **Done**.
3. Write `queue/<id>/spec.json` (id = `YYYYMMDD-HHMM`, e.g. `20261120-1300`). Use the format below and only names that exist in `engine/scenes.py` (`CHARACTERS`, `BROLL_DESC`, `APP_SCREENS`).
4. Check the frames: `python3 engine/render_spec.py queue/<id>/spec.json --sheet` and look at `queue/<id>/sheet.jpg`. Fix the spec if a caption, label or scene looks wrong.
5. Render: `python3 engine/render_spec.py queue/<id>/spec.json`. The video must be 20 to 45 seconds.
6. Write `queue/<id>/status.json`: `{"slot": "YYYY-MM-DD HH:MM", "status": "ready", "title": "..."}`
7. Move the idea line from **Next up** to **Done** as `- <id>: <title>`.
8. Commit and push (`git add -A && git commit -m "build <id>" && git -c push.negotiate=true push`). Do not commit `video_clean.mp4` (it is gitignored).
9. Create one scheduled task per video (run once, at the exact slot time, Africa/Casablanca) whose prompt is the POST run below for that `<id>`. No preview message.

## POST run (at each slot)

1. Open `queue/<id>/status.json` for the id named in the task prompt. If none, or its status is not `ready`, stop and say why.
2. Check both public links return HTTP 200 (`curl -sI`). If not, wait 2 minutes and check once more; if still not, set status `failed` with a reason, push, and tell the owner.
3. Publish with Windsor `execute_action`: connector `instagram`, action `create_video_post`, account `17841435392255864`, params `video_url`, `cover_url`, `caption` (from spec), `share_to_feed: true`.
4. Post the first comment with `create_comment` on the returned media id, message = `first_comment` from the spec.
5. Update status.json: `"status": "posted"`, `"media_id"`, `"posted_at"`. Commit and push.
6. Housekeeping: for queue folders posted more than 7 days ago, `git rm` their `video.mp4` (keep spec, cover, status). Commit and push.
7. End with one line: posted (title + time), or why not.

Never post anything that is not a `ready` item from this queue. Never post twice for one slot.

---

## Spec format

```json
{
 "id": "20261120-1300",
 "title": "short internal title",
 "cover_char": "owner",                       // cfo | owner | baker | maya
 "cover_text": ["Your best client", "could sink you"],   // 2-3 short lines, max ~16 chars each
 "mood_flip": "k6",                           // segment where the app/fix appears (music brightens)
 "segments": [
  {"key": "k1", "speaker": "cfo", "text": "...", "show": {"type": "cfo"}, "gap": 0.3},
  {"key": "k2", "speaker": "owner", "text": "...", "show": {"type": "face", "char": "owner"}, "label": ["myth #1"], "label_color": "red"},
  {"key": "k3", "speaker": "cfo", "text": "...", "show": {"type": "broll", "scene": "client_leaves"}},
  {"key": "k4", "speaker": "cfo", "text": "...", "show": {"type": "app", "screen": "weekly"}},
  {"key": "k5", "speaker": "cfo", "text": "It's in private beta. DM me BETA for a code.", "speed": 0.97, "show": {"type": "cfo"}, "gap": 0.35}
 ],
 "caption": "...",
 "first_comment": "..."
}
```
- `speaker` = whose voice says the line. A face only moves its mouth when the speaker matches the face shown.
- `show.type`: `cfo`, `face` (char + optional `zoom` 1.0 to 1.3), `broll` (scene), `app` (screen).
- `label`: optional pixel-font word(s) at the top. Colors: white, yellow, red, green, orange, teal.
- `caption: false` on a segment hides subtitles for it (use when a label already says the same words).

## Content rules (the owner's style; follow them)

- Two formats: (a) CFO explainer/story (the default), (b) skit with no narrator, built like this: POV hook about a recognisable person -> the product's line said out loud -> a challenge -> a reversal line + the app screen as visual proof -> walk-off -> family/colleague reactions as the punchline -> product end card. The product is part of the story from the start.
- New characters and settings are in assets/library (read its README). Never use real people's likeness or real company logos.

- Story first, product second. Hook in the first line (a surprising claim, a quote, a POV). 25 to 40 seconds.
- Each video has its own angle and story; do not repeat a hook or structure used under **Done**.
- The CFO is the recurring narrator and always closes with the DM BETA line. Other characters can tell their own story.
- Every number about money must come from an app screen description in `APP_SCREENS`. Do not invent figures or statistics.
- No camera zooms or moves are added by the engine; keep it that way. No sound effects; music only.
- B-roll must match what is said. Use 2 to 4 b-roll scenes, 1 to 3 app screens.
- Caption: plain, human, no em dashes, no hype words. Short hook line, the story in a few lines, one easy question to comment on (with 👇), one line about Cash Scenario Simulator, "Private beta is open. DM me BETA for a code.", then 6 to 8 hashtags. English only, written for the US and Europe.
- first_comment: one useful extra tip plus "DM me BETA".
