import gc, time, textwrap, wave, csv
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont, ImageOps 
import ollama, torch, gradio as gr
from pydantic import BaseModel
from diffusers import AutoPipelineForText2Image
from piper import PiperVoice
from moviepy import ImageClip, AudioFileClip, CompositeVideoClip, concatenate_videoclips, vfx

try:
    from piper import SynthesisConfig
except ImportError:
    SynthesisConfig = None

LLM = "gemma3:4b"
IMG_MODEL = "stabilityai/sd-turbo"
OUT = Path("outputs"); OUT.mkdir(exist_ok=True)
SIZE = 768

STYLE_FUN = ", simple childlike cartoon doodle, thick uniform black outline, flat solid colors, no shading, no texture, plain simple background, funny cartoon style"
GHOST_PREFIX = "a cute cartoon ghost made of a white bedsheet with two black dot eyes and a small mouth, floating next to "
STYLE_GHOST = ", simple childlike cartoon doodle, thick uniform black outline, flat solid colors, plain simple background, a glowing candle and orange marigold flowers"
VOICES = {  # name: (piper voice file, length_scale; higher = slower)
    "Warm storyteller": ("en_US-lessac-medium", 1.0),
    "Cheerful": ("en_US-amy-medium", 0.95),
    "British narrator": ("en_GB-alba-medium", 1.0),
    "Ghostly whisper": ("en_US-ryan-medium", 1.25),
}

SYSTEM = """You turn a user's imagined future into a short, fun 4-scene cartoon film, plus one small real-life first step.
Scenes 1 to 3 tell the wish step by step in a cheerful way (the start, the main moment, a happy detail).
Scene 4 is the ghost scene. The ghost is the user's future self, who remembers the very first small step that started it all. Its narration must begin with "I remember when you" and mention the first_step in the past tense. Do NOT repeat the earlier scenes.
first_step: one small, concrete action the user can do today or this week to begin this future (under 15 words, related to their wish).
Rules:
- Use only what the user wrote. Do not invent names for businesses, people or places.
- image_prompt: under 15 words. At most ONE character, seen from far away or from behind, or just an object or place from the wish. Simple plain setting. NEVER crowds, audiences, groups, faces, hands or text. Suggest crowds with glowing lights or balloons instead.
- For scene 4, image_prompt is only a simple object or place from the wish, with no people.
- narration: 1-2 simple sentences, under 25 words, plain friendly language, speaking to the user as "you".
"""

class Scene(BaseModel):
    narration: str
    image_prompt: str

class Storyboard(BaseModel):
    title: str
    scenes: list[Scene]
    first_step: str

# ---------- 1. Storyboard (Gemma via Ollama) ----------
def make_storyboard(wish):
    resp = ollama.chat(
        model=LLM,
        messages=[{"role": "system", "content": SYSTEM}, {"role": "user", "content": wish}],
        format=Storyboard.model_json_schema(),
        options={"temperature": 0.6},
        keep_alive=0,  # unload Gemma so the GPU is free for images
    )
    sb = Storyboard.model_validate_json(resp.message.content)
    for i, s in enumerate(sb.scenes):
        print(f"scene {i+1}: {s.image_prompt} | {s.narration}")
    return sb

# ---------- 2. Images (SD-Turbo) ----------
_pipe = None
def make_images(scenes, workdir):
    global _pipe
    if _pipe is None:
        _pipe = AutoPipelineForText2Image.from_pretrained(
            IMG_MODEL, torch_dtype=torch.float16, variant="fp16")
        _pipe.enable_model_cpu_offload()
    paths = []
    for i, s in enumerate(scenes):
        g = torch.Generator("cpu").manual_seed(1234 + i)
        if i == len(scenes) - 1:
            prompt = GHOST_PREFIX + s.image_prompt + STYLE_GHOST
        else:
            prompt = s.image_prompt + STYLE_FUN
        img = _pipe(prompt, num_inference_steps=2, guidance_scale=0.0,
                    height=512, width=512, generator=g).images[0]
        p = workdir / f"scene{i}.png"
        img = ImageOps.expand(img, border=10, fill="black")
        img.save(p)
        paths.append(p)
    gc.collect(); torch.cuda.empty_cache()
    return paths

# ---------- 3. Voice (Piper) ----------
_voices = {}
def narrate(text, path, voice_name):
    model, speed = VOICES[voice_name]
    if model not in _voices:
        _voices[model] = PiperVoice.load(f"voices/{model}.onnx")
    voice = _voices[model]
    with wave.open(str(path), "wb") as wav:
        if SynthesisConfig:
            voice.synthesize_wav(text, wav, syn_config=SynthesisConfig(length_scale=speed))
        else:
            voice.synthesize_wav(text, wav)

# ---------- 4. Captions + video (MoviePy) ----------
def caption_png(text, path):
    img = Image.new("RGBA", (SIZE, SIZE), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("georgia.ttf", 28)
    except OSError:
        font = ImageFont.load_default()
    lines = textwrap.wrap(text, 42)
    h = len(lines) * 36 + 24
    d.rectangle([12, SIZE - h - 32, SIZE - 12, SIZE - 32], fill=(255, 246, 214, 240), outline=(0, 0, 0, 255), width=4)
    y = SIZE - h - 8
    for ln in lines:
        w = d.textlength(ln, font=font)
        d.text(((SIZE - w) / 2, y - 6), ln, font=font, fill=(20, 16, 10, 255))
        y += 36
    img.save(path)

def build_video(scenes, imgs, workdir, out_path, voice_name):
    clips = []
    for i, (s, img) in enumerate(zip(scenes, imgs)):
        wav = workdir / f"scene{i}.wav"
        narrate(s.narration, wav, voice_name)
        audio = AudioFileClip(str(wav))
        dur = audio.duration + 1.2
        base = (ImageClip(str(img)).with_duration(dur).resized((SIZE, SIZE))
                .resized(lambda t, d=dur: 1 + 0.08 * t / d)   # slow zoom
                .with_position("center"))
        cap_path = workdir / f"cap{i}.png"
        caption_png(s.narration, cap_path)
        cap = ImageClip(str(cap_path)).with_duration(dur)
        scene = (CompositeVideoClip([base, cap], size=(SIZE, SIZE))
                 .with_duration(dur).with_audio(audio)
                 .with_effects([vfx.FadeIn(0.6), vfx.FadeOut(0.6)]))
        clips.append(scene)
    final = concatenate_videoclips(clips)
    final.write_videofile(str(out_path), fps=24, codec="libx264",
                          audio_codec="aac", logger=None)

# ---------- 5. Comic page ----------
def _font(size):
    for name in ("comicbd.ttf", "impact.ttf", "arialbd.ttf", "DejaVuSans-Bold.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            pass
    return ImageFont.load_default(size)

def _wrap(d, text, font, max_w):
    lines, cur = [], ""
    for w in text.split():
        t = (cur + " " + w).strip()
        if d.textlength(t, font=font) <= max_w:
            cur = t
        else:
            if cur:
                lines.append(cur)
            cur = w
    if cur:
        lines.append(cur)
    return lines

def _caption(d, text, x, y, max_w, size=24, from_bottom=False):
    f, pad = _font(size), 10
    lines = _wrap(d, text.upper(), f, max_w - 2 * pad)
    lh = int(size * 1.25)
    w = max(d.textlength(l, font=f) for l in lines) + 2 * pad
    h = len(lines) * lh + 2 * pad
    if from_bottom:
        y -= h
    d.rectangle([x, y, x + w, y + h], fill=(255, 221, 64), outline="black", width=4)
    for i, l in enumerate(lines):
        d.text((x + pad, y + pad + i * lh), l, font=f, fill="black")

def _bubble(d, text, cx, top, max_w, tail_to, size=21):
    f = _font(size)
    lines = _wrap(d, text.upper(), f, max_w)
    lh = int(size * 1.25)
    tw = max(d.textlength(l, font=f) for l in lines)
    th = len(lines) * lh
    w, h = tw * 1.5 + 16, th * 1.5 + 16
    x0 = cx - w / 2
    by = top + h * 0.8
    tx, ty = tail_to
    d.polygon([(cx - 28, by), (cx + 12, by), (tx, ty)], fill="white")
    d.line([(cx - 28, by), (tx, ty), (cx + 12, by)], fill="black", width=4)
    d.ellipse([x0, top, x0 + w, top + h], fill="white", outline="black", width=4)
    for i, l in enumerate(lines):
        d.text((cx - d.textlength(l, font=f) / 2, top + (h - th) / 2 + i * lh), l, font=f, fill="black")

def comic_page(scenes, imgs, first_step, out_path):
    W, M, G, HEAD = 1200, 30, 24, 120
    PW = (W - 2 * M - G) // 2
    H = HEAD + 2 * PW + G + M + 36
    page = Image.new("RGB", (W, H), (252, 244, 214))
    d = ImageDraw.Draw(page)
    title, ft = "OFRENDA OF MAYBE", _font(58)
    tx = (W - d.textlength(title, font=ft)) / 2
    d.text((tx + 4, 20), title, font=ft, fill="black")
    d.text((tx, 16), title, font=ft, fill=(230, 90, 10))
    sub, fs = "ISSUE #1  -  A FUTURE THAT HASN'T HAPPENED YET", _font(20)
    d.text(((W - d.textlength(sub, font=fs)) / 2, 92), sub, font=fs, fill="black")
    last = len(imgs) - 1
    for i, (s, p) in enumerate(zip(scenes, imgs)):
        x = M + (i % 2) * (PW + G)
        y = HEAD + (i // 2) * (PW + G)
        im = Image.open(p).convert("RGB")
        im = im.crop((10, 10, im.width - 10, im.height - 10)).resize((PW, PW))
        page.paste(im, (x, y))
        d.rectangle([x, y, x + PW, y + PW], outline="black", width=7)
        if i < last:
            _caption(d, s.narration, x + 14, y + 14, PW - 28)
        else:
            _bubble(d, s.narration, x + PW // 2, y + 16, 330, (x + PW // 2 - 10, y + int(PW * 0.62)))
            _caption(d, "To be continued... Your first step: " + first_step,
                     x + 14, y + PW - 14, PW - 28, size=22, from_bottom=True)
    foot, ff = "Made on this laptop with open models: Gemma, SD-Turbo, Piper", _font(18)
    d.text(((W - d.textlength(foot, font=ff)) / 2, H - 30), foot, font=ff, fill="black")
    page.save(out_path)
    return out_path

# ---------- Gradio app ----------
def generate(wish, voice_name, progress=gr.Progress()):
    if not wish or not wish.strip():
        raise gr.Error("Describe a future you hope for first.")
    run = OUT / time.strftime("%Y%m%d-%H%M%S")
    run.mkdir(parents=True)
    progress(0.05, desc="The ghost is listening...")
    sb = make_storyboard(wish)
    scenes = sb.scenes[:4]
    progress(0.25, desc="Painting your scenes...")
    imgs = make_images(scenes, run)
    comic = comic_page(scenes, imgs, sb.first_step, run / "comic.png")
    progress(0.7, desc="Recording the narration and stitching the film...")
    out = run / "ofrenda.mp4"
    build_video(scenes, imgs, run, out, voice_name)
    return str(out), str(comic), "🕯️ " + sb.first_step
WEB = ('<svg viewBox="0 0 200 200" width="170" height="170"><g stroke="#d9d4e8" stroke-width="1.2" fill="none" opacity=".7">'
       '<path d="M0 0 L200 0 M0 0 L190 70 M0 0 L150 150 M0 0 L70 190 M0 0 L0 200"/>'
       '<path d="M40 0 Q35 35 0 40 M80 0 Q70 70 0 80 M120 0 Q105 105 0 120 M160 0 Q140 140 0 160"/></g></svg>')

DECOR_HTML = (
    '<div class="web tl">' + WEB + '</div><div class="web tr">' + WEB + '</div>'
    '<div class="web bl">' + WEB + '</div><div class="web br">' + WEB + '</div>'
    '<div class="spider"><div class="thread"></div><span>🕷️</span></div>'
    '<span class="ghost g1">👻</span><span class="ghost g2">👻</span><span class="ghost g3">👻</span>'
)

HEADER_HTML = (
    '<div class="hdr"><div class="pumpkin">🎃</div>'
    '<h1 class="ttl">Ofrenda of Maybe</h1>'
    '<p class="sub">Whisper a future you hope for. A friendly ghost from that future will show it to you.</p>'
    '<div class="bones">💀 🦴 🕯️ 🦴 💀</div></div>'
)

FOOTER_HTML = '<div class="foot">🦇 Gemma · SD-Turbo · Piper · running on this laptop, no cloud 🦇 💀</div>'

BASE = Path(__file__).parent
CSS = (BASE / "static" / "style.css").read_text(encoding="utf-8")
JS = (BASE / "static" / "scare.js").read_text(encoding="utf-8")

THEME = gr.themes.Base(primary_hue="orange", neutral_hue="slate")

EXAMPLES = [
    ["I am a famous singer performing on a big stage in Los Angeles, and I drive a Mercedes."],
    ["I have a big garden full of sunflowers and a small cottage in the hills."],
    ["I run a small bakery and it is famous for its pastries."],
]

FEEDBACK_FILE = Path("feedback.csv")

def save_feedback(rating, comment):
    comment = (comment or "").strip()
    if not rating and not comment:
        return "👻 Pick a mood or leave a note first.", comment
    new = not FEEDBACK_FILE.exists()
    with open(FEEDBACK_FILE, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["time", "rating", "comment"])
        w.writerow([time.strftime("%Y-%m-%d %H:%M:%S"), rating or "", comment])
    return "🕯️ Thank you! Your note was saved on this laptop.", ""

with gr.Blocks(title="🎃 Ofrenda of Maybe") as demo:
    gr.HTML(DECOR_HTML)
    gr.HTML(HEADER_HTML)
    with gr.Row():
        with gr.Column():
            wish = gr.Textbox(label="🕯️ Whisper your future to the ghost...", lines=5, elem_id="wish",
                              placeholder="I run a small bakery and my family visits every Sunday...")
            voice = gr.Dropdown(list(VOICES), value="Warm storyteller", label="🎙️ Narrator voice")
            go = gr.Button("🔮 Summon my film", variant="primary", elem_id="go")
            gr.Examples(examples=EXAMPLES, inputs=[wish], label="👻 Need an idea? Click one")
            with gr.Accordion("⚙️ Settings", open=False):
                gr.Checkbox(value=True, label="👻 Jump scares while I type", elem_id="scare_toggle", interactive=True)
                gr.Checkbox(value=True, label="🔊 Scream sound", elem_id="sound_toggle", interactive=True)
        with gr.Column():
            with gr.Tabs():
                with gr.Tab("📖 Comic page"):
                    comic = gr.Image(label="Your comic", elem_id="comic", type="filepath")
                with gr.Tab("🎬 Film"):
                    film = gr.Video(label="Your ofrenda", elem_id="film")
            offering = gr.Textbox(label="🕯️ The ghost's offering: your first small step",
                                  lines=2, elem_id="offering", interactive=False)
            with gr.Accordion("💌 Leave a note for the ghost", open=True):
                rating = gr.Radio(["😍 Loved it", "🙂 Good", "😐 Meh", "👻 Too spooky"],
                                  label="How did it feel?", elem_id="rating")
                note = gr.Textbox(label="Anything to improve?", lines=2, elem_id="note",
                                  placeholder="A scene that looked odd? An idea for the ghost?")
                send = gr.Button("🕯️ Send to the ghost", elem_id="send")
                status = gr.Markdown("")
                gr.HTML('<div class="issues">Found a bug or have an idea? '
                        '<a href="https://github.com/Ankita562/ofrenda-of-maybe/issues" target="_blank">Open an issue on GitHub</a></div>')
    gr.HTML(FOOTER_HTML)
    go.click(generate, [wish, voice], [film, comic, offering])
    send.click(save_feedback, [rating, note], [status, note])
    demo.load(None, None, None, js=JS)

if __name__ == "__main__":
    demo.launch(allowed_paths=[str(OUT.resolve())], theme=THEME, css=CSS)