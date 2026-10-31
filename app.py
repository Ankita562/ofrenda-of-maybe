import gc, time, textwrap, wave
from pathlib import Path

import ollama, torch, gradio as gr
from PIL import Image, ImageDraw, ImageFont
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

STYLE_FUN = ", colorful storybook illustration, warm golden light, cheerful, detailed"
GHOST_PREFIX = "a cute friendly glowing white ghost with big eyes floating among marigold petals, "
STYLE_GHOST = ", colorful storybook illustration, warm candlelight, magical, detailed"
VOICES = {  # name: (piper voice file, length_scale; higher = slower)
    "Warm storyteller": ("en_US-lessac-medium", 1.0),
    "Cheerful": ("en_US-amy-medium", 0.95),
    "British narrator": ("en_GB-alba-medium", 1.0),
    "Ghostly whisper": ("en_US-ryan-medium", 1.25),
}

SYSTEM = """You turn a user's imagined future into a short, fun 4-scene storybook film.
Scenes 1 to 3 tell the wish step by step in a cheerful way (the start, the main moment, a happy detail).
Scene 4 is a light ghost touch: a small friendly ghost appears in the same setting, and its narration begins with "I remember when you" and is a warm, short message about the wish.
Rules:
- Do not invent names for businesses, people or places. Use only what the user wrote.
- Stay literal to the wish. Every image_prompt must name the main subject from the wish (the bakery, the pastries, the customers, etc).
- image_prompt: under 20 words, concrete, starting with the main subject and setting. No abstract ideas. No rain or darkness.
- For scene 4's image_prompt, keep the same setting and add "with a small friendly ghost".
- narration: 1-2 simple sentences, under 25 words, plain friendly language, speaking to the user as "you".
"""

class Scene(BaseModel):
    narration: str
    image_prompt: str

class Storyboard(BaseModel):
    title: str
    scenes: list[Scene]

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
            prompt = GHOST_PREFIX + s.image_prompt.replace("with a small friendly ghost", "") + STYLE_GHOST
        else:
            prompt = s.image_prompt + STYLE_FUN
        img = _pipe(prompt, num_inference_steps=2, guidance_scale=0.0,
                    height=512, width=512, generator=g).images[0]
        p = workdir / f"scene{i}.png"
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
    d.rectangle([0, SIZE - h - 20, SIZE, SIZE - 20], fill=(10, 8, 30, 170))
    y = SIZE - h - 8
    for ln in lines:
        w = d.textlength(ln, font=font)
        d.text(((SIZE - w) / 2, y), ln, font=font, fill=(255, 214, 120, 255))
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

# ---------- Gradio app ----------
def generate(wish, voice_name, progress=gr.Progress()):
    if not wish or not wish.strip():
        raise gr.Error("Describe a future you hope for first.")
    run = OUT / time.strftime("%Y%m%d-%H%M%S")
    run.mkdir(parents=True)
    progress(0.05, desc="The ghost is listening...")
    scenes = make_storyboard(wish).scenes[:4]
    progress(0.25, desc="Painting your scenes...")
    imgs = make_images(scenes, run)
    progress(0.7, desc="Recording the narration and stitching the film...")
    out = run / "ofrenda.mp4"
    build_video(scenes, imgs, run, out, voice_name)
    return str(out)
CSS = """
.gradio-container {background: radial-gradient(circle at top, #2b1a3d 0%, #0e0a1a 70%) !important;}
h1 {color: #ffb347 !important; text-shadow: 0 0 18px rgba(255,160,40,.6);}
button.primary {background: #e8590c !important;}
"""
THEME = gr.themes.Base(primary_hue="orange", neutral_hue="slate")
demo = gr.Interface(
    fn=generate,
    inputs=[
        gr.Textbox(label="Describe a future you hope for", lines=3,
                   placeholder="I run a famous bakery and my family visits every Sunday..."),
        gr.Dropdown(list(VOICES), value="Warm storyteller", label="Narrator voice"),
    ],
    outputs=gr.Video(label="Your ofrenda"),
    title="🕯️ Ofrenda of Maybe",
    description="Describe a future you hope for. A friendly ghost from that future turns it into a short film. Runs entirely on this laptop with open models.",
    flagging_mode="never",
)

if __name__ == "__main__":
    demo.launch(allowed_paths=[str(OUT.resolve())], theme=THEME, css=CSS)