import soundfile as sf
import torch
import torchaudio
from liquid_audio import LFM2AudioModel, LFM2AudioProcessor, ChatState, LFMModality
import time

# Load models
HF_REPO = "LiquidAI/LFM2.5-Audio-1.5B"

processor = LFM2AudioProcessor.from_pretrained(HF_REPO).eval()
model = LFM2AudioModel.from_pretrained(HF_REPO).eval()

# Set up inputs for the model
chat = ChatState(processor)

chat.new_turn("system")
chat.add_text("Perform TTS. Use the US female voice.")
chat.end_turn()

chat.new_turn("user")
chat.add_text(
    "LLMs vs GenAI vs DSA: Understanding the Differences. Vanshdeep you're soooo confused!")
chat.end_turn()

chat.new_turn("assistant")

# Start timing
start_time = time.time()
print("Generating reply")

# Generate text
audio_out: list[torch.Tensor] = []
for t in model.generate_sequential(**chat, max_new_tokens=512, audio_temperature=0.8, audio_top_k=64):
    if t.numel() > 1:
        audio_out.append(t)

generation_time = time.time()
print(f"Making audio from generated tokens")
print(f"Time to generate reply: {generation_time - start_time:.2f} seconds")

# Detokenize audio
audio_codes = torch.stack(audio_out[:-1], 1).unsqueeze(0)
waveform = processor.decode(audio_codes)
# torchaudio.save("tts.wav", waveform.cpu(), 24_000)
sf.write(
    "tts_female_uk.wav",
    waveform.squeeze().cpu().numpy(),
    24_000
)

end_time = time.time()
print(f"Time to make audio: {end_time - generation_time:.2f} seconds")
print(f"Total time: {end_time - start_time:.2f} seconds")
