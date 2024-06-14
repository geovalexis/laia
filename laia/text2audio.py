from os import getenv
from pathlib import Path

from dotenv import load_dotenv
from elevenlabs import save
from elevenlabs.client import ElevenLabs

load_dotenv()


client = ElevenLabs(
    api_key=getenv("ELEVENLABS_API_KEY"),
)


def generate_audio(
    text: str, voice: str = "Glinda", model: str = "eleven_multilingual_v1"
) -> Path:
    audio = client.generate(
        text=text,
        voice=voice,
        model=model,
    )
    audio_path = Path(f"/tmp/audio.mp3")
    save(audio, str(audio_path))
    return audio_path


if __name__ == "__main__":
    text = "Hello, how are you?"
    audio_path = generate_audio(text)
    print(f"Audio saved as '{audio_path}'")
