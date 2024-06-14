import logging
import os
from pathlib import Path

import requests
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)


def generate_video(audio_path):
    # Create form data
    with open(audio_path, "rb") as file:
        form_data = {"audio": file}

        try:
            # Send POST request
            response = requests.post(
                "https://b0fd-139-47-125-37.ngrok-free.app/generate",
                files=form_data,
                headers={"X-API-KEY": os.getenv("JQBROTON_API_KEY")},
            )
            response.raise_for_status()

            # Save the response blob as a file
            result_file = Path("result_video.mp4")
            with open(result_file, "wb") as video_file:
                video_file.write(response.content)
            logger.info(f"File uploaded successfully. Video saved as '{result_file}'")

            return result_file

        except requests.exceptions.RequestException as e:
            logger.error(f"Error uploading file: {e}")
            raise


if __name__ == "__main__":
    generate_video("audio.mp3")
