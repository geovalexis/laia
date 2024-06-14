import os

import requests
from dotenv import load_dotenv

load_dotenv()


def upload_audio(audio_path):
    if not os.path.isfile(audio_path):
        print(f"File not found: {audio_path}")
        return

    # Create form data
    with open(audio_path, "rb") as file:
        form_data = {"audio": file}
        headers = {"X-API-KEY": os.getenv("JQBROTON_API_KEY")}

        try:
            # Send POST request
            response = requests.post(
                "https://b0fd-139-47-125-37.ngrok-free.app/generate",
                files=form_data,
                headers=headers,
            )
            response.raise_for_status()

            # Save the response blob as a file
            result_file = "result_video.mp4"
            with open(result_file, "wb") as video_file:
                video_file.write(response.content)
            print(f"File uploaded successfully. Video saved as '{result_file}'")

        except requests.exceptions.RequestException as e:
            print(f"Error uploading file: {e}")


if __name__ == "__main__":
    upload_audio("audio.mp3")
