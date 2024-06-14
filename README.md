## 💫 Overview

Leverage an AI model that assesses patient-reported symptoms and medical history during telehealth consultations to accurately determine case severity and direct patients to the appropriate department or level of care, saving time and enhancing the safety and efficiency of remote healthcare.

## 📦️ Architecture

We created a chatbot with an avatar in a GitHub page. This integrates an Azure function and Llamaindex with state-of-the-art AI models: 
* Llumalabs
* SpeechRecognition
* LLama3,
* SadTalker
* ElevenLabs
* Stable diffusion3​.

<p align="center">
   <img width="886" alt="image" src="https://github.com/geovalexis/laia/assets/56429448/5fab0649-a6eb-460f-a342-aed3f80f2c16">
</p>

## 🎯️ Getting started

1. Make sure you have python v3.11 installed on your machine. You can download it from [here](https://www.python.org/downloads/).

2. Install dependencies:
   ```
   $ pip install -r requirements.txt
   ```

3. Fill in environment variables in the `.env` file. You can copy the `.env.example` file and rename it to `.env`.

4. Run the tool by the following command:
   ```
   $ python main.py
   ```
