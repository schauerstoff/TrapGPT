from openai import OpenAI

audio_file ="audio/Money Boy - Swagger Rap Outro [ZIKfbk6lq3E].mp3"

client = OpenAI(api_key = "sk-..." )

with open(audio_file, "rb") as file:
    transcript = client.audio.transcriptions.create(
        model="whisper-1",
        file=file,
        language="de"  
    )

print(transcript.text)
