import requests
from api_secrets import apiKey
import time

upload_endpoint = 'https://api.assemblyai.com/v2/upload'
transcript_endpoint = "https://api.assemblyai.com/v2/transcript"
# filename = "./audioFiles/test.wav"
headers = {'authorization': apiKey}

def upload(filename):
    def read_file(filename2, chunk_size=5242880):
        with open(filename2, 'rb') as _file:
            while True:
                data = _file.read(chunk_size)
                if not data:
                    break
                yield data

    response = requests.post(upload_endpoint,
                             headers=headers,
                             data=read_file(filename))

    audio_url = response.json()['upload_url']
    print("done with upload")
    return audio_url
# transcribe
def transcribe(audio_url2):
    json = {"audio_url": audio_url2}
    response = requests.post(transcript_endpoint, json=json, headers=headers)
    print("done with transcription")
    job_id = response.json()['id']
    return job_id


# poll
def polling(transcript_id):
    polling_endpoint = transcript_endpoint + '/' + transcript_id
    polling_response = requests.get(polling_endpoint, headers=headers)
    return polling_response.json()


def get_transcription_result_url(audio_url2):
    transcript_id = transcribe(audio_url2)
    while True:
        data = polling(transcript_id)
        if data['status'] == 'completed':
            return data, None
        elif data['status'] == 'completed':
            return data, data['error']
        print("Waiting 5 seconds. . .")
        time.sleep(5)


# save transcript
def save_transcript(audio_url2, filename):
    data, error = get_transcription_result_url(audio_url2)
    if data:
        text_filename = filename + ".txt"
        with open(text_filename, "w") as f:
            f.write(data['text'])
        print("transcription is saved")
    elif error:
        print("ERROR!", error)
