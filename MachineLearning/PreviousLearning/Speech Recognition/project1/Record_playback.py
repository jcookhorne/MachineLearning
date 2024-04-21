import pyaudio
import wave
from pydub import AudioSegment
from pydub.playback import play

FRAMES_PER_BUFFER=3200
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
p= pyaudio.PyAudio()
stream = p.open(
    format=FORMAT,
    channels=CHANNELS,
    rate=RATE,
    input=True,
    frames_per_buffer=FRAMES_PER_BUFFER
)

print("Start recording")

seconds = 10
frames = []

for i in range (0, int(RATE/FRAMES_PER_BUFFER*seconds)):
    data = stream.read(FRAMES_PER_BUFFER)
    frames.append(data)

stream.stop_stream()
stream.close()
p.terminate()

print("Stopped Recording")

obj = wave.open("../audioFiles/test.wav", "wb")
obj.setnchannels(CHANNELS)
obj.setsampwidth(p.get_sample_size(FORMAT))
obj.setframerate(RATE)
obj.writeframes(b"".join(frames))
obj.close()


song = AudioSegment.from_wav("../audioFiles/test.wav")
play(song)