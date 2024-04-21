from pydub import AudioSegment
from pydub.playback import play
audio = AudioSegment.from_wav("../audioFiles/output.wav")

#this will increase the volume by 6db (decibles)
audio = audio + 6

audio = audio * 2

audio = audio.fade_in(2000)
audio.export("mashup.mp3", format="mp3")

audio2 = AudioSegment.from_mp3("../audioFiles/mashup.mp3")
#this doesn't work look into this
#I would like to know why it doesnt work but later

print("done")
