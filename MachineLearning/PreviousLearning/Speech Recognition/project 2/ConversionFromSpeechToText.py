from api_communication import *
from scipy.io import wavfile

# use sys.argv when your using the terminal it will take the second
# variable you put in
# filename = sys.argv[1]
#
#????
wavfile.read("../audioFiles/test.wav")
filename = "../audioFiles/test.wav"

audio_url = upload(filename)
save_transcript(audio_url, filename)
