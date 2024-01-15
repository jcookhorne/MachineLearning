from api_communication import *


# use sys.argv when your using the terminal it will take the second
# variable you put in
# filename = sys.argv[1]
#
filename = "../audioFiles/test.wav"

audio_url = upload(filename)
save_transcript(audio_url, filename)
