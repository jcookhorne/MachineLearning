#.mp3
#/.flac
#.wav
import wave
#rb means read binary
obj = wave.open("../audioFiles/joshuaTalkingTest.wav", "rb")

print("NUmber of Channels", obj.getnchannels())
print("sample width", obj.getsampwidth())
print("frame rate ", obj.getframerate())
print("Number of frames ", obj.getnframes())
print("Parameters", obj.getparams())

t_audio = obj.getnFrames() / obj.getFrameRate()
print(t_audio)

frames = obj.readframes(-1)
print(type(frames), type(frames[0]))
print(len(frames))

obj.close()

obj_new = wave.open("../audioFiles/joshuaTalkingTest.wav", "wb")
obj_new.setnchannels(1)
obj_new.setnsampwidth(2)
obj_new.setframerate(16000.0)


obj_new.writeframes(frames)
obj_new.close()

