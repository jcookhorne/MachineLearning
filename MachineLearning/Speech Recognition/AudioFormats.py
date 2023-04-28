#.mp3
#/.flac
#.wav
import wave
#rb means read binary
read = wave.open("joshuaTalkingTest.wav", "rb")

print("NUmber of Channels", read.getnchannels())
print("sample width", read.getsampwidth())
print("frame rate ", read.getframerate())
print("Number of frames ", read.getnframes())
print("Parameters", read.getparams())


