# using pytorch they have a text to speech model we can look at


import torch
import torchaudio
import IPython
import matplotlib.pyplot as plt


torch.random.manual_seed(0)
device = "cuda" if torch.cuda.is_available() else "cpu"

# print(torch.__version__)
# print(torchaudio.__version__)
# print(device)


'''
Text Processing
Character-based encoding
In this section, we will go through how the character-based encoding works.

Since the pre-trained Tacotron2 model expects specific set of symbol tables, the same 
functionalities is available in torchaudio. However, we will first manually
 implement the encoding to aid in understanding.

First, we define the set of symbols '_-!\'(),.:;? abcdefghijklmnopqrstuvwxyz'. Then, we will map 
the each character of the input text into the index of the corresponding symbol in the table. 
Symbols that are not in the table are ignored.

'''

# symbols = "_-!'(),.:;? abcdefghijklmnopqrstuvwxyz"
# look_up = {s: i for i, s in enumerate(symbols)}
# symbols = set(symbols)
#
#
# def text_to_sequence(text):
#     text = text.lower()
#     return [look_up[s] for s in text if s in symbols]
#
# text = "Hello world! Text to speech!"
# print(text_to_sequence(text))

# processor = torchaudio.pipelines.TACOTRON2_WAVERNN_CHAR_LJSPEECH.get_text_processor()
# processed, lengths = processor(text)
#
# print(processed)
# print(lengths)
# print([processor.tokens[i] for i in processed[0, : lengths[0]]])




'''
Phoneme-based encoding
Phoneme-based encoding is similar to character-based encoding, but it uses a symbol 
table based on phonemes and a G2P (Grapheme-to-Phoneme) model.

The detail of the G2P model is out of the scope of this tutorial, 
we will just look at what the conversion looks like.
'''
# bundle = torchaudio.pipelines.TACOTRON2_WAVERNN_PHONE_LJSPEECH
#
# processor = bundle.get_text_processor()
# with torch.inference_mode():
#     processed, lengths = processor(text)
#
# print(processed)
# print(lengths)
# print([processor.tokens[i] for i in processed[0, : lengths[0]]])


'''
Spectrogram Generation
Tacotron2 is the model we use to generate spectrogram from the encoded text.
 For the detail of the model, please refer to the paper.

It is easy to instantiate a Tacotron2 model with pretrained weights, 
however, note that the input to Tacotron2 models need to be processed by the matching text processor
'''

bundle = torchaudio.pipelines.TACOTRON2_WAVERNN_PHONE_LJSPEECH
processor = bundle.get_text_processor()
tacotron2 = bundle.get_tacotron2().to(device)

text = "Hello world! Text to speech!"

with torch.inference_mode():
    processed, lengths = processor(text)
    processed = processed.to(device)
    lengths = lengths.to(device)
    spec, _, _ = tacotron2.infer(processed, lengths)


_ = plt.imshow(spec[0].cpu().detach(), origin="lower", aspect="auto")


def plot():
    fig, ax = plt.subplots(3, 1)
    for i in range(3):
        with torch.inference_mode():
            spec, spec_lengths, _ = tacotron2.infer(processed, lengths)
        print(spec[0].shape)
        ax[i].imshow(spec[0].cpu().detach(), origin="lower", aspect="auto")


plot()
plt.show()



