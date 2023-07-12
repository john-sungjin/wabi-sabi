import os
import struct
import sys

import torch
from transformers import PreTrainedTokenizerFast

# to allow model import
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from model import WSConfig

script_dir = os.path.dirname(os.path.abspath(__file__))


# ref: https://github.com/openai/gpt-2/blob/master/src/encoder.py
def bytes_to_unicode():
    """
    Returns list of utf-8 byte and a corresponding list of unicode strings.
    The reversible bpe codes work on unicode strings.
    This means you need a large # of unicode characters in your vocab if you want to avoid UNKs.
    When you're at something like a 10B token dataset you end up needing around 5K for decent coverage.
    This is a signficant percentage of your normal, say, 32K bpe vocab.
    To avoid that, we want lookup tables between utf-8 bytes and unicode strings.
    And avoids mapping to whitespace/control characters the bpe code barfs on.
    """
    bs = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    cs = bs[:]
    n = 0
    for b in range(2**8):
        if b not in bs:
            bs.append(b)
            cs.append(2**8 + n)
            n += 1

    cs = [chr(n) for n in cs]

    return dict(zip(bs, cs))


if len(sys.argv) < 2:
    print("Usage: convert-h5-to-ggml.py dir-model")
    sys.exit(1)

# output in the same directory as the model
dir_model = sys.argv[1]

# possible data types
#   ftype == 0 -> float32
#   ftype == 1 -> float16
#
# map from ftype to string
ftype = 0
ftype_str = "f32"
fname_out = dir_model + "/ggml-model-" + ftype_str + ".bin"

tokenizer: PreTrainedTokenizerFast = PreTrainedTokenizerFast.from_pretrained(dir_model)
config = WSConfig.from_pretrained(dir_model)
hparams = config.to_dict()

fout = open(fname_out, "wb")

fout.write(struct.pack("i", 0x67676D6C))  # magic: ggml in hex
fout.write(struct.pack("i", hparams["d_model"]))
fout.write(struct.pack("i", hparams["n_heads"]))
fout.write(struct.pack("i", hparams["n_layers"]))
fout.write(struct.pack("i", hparams["vocab_size"]))
fout.write(struct.pack("i", hparams["d_head"]))
fout.write(struct.pack("i", ftype))

vocab_size = hparams["vocab_size"]

encoder = tokenizer.vocab
encoder.update(tokenizer.get_added_vocab())  # is this necessary?

byte_encoder = bytes_to_unicode()
byte_decoder = {v: k for k, v in byte_encoder.items()}

counter = 0
# sort by token id
# encoder is {token: id}
for key in sorted(encoder, key=encoder.get):
    text = ""
    for c in key:
        if c not in byte_decoder:
            raise ValueError(
                "Can't find char {} in byte_decoder. Do you have the right encoder?"
            )
        text += chr(byte_decoder[c])
    text = bytearray(text, encoding="utf-8")
    fout.write(struct.pack("i", len(text)))
    fout.write(text)
    counter += 1

# Repeat last token until vocab_size
# Shouldn't be necessary for me since the entire vocab size is filled
while counter < vocab_size:
    fout.write(struct.pack("i", len(text)))
    fout.write(text)
    counter += 1

model_name = "pytorch_model.bin"
print(f"\n* Loading: {model_name}")
model = torch.load(os.path.join(dir_model, model_name), map_location="cpu")

for name in model.keys():
    data = model[name].squeeze()
    n_dims = len(data.shape)

    # ftype == 0 -> float32, ftype == 1 -> float16
    # default type is fp32
    ftype_cur = 0
    if ftype == 1 and name[-7:] == ".weight" and n_dims > 1:
        ftype_cur = 1
    data = data.to(dtype=torch.float16 if ftype_cur == 1 else torch.float32).numpy()

    print(
        "Processing variable: " + name + " with shape: ",
        data.shape,
        "->",
        data.dtype,
    )

    # header
    str = name.encode("utf-8")
    fout.write(struct.pack("iii", n_dims, len(str), ftype_cur))
    for i in range(n_dims):
        fout.write(struct.pack("i", data.shape[n_dims - 1 - i]))
    fout.write(str)

    data.tofile(fout)

fout.close()
print("Done. Output file: " + fname_out)
