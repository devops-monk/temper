# Images

The **Images** page makes pictures from a prompt, or changes a picture you have, with diffusion models that
run on your computer: on an NVIDIA GPU, on a Mac's GPU (Metal), or on the CPU (slowly).

## Setting it up

The first time, the page asks to install image generation. Diffusion models need PyTorch and diffusers, so
they go into their own environment (about 1 GB), apart from the rest of Temper. Then it downloads the model
you pick:

| Model | Download | Licence | Good for |
|---|---|---|---|
| LCM Dreamshaper v7 (the default) | 4.3 GB | MIT | 512-pixel images in four steps; about 2 seconds each on an M1 Pro |
| SD-Turbo | 2.6 GB | Stability AI non-commercial | One step, the fastest; non-commercial use only |
| SSD-1B | 4.5 GB | Apache-2.0 | 1024-pixel images in the SDXL style; slower, best with a GPU |

Only the files a model needs are downloaded (no safety-checker weights, ONNX copies or duplicate
checkpoints). The models live in the same folder as your other models, so Settings → Storage counts them.

From a terminal: `temper image --install` (or `--install --model segmind/SSD-1B`).

## Making pictures

Describe the picture, pick a shape (square, landscape, portrait or wide) and how many, and press
**Generate** (or ⌘/Ctrl-Enter). Each picture is saved as a PNG in `images/` in Temper's folder, with the
settings that made it. Generating is a job, so it carries on if you leave the page.

**More settings** has the model, what to leave out (a negative prompt), steps, guidance and the seed. Empty
boxes use the model's own defaults. The same seed and settings make the same picture again.

Click a picture in the gallery to see it larger with its settings, then:

- **Use these settings** fills in the prompt, seed and the rest, to make variations.
- **Change it** starts the next picture from this one. Describe the change; the **Change** slider sets how
  far it may go from the original (lower keeps more of it). *From a picture…* does the same with a file of
  yours.
- **Show in folder** and **Delete**.

## LoRAs

A LoRA adds a style or a subject to a model. Give the path to a `.safetensors` LoRA made for the model's
family (Stable Diffusion 1.5 for LCM Dreamshaper, 2.1 for SD-Turbo, SDXL for SSD-1B) and set its strength.
A LoRA made for another family fails with an error.

## From the command line

```bash
temper image "a lighthouse at dawn, oil painting"            # prints each picture's seed and path
temper image "a red fox" -n 4 --size 768x512 --seed 7 -o ~/Pictures
temper image "make it winter" --from photo.png --strength 0.5
temper image "a portrait" --lora my-style.safetensors
temper image --models                                        # * marks the one in use
```

## From your code

Temper answers OpenAI's image API at `/v1/images/generations`, with the pictures as base64 PNGs:

```python
from openai import OpenAI
import base64

client = OpenAI(base_url="http://127.0.0.1:8765/v1", api_key="<your key>")
out = client.images.generate(prompt="a watercolor fox", n=1, size="512x512")
open("fox.png", "wb").write(base64.b64decode(out.data[0].b64_json))
```

`model` can name one of the models above; anything else uses the one picked on the Images page.

## Not yet

ControlNet (steering a picture with a pose, edges or a depth map) and GGUF image models through
stable-diffusion.cpp aren't in Temper yet.
