# Troubleshooting

Start with **Settings → Diagnostics**, or `temper doctor` in a terminal. It checks the engine and its
packages, the GPU and its drivers, memory, disk space, the API port and the network, and says how
to fix whatever fails.

## The engine won't start

Open **Settings → Engine**. *Restart engine* fixes most hiccups. If it still fails, *Reinstall engine*
deletes Temper's private Python environment and installs it again. Your models, chats, datasets and runs
are kept.

## A model won't load, or is slow

- The memory check on the Models page is an estimate. If loading fails, pick a smaller quant (Q4_K_M
  rather than Q8_0) or a smaller context length.
- Long contexts cost memory as well as speed. 8K–32K is plenty for chat; agents need 32K or more.
- If another app is holding GPU memory, close it, or unload the model in Temper and load it again.

## Training runs out of memory

In order: lower `batch_size` (raise `grad_accum` to keep the same effective batch), lower
`max_seq_length`, switch LoRA to QLoRA, or choose a smaller base model. On a Mac, train a 4-bit MLX model.

## The loss doesn't go down, or the replies get worse

- If the loss stays flat, check the dataset preview: is the model seeing the replies you expect? Try a higher
  learning rate.
- If validation loss rises while training loss falls, the model is memorising. Train for fewer steps, use more
  varied data, or turn on early stopping.
- If replies come out garbled after training, the learning rate is too high or the data's chat format is wrong.

## Coding agents can't reach Temper

- If something else held port 8765 when Temper started, it is on another port. Diagnostics names the program;
  quit it and restart the engine, or copy the new address from Connect.
- Load a model first: the API answers with the model that's loaded.
- If an agent stalls, its prompts may be too long for the model's context or too slow on this computer. Use a
  larger context or a smaller model.

## Something crashed

Temper saves a report under **Settings → Privacy**, with your home folder, keys and tokens removed. Turn on
*Offer to report crashes* to open one as a GitHub issue, which you read before sending. Or
[open an issue](https://github.com/devops-monk/temper/issues) yourself, with what you did and what
Diagnostics says.
