# Getting started

## Install

Download Temper for your system from the [latest release](https://github.com/devops-monk/temper/releases/latest),
or install it from a terminal:

```bash
curl -fsSL https://temper.devops-monk.com/install.sh | sh       # macOS and Linux
irm https://temper.devops-monk.com/install.ps1 | iex             # Windows PowerShell
```

You need macOS 11 or later, Windows 10 or later, or a recent 64-bit Linux. Chat works on any of them.
Training needs an NVIDIA or AMD GPU, or an Apple Silicon Mac.

## First launch

On first launch Temper sets up its engine: a private Python environment kept inside its own data folder,
which touches nothing else on your computer. Pick what you want to do:

- **Chat** installs what running models needs (about 150 MB).
- **Chat and training** adds MLX on a Mac (about 600 MB more), or PyTorch and the training libraries for an NVIDIA or AMD GPU (about 6 GB).

You can add training later from Settings → Engine. A short tour then points out where everything is;
take it again any time from Settings.

## Your first chat

1. Open **Models**. *Catalog* lists models picked for local use; *Find models* searches Hugging Face.
   Each model shows whether it fits this computer's memory.
2. Download one. Downloads carry on in the background and resume if interrupted. If you're unsure,
   start with a 3–4B model in Q4_K_M.
3. Press **Load**, then open **Chat** and ask something.

Already have GGUF files? Drag them onto the window, use *Import GGUF file* on the Models page, or
double-click a `.gguf` file once Temper is installed. Temper remembers where the file is and doesn't copy it.

## Where your things live

Temper keeps its chats, datasets, runs and exports in one folder:

| System  | Folder |
|---------|--------|
| macOS   | `~/Library/Application Support/app.temper.studio` |
| Windows | `%APPDATA%\app.temper.studio` |
| Linux   | `~/.local/share/app.temper.studio` |

Downloaded models go to the standard Hugging Face cache, so other tools can use them too. Settings → Storage
shows how much space each part takes, and can move the models folder to another drive.

## Privacy

Temper sends no telemetry. It goes online only for what you ask it to do (downloading a model, searching the web,
publishing) and to check for updates, which you can turn off. Crash reports stay on your computer
unless you choose to send one.

Next: [make a dataset](datasets.md).
