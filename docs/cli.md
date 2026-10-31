# Command line

The app's own binary takes these commands. They talk to the engine of the open app, or of `temper serve`.

| System  | Binary |
|---------|--------|
| macOS   | `/Applications/Temper.app/Contents/MacOS/temper` |
| Windows | `%LOCALAPPDATA%\Temper\temper.exe` |
| Linux   | `temper` from the .deb or .rpm (in `/usr/bin`), or the AppImage itself |

On macOS, add `alias temper=/Applications/Temper.app/Contents/MacOS/temper` to your shell profile.

## Models and chat

```bash
temper models                      # the models on this computer
temper load Qwen3-4B-GGUF          # a name, a GGUF file name or a path
temper load my-model --quant Q8_0 --context 32768
temper load my-model --estimate-only          # the memory it would take, without loading
temper load small-model --alongside --ttl 600 --alias fast   # next to chat's model, unloads after 10 idle minutes
temper ps                          # what's loaded: role, engine, context, idle time, when it unloads (--json)
temper unload fast                 # or --all
temper chat "Explain a monad"      # ask once and print the reply
cat notes.md | temper chat "Summarise this"
temper chat                        # chat until /bye; /clear starts over
```

`temper load` refuses a model that doesn't look like it fits; `-y` tries anyway. `--gpu-layers N` keeps some
layers on the CPU (0 runs it all there). An `--alias` is the name API requests use for the model.

## Training and export

A training config uses the Train page's setting names (see [Training](training.md)):

```yaml
# tutor.yaml
name: math-tutor
base_model: mlx-community/Qwen3-4B-Instruct-2507-4bit
dataset: ~/data/gsm8k.jsonl       # a file, a Hugging Face dataset or an https:// / s3:// link
method: lora
max_steps: 200
learning_rate: 2e-4
```

```bash
temper train tutor.yaml --dry-run            # check it and print every setting
temper train tutor.yaml --set max_steps=400  # start it and follow the loss; Ctrl-C leaves it running
temper runs                                  # past and current runs
temper export math-tutor --format gguf --quant q4_k_m --quant q8_0 --imatrix
```

## Running without the window

```bash
temper serve                       # this computer only, on port 8765
temper serve --host 0.0.0.0        # reachable from other computers
```

`temper serve` runs the engine on a server, in CI or over SSH. Its key is kept in `api-key` in Temper's
folder, so it stays the same between runs. Anyone with that key can manage the whole engine, so give apps
named API keys instead (see [Connect](connect.md#api-keys)).

## Agents and checks

```bash
temper start claude                # or codex, opencode, aider; extra arguments go to the agent
temper mcp add claude              # or codex, cursor, vscode
temper doctor                      # drivers, memory, disk, ports, network, engine packages
temper doctor --mcp                # checks the MCP server answers
```
