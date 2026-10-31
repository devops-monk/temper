# Training

Fine-tuning teaches a model to do one job well: answer in your style, follow your format, know your domain.
Open **Train**, choose a base model and a dataset, and press **Start**. Temper fits the batch size and
sequence length to this computer's memory and warns before a setting won't fit.

## Where it runs

| Computer | Trainer | Methods |
|----------|---------|---------|
| NVIDIA or AMD GPU | PyTorch | LoRA, QLoRA, full fine-tuning |
| Apple Silicon Mac | MLX | LoRA (on 4-bit or 8-bit models), full fine-tuning (on 16-bit models) |

A computer without either can still chat, prepare datasets and export, but not train.

## Choosing a method

**How much of the model changes:**

- **QLoRA** (the default on GPUs) loads the model in 4 bits and trains a small adapter beside it. It uses
  the least memory, so it is the place to start, and it trains models that would not otherwise fit.
- **LoRA** trains the same kind of adapter on the full-precision model. It is slightly more faithful than
  QLoRA and needs more memory. On a Mac, LoRA on a 4-bit MLX model is the equivalent of QLoRA.
- **Full fine-tuning** changes every weight. It needs several times more memory and data, and is
  rarely worth it below a few thousand high-quality examples.

`lora_r` sets the adapter's size: 16 suits most jobs; raise it to 32–64 for harder or larger datasets.

**What it learns from:**

| Objective | Data | Use it to |
|-----------|------|-----------|
| SFT (supervised) | examples of good replies | teach a task, format or style; almost always the first step |
| DPO | pairs of a better and a worse reply | steer preferences after SFT: tone, safety, fewer wrong answers |
| ORPO | the same pairs | do SFT and preference training in one pass, without a reference model |
| SimPO | the same pairs | a lighter DPO that doesn't need a reference model and favours shorter replies less |

`pref_beta` sets how far a preference objective may move the model from where it started; leave it at the default unless replies drift.

Preference objectives train an adapter, so they need LoRA or QLoRA. Run them on a model you have already
trained with SFT, or on an instruct model.

**Continued pretraining:** turn on *Train embeddings* with a `text` dataset to teach new vocabulary or a new
language. It trains the token embeddings and output head as well as the adapter, so it needs more memory.
On a Mac it needs the 16-bit version of the model.

## While it trains

- The loss chart updates live. With held-out rows (Datasets keeps 5% aside by default) it also shows the
  validation loss, which shows whether the model is learning or memorising.
- *Sample replies* answers a couple of held-out prompts at every evaluation, so you can read the
  replies improve.
- *Early stopping* ends the run once validation loss stops improving and keeps the best adapter. On a GPU it
  needs held-out rows, which Datasets → Prepare sets aside.
- Checkpoints are saved through the run (five by default). A stopped or failed run can resume from
  its last checkpoint.
- The computer is kept from idle-sleeping until the run finishes.

## Good first settings

| Setting | Start with |
|---------|-----------|
| Base model | a 1–4B instruct model, for quick runs |
| Examples | 500–2,000 clean ones beat 50,000 noisy ones |
| `max_steps` | 60 for a smoke test, then 0 to train for full epochs |
| `learning_rate` | 2e-4 for LoRA/QLoRA, 1e-5 for full fine-tuning |
| Epochs | 1–3; stop earlier if validation loss rises |

The same settings work from a YAML file with `temper train` (see [Command line](cli.md)).

Next: [export what you trained](export.md).

## Classifiers

Not every job needs a chat model. To sort text into labels you choose (billing, bug or feature; spam or
not; positive, neutral or negative), train a **classifier**: *Train → Classifier*. It learns from a file
of examples and then answers in a few milliseconds, with how sure it is of each label.

- **Examples**: a CSV, JSON or JSONL file with a column of text and a column of labels. Temper finds them by
  name (`text`, `message`, `content`… and `label`, `category`, `class`…), or you pick them. A dozen examples
  of each label is a good start; every label needs at least two.
- **How it works**: a small sentence encoder turns each text into a vector, and a classification head learns
  your labels from those vectors. On a Mac this takes seconds. *Tune the encoder* first adjusts the encoder
  on pairs of your examples (the SetFit recipe), which helps most when there are only a few per label.
- **Encoders**: MiniLM L6 (English, 87 MB, the default), BGE small (English), and Multilingual MiniLM (over 50
  languages).
- **How it did**: a fifth of each label is held out (or a prepared dataset's evaluation file is used), and
  the classifier's row shows its accuracy, precision and recall per label, which labels it confused, and
  the examples it got wrong.

The first time, Temper installs PyTorch and sentence-transformers into their own environment (about 1 GB).

Use one from your code at `POST /v1/classifiers/<name>` with `{"text": "..."}` (or `"texts"`), or make it
answer a saved decision on *Connect → Decisions* (see [Decisions](connect.md#decisions)). From a terminal:

```bash
temper classify --install
temper classify train tickets examples.csv [--text message --label team] [--tune]
temper classify tickets "I was charged twice"      # billing  (92%)
temper classify tickets < inbox.txt                 # one answer per line
```

## Embedding models

An embedding model turns text into a vector so that texts which mean the same thing land close together:
it's what knowledge bases and semantic search run on. A general one doesn't know your product names,
codenames or jargon. *Train → Embeddings* tunes one on **pairs** from your data: a question and the passage
that answers it, a search and the product it should find.

- **Pairs**: a CSV, JSON or JSONL file with two columns, found by name (`query`, `question`, `anchor`… and
  `passage`, `answer`, `positive`…) or picked. At least 10; a hundred or more is better. No negatives are
  needed: each pair's positive is pulled close and the other pairs' positives pushed away.
- **Measured before and after**: a tenth of the pairs is held out. Each held-out question searches all their
  answers (and some training answers as distractors), and the row shows how often the right one came first
  and how often it was in the top 5, before and after tuning.
- **Start from** MiniLM L6, BGE small or Multilingual MiniLM. More passes help small sets; too many and the
  model forgets what it knew, which the held-out numbers show.

It shares the classifiers' environment. Use a tuned model from your code as `model` at `/v1/embeddings`
(see [Connect](connect.md#embeddings)), or from a terminal:

```bash
temper embed train our-docs pairs.jsonl [--epochs 3] [--base BAAI/bge-small-en-v1.5]
temper embed "Who owns project Falcon?" --model our-docs     # the vector, as JSON
temper embed --list
```
