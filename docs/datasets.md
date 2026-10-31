# Datasets

Training data is a list of examples of what you want the model to do. Open **Datasets** and give it a source:

- a file: `.jsonl`, `.json`, `.csv` or `.parquet` (drop it on the window, or open it with Temper)
- a Hugging Face dataset id, such as `openai/gsm8k` (pick the subset and split if it has several)
- a web address: an `https://` link, or a public file in S3 (`s3://`) or Google Cloud Storage (`gs://`)

## Formats

Temper detects the shape of the rows and shows how the model will see them:

| Format     | Columns |
|------------|---------|
| `messages` | `messages`: a list of `{"role", "content"}` turns |
| `sharegpt` | `conversations`: `{"from", "value"}` turns |
| `alpaca`   | `instruction`, optional `input`, `output`, optional `system` |
| `text`     | `text`: plain text, for continued pretraining |

If your columns have other names (`question`/`answer`, `prompt`/`response`), map them to the roles on the page.
You can add a system prompt to every example.

## Cleaning and preparing

Before training, **Prepare** writes a clean copy:

- removes duplicates (ignoring case and spacing) and shuffles
- drops rows shorter or longer than limits you set
- keeps a share of the rows aside (5% by default) to measure validation loss while training
- caps the number of rows, for a quick first run

The prepared dataset appears in the Train form, ready to use.

## Preference data

The DPO, ORPO and SimPO objectives learn from pairs of answers, a better one and a worse one:

```json
{"prompt": "Explain recursion", "chosen": "A good answer…", "rejected": "A worse answer…"}
```

Rows may also hold `chosen` and `rejected` as whole conversations ending in each reply.

## Making data with Recipes

Have documents but no examples? **Recipes** uses a model to write training examples from PDFs, Word files,
web pages, GitHub repositories or topics you list, then you check them and train on the result.

Next: [train a model](training.md).
