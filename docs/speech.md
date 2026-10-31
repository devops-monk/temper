# Speech to text

Temper turns speech into text with Whisper, on your computer. Nothing you say is sent anywhere.

## Set it up

**Settings → Speech to text → Install** sets up Whisper in its own environment, separate from the rest
of Temper: about 470 MB on a Mac. On Apple Silicon it runs on MLX. On Windows and Linux it uses
faster-whisper, which runs on an NVIDIA GPU when CUDA's libraries are installed and on the CPU otherwise.
Then download a model in the same place:

| Model | Size | |
|-|-|-|
| Large v3 Turbo | 1.6 GB | fast and accurate in about 100 languages; the default |
| Small | 0.5 GB | quickest, fine for clear English; the default on a computer without an NVIDIA GPU |
| Large v3 | 3.1 GB | the most accurate, and the slowest |

From a terminal: `temper speech install`, then `temper speech use turbo` (it downloads the model if needed).

## Dictate in Chat

Click the microphone in the message box, speak, and click it again. What you said goes into the message,
ready to edit and send. The first dictation after a while loads the model, which can take half a minute;
it starts loading when you start speaking, and stays loaded for 10 minutes after the last use. The
first time, your computer asks whether Temper may use the microphone.

## Transcribe a recording

**Settings → Speech to text → Transcribe a file** takes any audio or video file that FFmpeg reads (WAV, MP3,
M4A, FLAC, OGG, MP4, MOV and more). Each transcript is saved in `transcripts/` in Temper's folder as plain
text, SRT and WebVTT subtitles, and JSON with each segment's start and end.

```bash
temper transcribe interview.m4a                    # the text
temper transcribe talk.mp4 -f srt -o talk.srt      # subtitles; also vtt or json
temper transcribe lecture.mp3 --language de        # skip language detection
temper transcribe news.mp3 --translate             # English text from speech in any language
```

## From other apps

Temper answers OpenAI's audio API, so apps that transcribe through OpenAI work with Temper's address and
key:

```bash
curl http://127.0.0.1:8765/v1/audio/transcriptions -H "Authorization: Bearer $TEMPER_KEY" \
  -F file=@meeting.m4a -F model=whisper-1 -F response_format=srt
```

`response_format` can be `json` (the default), `text`, `srt`, `vtt` or `verbose_json`.
`/v1/audio/translations` returns English text. A `model` that isn't one of the three above, like
`whisper-1`, uses the model picked in Settings.
