# Speech

Temper turns speech into text with Whisper, and text into speech with Kokoro, on your computer. Nothing
you say or type is sent anywhere.

## Speech to text

### Set it up

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

### Dictate in Chat

Click the microphone in the message box, speak, and click it again. What you said goes into the message,
ready to edit and send. The first dictation after a while loads the model, which can take half a minute;
it starts loading when you start speaking, and stays loaded for 10 minutes after the last use. The
first time, your computer asks whether Temper may use the microphone.

### Transcribe a recording

**Settings → Speech to text → Transcribe a file** takes any audio or video file that FFmpeg reads (WAV, MP3,
M4A, FLAC, OGG, MP4, MOV and more). Each transcript is saved in `transcripts/` in Temper's folder as plain
text, SRT and WebVTT subtitles, and JSON with each segment's start and end.

```bash
temper transcribe interview.m4a                    # the text
temper transcribe talk.mp4 -f srt -o talk.srt      # subtitles; also vtt or json
temper transcribe lecture.mp3 --language de        # skip language detection
temper transcribe news.mp3 --translate             # English text from speech in any language
```

### From other apps

Temper answers OpenAI's audio API, so apps that transcribe through OpenAI work with Temper's address and
key:

```bash
curl http://127.0.0.1:8765/v1/audio/transcriptions -H "Authorization: Bearer $TEMPER_KEY" \
  -F file=@meeting.m4a -F model=whisper-1 -F response_format=srt
```

`response_format` can be `json` (the default), `text`, `srt`, `vtt` or `verbose_json`.
`/v1/audio/translations` returns English text. A `model` that isn't one of the three above, like
`whisper-1`, uses the model picked in Settings.

## Text to speech

The **Audio** page (*Text to speech*) reads text aloud with Kokoro, a small speech model that runs on the CPU on any
computer. The first time, it asks to install Kokoro into its own environment and then to download it
(354 MB, with all its voices). From a terminal: `temper say --install`.

- **Voices**: 54, in American and British English, Spanish, French, Hindi, Italian, Japanese, Brazilian
  Portuguese and Mandarin. The speaker button plays a sample. The voice you pick is the one Chat uses.
- **Read aloud** plays the text; **Save to gallery** keeps it as a WAV file in `voice/` in Temper's folder.
- **Chat**: each reply has a *Read aloud* button. Code blocks and Markdown marks are left out.

```bash
temper say "Your model finished training."         # plays it
temper say -v bf_emma -o note.mp3 < notes.txt        # or .wav, .opus, .flac, .aac
temper say --voices
```

Apps that use OpenAI's speech API work too:

```bash
curl http://127.0.0.1:8765/v1/audio/speech -H "Authorization: Bearer $TEMPER_KEY" \
  -H "Content-Type: application/json" -d '{"input": "Hello", "voice": "af_heart"}' -o hello.mp3
```

Kokoro's voice names work, and so do OpenAI's where Kokoro has a voice by that name (alloy, nova, echo,
onyx, fable and others); any other name uses the voice picked on the Audio page.

## Separate stems

*Audio → Separate stems* splits a song into vocals, drums, bass and everything else with Demucs, and also
saves the song without its vocals, for karaoke or a backing track. Demucs needs PyTorch, so it installs
into its own environment (about 650 MB), and its model is an 84 MB download. It runs on an NVIDIA GPU when
there is one and on the CPU otherwise; a 3-minute song takes about a minute on an M1 Pro. (PyTorch's Metal
backend lacks a convolution Demucs needs, so Macs use the CPU.) The stems are WAV files in
`stems/` in Temper's folder.

```bash
temper stems --install
temper stems song.mp3 -o song-stems/
```
