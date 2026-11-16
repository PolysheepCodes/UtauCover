# UtauCover

Turn a song into a ready-to-edit **OpenUTAU / UtauV** cover project:

1. **Splits the song into stems**: vocals, instrumental, lead vs backing vocals, and optionally drums / bass / other.
2. **Charts the vocals into tracks**: a Lead track plus one track per harmony line (named *Harmony high*, *Harmony low 1*, …), with notes snapped to the song's beat grid.
3. **Fills in lyrics**: from your lyric sheet (.txt), or transcribed automatically by Whisper, in any of Whisper's ~99 languages.
4. Writes a `.ustx` (plus `.mid`) with the instrumental already placed and the original vocals as muted guide tracks for A/B checks.

You still do the artistic part (tuning, phonemes, expression). This tool does the transcription grunt work.

---

## Install

Download the package for your computer. Each one is self-contained (no Python or anything else to install
first). AI models download the first time they're needed: about 3 GB with the default settings, with progress
in the window.

| | Download | Then |
|---|---|---|
| **macOS** 14+, Apple Silicon | `UtauCover-<v>-macos-arm64.dmg` | Drag **UtauCover** to **Applications** |
| **Windows** 10/11 | `UtauCover-<v>-windows-x64-setup.exe` | Installs for you only (no admin); it can also install for all users in Program Files |
| | `UtauCover-<v>-windows-x64-portable.zip` | Unzip anywhere and run `UtauCover.exe`. Nothing is installed, and its data stays in the folder |
| **Linux** x64 | `UtauCover-<v>-linux-x64.tar.gz` | Run `./UtauCover`, or `./install.sh` for a menu entry and the `utaucover` command |

UtauCover works with **UtauV or OpenUtau** on all three systems: it reads your voicebanks and phonemizers
from it, and uses its GAME note transcriber. If it isn't found automatically, the window asks for its folder.
Building the packages is described in [packaging/README.md](packaging/README.md).

### From source (development, macOS)

```bash
git clone https://github.com/PolysheepCodes/UtauCover.git
cd UtauCover
./setup.sh
```

This installs into `~/Library/Application Support/UtauCover/` (not this folder, see [iCloud note](#why-the-environment-lives-outside-this-folder)) and downloads the Whisper lyrics model (1.6 GB). Separation models download the first time you use them (0.2–0.7 GB each).

> **After updating UtauCover, quit and reopen the window.** It loads its code when it opens; the title shows the version (e.g. *UtauCover 0.7.0*), and each project's comment records which version made it.

## Three ways to use it

### 1. The window (drag and drop)

Double-click **UtauCover.app** in this folder (or run `.venv/bin/utaucover gui`).

Drop a song (and optionally a lyrics `.txt`), pick the singer, and click **Build project**. When it's done: **Open in UtauV**, **Show in Finder**, or **Fix lyrics & rebuild**. That last one loads the detected lyrics into an editor; correct the words and build again. Stems are cached, so the rebuild only redoes lyrics and notes.

### 2. The terminal

```bash
.venv/bin/utaucover build "song.mp3" --lyrics lyrics.txt --singer Googoloid
```

Useful options:

| Option | What it does |
|---|---|
| `--lyrics FILE` | Lyric sheet. Omit to transcribe automatically. |
| `--language ja` | Force the language (Whisper code: en, ja, zh, ko, es, fr, de, pt, ru, …). Auto-detected otherwise. |
| `--singer NAME` / `--harmony-singer NAME` | Voicebank (name or folder; see `utaucover singers`). |
| `--phonemizer "EN ARPA+"` / `--harmony-phonemizer …` | Phonemizer per track, by the tag UtauV shows (or class name); see `utaucover phonemizers`. Default: the voicebank's own declared phonemizer, else a sensible one for the language. |
| `--align whisper\|melody` | With `--lyrics`: time the sheet with Whisper alignment (default, accurate) or by melody only (no Whisper; much rougher, see below). |
| `--no-respell` | Don't write explicit phonemes for lines in another language. |
| `--quality best\|balanced\|fast` | Separation quality. **best** (default) leaves almost no singing in the instrumental; balanced leaves a faint ghost. Rough times for a 3-minute song on an M4 Air: 11 / 5 / 2–3 min. Only the first build of a song pays this (stems are cached per quality). |
| `--pitch-curve` | Copy the original singer's slides and vibrato into the **pitd** curve. |
| `--stems` / `--six-stems` | Also split drums / bass / other (/ guitar / piano) as muted tracks. |
| `--backing auto\|none\|split` | Backing vocals. **auto** (default): finds the main singer in both separated stems and writes harmony tracks only when the backing stem really sings a part of its own. **none**: one singer, every vocal is the lead, no lead/backing split (quickest). **split**: use the separated lead and backing stems as they are. |
| `--no-harmonies` | Same as `--backing none`. |
| `--notes auto\|game\|builtin` | Note engine. **auto** (default): UtauV's GAME transcriber when it's installed, otherwise the built-in engine with a warning. **game**: GAME or fail. **builtin**: UtauCover's own pitch tracking and segmentation (less accurate, kept as a fallback). |
| `--whisper-model turbo\|large-v3\|medium\|small` | Whisper size used to time a lyric sheet (or transcribe without one). `utaucover models` lists what's downloaded. |
| `--no-lyrics` | Notes only (every note gets a placeholder syllable). |
| `--lyric-mode word\|syllable` | `word`: a word on its first note and `+` on the rest (default for English and other alphabet languages). `syllable`: one syllable or mora per note (default for Japanese, Chinese, Korean). |
| `--script romaji\|pinyin` | Japanese in romaji instead of hiragana; Chinese pinyin (default) or `native` for hanzi. |
| `--bpm 128` | Override the detected tempo. |
| `--snap 16\|32\|64\|0` | Grid for note starts and ends (default 1/32; 0 = off). |
| `--transpose -3` | Shift every note. |
| `--dereverb` | Remove reverb from the lead before analysis (cleaner note ends on wet mixes). |
| `--fill-lyric oo` | Syllable for sung passages with no recognizable words (humming, vocalises). Default `a`. |
| `--no-legato` | Keep the small consonant gaps between notes (by default notes in a phrase touch, like a hand-made chart). |
| `--open` | Open the result in UtauV when done. (If UtauV is already running it can't be handed a file, so the .ustx is shown in Finder instead; drag it onto the UtauV window.) |

Other commands:

```bash
.venv/bin/utaucover phonemizers --language en   # phonemizers installed in UtauV/OpenUTAU (+ --refresh after adding a plugin)
.venv/bin/utaucover singers              # list installed voicebanks and their default phonemizer
.venv/bin/utaucover separate song.mp3    # only split stems
.venv/bin/utaucover lyrics vocals.wav    # only transcribe lyrics
```

### 3. Ask Claude (MCP)

UtauCover includes an MCP server, so Claude can build projects for you. Print the entry for your installation and add it to Claude Desktop's settings (`claude_desktop_config.json`) or a Claude Code `.mcp.json`:

```bash
utaucover-cli mcp --print-config        # packaged app (macOS: /Applications/UtauCover.app/Contents/MacOS/utaucover-cli)
.venv/bin/utaucover mcp --print-config  # from source
```

Then just ask, for example: *"make a cover project from ~/Downloads/song.mp3 with Googoloid, lyrics are in ~/Downloads/lyrics.txt"*.

Tools: `list_singers`, `list_phonemizers`, `start_cover_build` (takes `phonemizer`, `harmony_phonemizer`, `align`, `backing`) → `get_build_status` (builds run as background jobs), `start_stem_separation`, `transcribe_lyrics`, `read_project`, `open_in_utau`, `list_jobs`.

---

## What you get

```
<song> UtauCover/
  <song>.ustx            the project: Lead, harmony tracks, Instrumental, muted guide tracks
  <song>.mid             same notes and lyrics as MIDI
  lyrics_detected.txt    the words it used (edit, then rebuild with --lyrics this file)
  check_by_ear.txt       lead notes whose octave is in doubt, by bar and beat (only when there are some)
  pronunciation_report.txt  words checked against the phonemizer's dictionaries
  stems/                 vocals, instrumental, lead, backing (+ drums/bass/other) .wav
  analysis/              pitch tracks, word timings, result.json (for debugging)
```

The audio starts a fraction of a bar into the project so the song's bar lines land on UtauV's bar lines. The project comment says by how much.

## Note engines: GAME (default) or built-in

UtauV ships **GAME** (Generative Adaptive MIDI Extractor, by openvpi) for its "transcribe audio to notes" tool, and it's UtauCover's default note engine whenever it's installed. UtauCover doesn't bundle GAME: it runs the copy in `~/Library/UtauV/Dependencies/game` through UtauV's own code.

**If GAME isn't installed**, the window shows a warning banner at the top with a *Download GAME* button, and the CLI and MCP builds warn too. Builds still work, using the built-in engine. To install GAME, download the `.oudep` package from [openvpi/GAME releases](https://github.com/openvpi/GAME/releases/tag/oudep) and drag it onto UtauV's window, or run UtauV's "transcribe audio to notes" tool once (it offers to download it). Then click *Check again*. To try the fallback while GAME is installed, use `--notes builtin` (or set `UTAUCOVER_NO_GAME=1`).

Everything else stays the same with either engine: stems, lyrics, one note per syllable, phonemizers and the project layout. With GAME, a few extra checks run on its notes:
- **Octave slips:** a note that leaps an octave away from both neighbours and straight back is moved back.
- **Missed note changes:** where GAME holds one pitch over a sung note change (a steady second pitch for at least 0.18 s), that stretch gets its own note.
- **Context:** GAME needs a couple of seconds of audio before a phrase to hear it (a phrase right after a silence lost most of its notes), so it gets 2 s on each side of every piece it transcribes.
- **Dropouts:** if GAME still skips a stretch the pitch track clearly hears as singing, the built-in engine's notes fill it, in the surrounding octave.
- **Harmony:** on a one-singer song, harmony is read by GAME from the backing stem, then checked note by note against basic-pitch's polyphonic reading, which hears several voices at once. A note moves to the pitch a voice is actually singing (within two semitones, or an octave, but never onto the lead's pitch), and a note with no voice near it is dropped. `analysis/harmony_stages.json` shows what each step kept.

On your hand-charted projects (lead track against your chart):

| | flowerman: notes matched (F1, 50 ms) | pitch right | octave errors | die-in-a-fire: notes matched | pitch right |
|---|---|---|---|---|---|
| Built-in | 0.63 | 86% | 5.3% | 0.42 | 76% |
| GAME | **0.82** | **91%** | **0%** | **0.50** | **83%** |

GAME is a generative model, so two runs can differ (on a real voice, about 1 sung frame in 10). UtauCover runs it three times and gives each note the pitch most runs agree on. That makes it steadier (93% run-to-run agreement on Rubio's Lament), but a build takes 2–3 minutes longer. On Rubio's Lament it kept every line in one octave, where the built-in engine sang some lines an octave low. Harmonies came out the same with either engine; words landed on the right note a little less often with GAME on flowerman (63% vs 69%).

### Whisper size

Whisper only times your lyric sheet (or transcribes when there's none). `--whisper-model` / the window's *Whisper model* picks the size, and `utaucover models` shows what's downloaded. On flowerman's lyric sheet:

| Whisper | words on the right note | within 0.5 s | build time |
|---|---|---|---|
| small | 73% | 97% | 18 s |
| medium | 74% | 97% | 40 s |
| turbo (default) | 69% | 94% | ~2 min |
| large-v3 | 22% | 83% | 84 s |

Bigger isn't better for timing a sheet: large-v3's word timestamps drifted on this voice. That's one song with a synthesized voice, so turbo stays the default (it's also the best at transcribing when there's no sheet). If words land a note off, try `medium`.

## Phonemizers and pronunciation

UtauCover reads the list of phonemizers straight from UtauV / OpenUTAU (the same metadata the app uses for its menu, including plugins in `~/Library/UtauV/Plugins` and `~/Library/OpenUtau/Plugins`), so the window's dropdowns and `utaucover phonemizers` show exactly what you have installed. Each track can use a different one.

After every build, `pronunciation_report.txt` lists:

- **Which dictionaries were used.** These are the files the phonemizer itself reads: the voicebank's own (e.g. Googoloid's `en-cPv.yaml`), the Plugins folder copy, and the built-in English dictionary for ARPAbet phonemizers.
- **Words in none of them.** UtauV will guess those letter by letter, which is the usual cause of odd pronunciation. Respell them, or write the phonemes yourself as `word [p h o n e m e s]`.
- **Words written with explicit phonemes.** Lines in another language are respelled automatically, e.g. a Spanish verse on an English ARPAbet bank: `tampoco [t aa m p ow k ow]`. Spanish is supported today. Other languages are flagged but not respelled.

Multi-syllable words always get at least one note per syllable, so the phonemizer never has to cram "mel-an-cho-ly" onto one note.

### Why Whisper still runs when you give lyrics

With a lyric sheet, Whisper doesn't transcribe anything. It *aligns*: it works out when each of your words is sung, which is what puts words on the right notes. It no longer detects the language (that's read from your sheet) or transcribes the backing vocals (they reuse your sheet's words). `--align melody` (or "Melody only" in the window) skips Whisper entirely and places lines by matching syllable counts to notes and rests. On a hand-charted test it put 46% of words within half a second, against 97% with Whisper, so only use it for a quick draft.

## Lyric sheets

Plain text, one line per sung phrase:

```
[Chorus]                      <- section headers are ignored
Flower man, flower man
With his heart in the sand
(ooh, ooh)                    <- text in parentheses goes to the backing vocals
So he stands to watch x2      <- "x2" repeats the line
```

Your own phonetic spellings ("evry", "thee") are kept as written.

**Sheets that miss lines** (official lyrics often skip intros, ad-libs and outros) are fine:
- Sheet words stay where Whisper hears them. A word Whisper can't time gets a position from its neighbours, so the first sheet word won't be dragged onto an intro.
- Where Whisper's timing drifts (it can run out of sheet before the song ends and stretch the last words over the outro), the misplaced words are detected and placed by melody and phrasing instead.
- Sung stretches with no sheet words are transcribed. Confident words go on them (the log and the window list them), and wordless parts (humming, "ooh"s, wails) keep the placeholder syllable.

## Tips

- **Synthesized source vocals (UTAU / Vocaloid renders): use a lyric sheet.** Whisper can barely understand them, but with a sheet the tool places words by timing, syllable count, and phrasing. On the flowerman test it put all 185 words on notes, 95% of them within half a second of where you'd placed them.
- **Real singers** transcribe reasonably well, but Whisper still mishears words ("Trish bells" for "church bells"). When it matters, give it the official lyrics: paste them into the window's lyrics box, or pass `--lyrics`. Otherwise check `lyrics_detected.txt`, fix it, and rebuild with `--lyrics` pointing to it.
- Passages with no recognizable words (humming, "ooh"s, or a line Whisper missed) get the placeholder syllable rather than borrowing a neighbouring word. Search for `a` notes to find them.
- **Try `--pitch-curve`** (or the "Copy pitch bends & vibrato" box) for a more human result. The notes alone are deliberately plain.
- **Lead vs backing.** Lead/backing separation models guess from how a voice is mixed, and on a song with one singer they often hand whole lines to "backing". The faint residue they leave in the lead stem reads an octave or more too high. That was the source of the runaway high notes and of the lead "cutting out" into the backing guide. With `--backing auto`, the tool checks moment by moment which stem actually holds the singer and takes those lines from the full vocal stem (`stems/lead_combined.wav`). The backing guide track (`stems/backing_rest.wav`) then has them removed, so you don't hear them twice. Harmony tracks are only written when the backing stem sings a separate part (its own pitch, not the singer's unison, octave or overtones) for at least 12% of the song's singing. A song with one singer gets just the Lead track. If a real backing part is short and gets skipped, rebuild with `--backing split`. The build log and the window say what was decided.
- **Octaves.** A low or breathy voice has a strong overtone an octave up, and pitch trackers sometimes pick it. Where two readings disagree, the tool keeps the octave that fits the melody around it, and lists any lead notes still in doubt in `check_by_ear.txt` with bar and beat. Give those a listen.
- English lines in another language (e.g. a Spanish verse) are flagged in the warnings, since one voicebank's phonemizer will mispronounce them.
- **Japanese:** kanji are converted to hiragana automatically, but readings that depend on context can come out wrong (今日は → こんにちは instead of きょうは), and particles are written as spelled (は, not わ). Skim `lyrics_detected.txt` before rebuilding.
- **Instrumental quality.** *Best* (BS-RoFormer, the default) leaves almost no singing in the instrumental (about 41 dB below the vocal on Here Come the Cats); *Balanced* (MDX) leaves a faint ghost (about 25 dB below) but is ~4 minutes faster. Loud masters that overshoot full scale (common with mp3s) are turned down before separation; before v0.6.1 Balanced/Fast left the vocals clearly audible in the instrumental on such songs. Cached stems from those versions are redone automatically. All stems are written as plain 16-bit WAVs: UtauV/OpenUTAU can't read 32-bit or 'extensible' WAVs and plays them as silence (0.6.1 produced those for some songs; 0.6.2 converts them on the next build). `tools/utauv_wave_check.py FILE.wav` opens a file with UtauV's own reader. `tools/vocal_left.py` measures how much singing an instrumental still has.
- **Builds slow down a lot when the Mac is low on memory.** A build holds Whisper, basic-pitch and GAME at different moments (several GB). With many other apps open, macOS swaps to disk and a 4-minute build can take 15. Closing browsers and chat apps during a build helps.
- Tempo changes and swing aren't modelled (constant tempo is assumed). Use `--bpm` if detection picks half or double time.

## How it works

| Step | Method |
|---|---|
| Stems | [python-audio-separator](https://github.com/nomadkaraoke/python-audio-separator) running UVR models: MDX-Net / BS-RoFormer for vocals, Mel-RoFormer karaoke for lead vs backing, Demucs for instruments |
| Tempo | Whole-song phase coherence of the onset envelope at beat, 1/8 and 1/16 rates (exact BPM); vocal onsets pick on-beat vs off-beat |
| Pitch | [FCPE](https://github.com/CNChTu/FCPE) (≈100× faster than CREPE on a Mac, same accuracy on separated vocals); CREPE available |
| Notes (GAME engine) | UtauV's GAME (openvpi), run through UtauV's own code; phrases cut at silences, notes split where confident words start |
| Notes | Dynamic-programming segmentation into steady pitches (robust to vibrato), splits at consonant dips, bleed filter, tuning compensation; scoops and passing blips folded into their target notes; legato phrasing; word notes start on the vowel |
| Lyrics | Whisper large-v3-turbo via stable-ts. A lyric sheet is force-aligned and automatically re-aligned where Whisper loses sync (e.g. after instrumental breaks) |
| Lyrics → notes | Segment-level alignment weighing timing, syllable counts (CMU dictionary for English, hyphenation dictionaries for ~30 languages, kana moras, hanzi, hangul) and phrasing |
| Lead vs backing | Frame by frame: where the lead stem is much quieter than the full vocal stem and the backing stem carries the full vocal's melody, the main singer is taken from the full vocal stem. Octave disagreements between readings are settled by the surrounding melody; short notes far above or below their neighbours are treated as tracking slips |
| Harmonies | Spotify basic-pitch (polyphonic); the lead's unison, octave and overtones (2nd–8th harmonic) and faint leakage removed; harmony tracks only when the backing stem's own pitch confirms a separate part; lines dealt out by pitch rank and continuity |

`tools/eval_against_ustx.py` scores a generated project against one you charted by hand. That's how the defaults were tuned, on two hand-charted covers (flowerman and die-in-a-fire). `tools/bench.py build` rebuilds the test songs (flowerman, die-in-a-fire, Rubio's Lament, Here Come the Cats) into `~/Library/Application Support/UtauCover/bench` and scores them. `tools/bench.py validate FILE.ustx` loads a project with UtauV's own loader.

## Where things live

UtauCover's data folder is `~/Library/Application Support/UtauCover` on macOS, `%LOCALAPPDATA%\UtauCover` on
Windows (a portable copy uses its own `data` folder) and `~/.local/share/UtauCover` on Linux. It holds:

- Separation models: `models/`
- Stem cache (per song and quality preset, so rebuilds are fast): `stems/`. Delete folders there to free space.
- Settings (`settings.json`) and the window's log (`logs/window.log`)
- Whisper models: `~/.cache/whisper` (portable Windows copy: `data/models/whisper`)
- Development environment (from source): `~/Library/Application Support/UtauCover/venv` (`.venv` here is a link to it)

### Why the environment lives outside this folder

`~/Documents` is synced by iCloud Drive. A Python environment is ~20,000 files and iCloud intercepts every file access, which made imports stall for minutes. Keep large or generated folders out of `~/Documents`, or put the song and output somewhere like `~/Music` or `~/Downloads`.
