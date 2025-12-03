# MLX Swift Audio

**This package is in early development. Expect breaking changes.**

## Installation

```swift
.package(url: "https://github.com/DePasqualeOrg/mlx-swift-audio.git", branch: "main")
```

## Usage

```swift
import MLXAudio

// Kokoro - 60+ voices, speed control
let kokoro = KokoroEngine()
try await kokoro.say("Hello, world!", voice: .afHeart)

// With speed adjustment
try await kokoro.say("Speaking faster", voice: .afNova, speed: 1.5)

// Orpheus - emotional expressions
let orpheus = OrpheusEngine()
try await orpheus.say("Ha! <laugh> That's funny.", voice: .tara)

// Marvis - streaming audio
let marvis = MarvisEngine()
try await marvis.load(voice: .conversationalA)
try await marvis.sayStreaming("This plays as it generates.")

// OuteTTS - custom voices with reference audio
let outetts = OuteTTSEngine()
let speaker = try OuteTTSEngine.loadSpeaker(from: "speaker.json")
try await outetts.say("Using reference audio.", speaker: speaker)
```

For more control over playback:

```swift
let audio = try await kokoro.generate("Hello!", voice: .afHeart)
await audio.play()
```

## Building

Build the library:

```sh
xcodebuild -scheme mlx-audio -destination 'platform=macOS' build
```

Build the example app:

```sh
xcodebuild -project 'examples/TTS App/TTS App.xcodeproj' -scheme 'TTS App' -destination 'platform=macOS' build
```

## Engines

- **Kokoro**: 60+ voices, 10+ languages, speed control
- **Orpheus**: Emotional expressions (`<laugh>`, `<sigh>`, etc.)
- **Marvis**: Streaming audio generation
- **OuteTTS**: Custom voices with reference audio

## Legal and ethical considerations

Voice synthesis technology should be used responsibly. Obtain consent before using voice recordings, respect intellectual property and personality rights, never use synthetic voices for deception or fraud, and comply with applicable laws in your jurisdiction.

## History

Commit [22b498c](https://github.com/DePasqualeOrg/mlx-swift-audio/commit/22b498ceaf01fa2ee138bb36c62799172efbd6ab) in this repository corresponds to commit [0ee931b](https://github.com/DePasqualeOrg/mlx-audio/commit/0ee931b6971a338f7c48176a86db217a434a0036) ([PR #279](https://github.com/Blaizzy/mlx-audio/pull/279)) in [Blaizzy/mlx-audio](https://github.com/Blaizzy/mlx-audio), in which the Swift library and example app were completely rewritten. The commit history of files from mlx-audio has been preserved.
