Download the installer for your system from **Assets** below.

| System | File |
| --- | --- |
| macOS, Apple Silicon | `Temper_*_aarch64.dmg` |
| macOS, Intel | `Temper_*_x64.dmg` |
| Windows x64 | `Temper_*_x64-setup.exe` or `.msi` |
| Windows ARM64 | `Temper_*_arm64-setup.exe` or `.msi` |
| Linux | `.AppImage`, `.deb` or `.rpm` for amd64 / aarch64 |

On first launch Temper installs Python and its engine for you: about 150 MB for chat, or 600 MB with training on a Mac.

**These builds are not code-signed yet.**
- macOS: if it says Temper "is damaged" or "can't be opened", run `xattr -dr com.apple.quarantine /Applications/Temper.app`, or open it from System Settings → Privacy & Security → Open Anyway.
- Windows: in the SmartScreen prompt, choose More info → Run anyway.
