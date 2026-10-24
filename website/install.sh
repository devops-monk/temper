#!/bin/sh
# Installs the latest Temper on macOS or Linux:
#   curl -fsSL https://temper.devops-monk.com/install.sh | sh
#
# macOS: copies Temper.app into /Applications (or ~/Applications when that isn't writable).
# Linux: puts the AppImage in ~/.local/bin/temper and adds Temper to the application menu.
# Every download is checked against the SHA-256 that GitHub publishes for the release.
#
# Settings (environment variables):
#   TEMPER_VERSION       a release tag such as v0.1.0 (default: the latest release)
#   TEMPER_INSTALL_DIR   where to put Temper.app or the AppImage

set -eu

REPO="devops-monk/temper"
ICON_URL="https://temper.devops-monk.com/assets/icon.svg"

say() { printf '%s\n' "$*"; }
fail() { printf 'Temper install failed: %s\n' "$*" >&2; exit 1; }
need() { command -v "$1" >/dev/null 2>&1 || fail "this script needs '$1'. Install it and try again."; }

need curl
need uname

case "$(uname -s)" in
  Darwin) os=macos ;;
  Linux) os=linux ;;
  *) fail "this script is for macOS and Linux. On Windows, run: irm https://temper.devops-monk.com/install.ps1 | iex" ;;
esac

case "$(uname -m)" in
  arm64 | aarch64) arch=arm64 ;;
  x86_64 | amd64) arch=x64 ;;
  *) fail "Temper has no build for $(uname -m) processors." ;;
esac

# A Mac running an Intel shell under Rosetta still gets the Apple Silicon build.
if [ "$os" = macos ] && [ "$arch" = x64 ] && [ "$(sysctl -n sysctl.proc_translated 2>/dev/null || echo 0)" = 1 ]; then
  arch=arm64
fi

case "$os-$arch" in
  macos-arm64) pattern='_aarch64\.dmg' ;;
  macos-x64) pattern='_x64\.dmg' ;;
  linux-arm64) pattern='_aarch64\.AppImage' ;;
  linux-x64) pattern='_amd64\.AppImage' ;;
esac

if [ -n "${TEMPER_VERSION:-}" ]; then
  api="https://api.github.com/repos/$REPO/releases/tags/$TEMPER_VERSION"
else
  api="https://api.github.com/repos/$REPO/releases/latest"
fi

tmp="$(mktemp -d)"
cleanup() {
  if [ -n "${mounted:-}" ]; then hdiutil detach -quiet "$mounted" 2>/dev/null || true; fi
  rm -rf "$tmp"
}
trap cleanup EXIT INT TERM

say "Finding the latest Temper release…"
# GITHUB_TOKEN is optional; it only lifts GitHub's rate limit for unauthenticated requests.
set -- -H "Accept: application/vnd.github+json"
if [ -n "${GITHUB_TOKEN:-}" ]; then set -- "$@" -H "Authorization: Bearer $GITHUB_TOKEN"; fi
curl -fsSL "$@" "$api" -o "$tmp/release.json" ||
  fail "couldn't reach GitHub. Check your internet connection and try again."

# The release JSON lists each asset's "digest" a few lines before its "browser_download_url".
url="" sum=""
digest=""
while IFS= read -r line; do
  case "$line" in
    *'"digest":'*) digest="$(printf '%s' "$line" | sed -n 's/.*"sha256:\([0-9a-f]*\)".*/\1/p')" ;;
    *'"browser_download_url":'*)
      candidate="$(printf '%s' "$line" | sed -n 's/.*"browser_download_url": *"\([^"]*\)".*/\1/p')"
      if printf '%s' "$candidate" | grep -Eq "$pattern\$"; then
        url="$candidate"
        sum="$digest"
        break
      fi
      digest=""
      ;;
  esac
done <"$tmp/release.json"

[ -n "$url" ] || fail "the release has no installer for $os on $arch yet."
file="$tmp/${url##*/}"
version="$(printf '%s' "$url" | sed -n 's#.*/download/\([^/]*\)/.*#\1#p')"

say "Downloading Temper $version for $os ($arch)…"
curl -fL --progress-bar "$url" -o "$file" || fail "the download didn't finish. Try again."

if [ -n "$sum" ]; then
  if command -v sha256sum >/dev/null 2>&1; then
    actual="$(sha256sum "$file" | cut -d' ' -f1)"
  else
    actual="$(shasum -a 256 "$file" | cut -d' ' -f1)"
  fi
  [ "$actual" = "$sum" ] || fail "the download doesn't match its published checksum. Nothing was installed."
else
  say "Note: GitHub published no checksum for this file, so it wasn't verified."
fi

if [ "$os" = macos ]; then
  dest="${TEMPER_INSTALL_DIR:-/Applications}"
  if [ -z "${TEMPER_INSTALL_DIR:-}" ] && [ ! -w "$dest" ]; then
    dest="$HOME/Applications"
  fi
  mkdir -p "$dest"
  mounted="$tmp/mnt"
  hdiutil attach -quiet -nobrowse -readonly -mountpoint "$mounted" "$file" || fail "couldn't open the disk image."
  if pgrep -xq temper 2>/dev/null; then
    say "Temper is running; quit it to finish updating."
    osascript -e 'quit app "Temper"' 2>/dev/null || true
    sleep 2
  fi
  rm -rf "$dest/Temper.app"
  ditto "$mounted/Temper.app" "$dest/Temper.app" || fail "couldn't copy Temper into $dest."
  # Files fetched with curl aren't quarantined, but clear the flag in case a proxy added it.
  xattr -dr com.apple.quarantine "$dest/Temper.app" 2>/dev/null || true
  say ""
  say "Temper $version is installed in $dest."
  say "Open it from Launchpad, or run: open \"$dest/Temper.app\""
else
  dest="${TEMPER_INSTALL_DIR:-$HOME/.local/bin}"
  mkdir -p "$dest"
  install -m 755 "$file" "$dest/temper"
  apps="${XDG_DATA_HOME:-$HOME/.local/share}/applications"
  icons="${XDG_DATA_HOME:-$HOME/.local/share}/icons/hicolor/scalable/apps"
  mkdir -p "$apps" "$icons"
  curl -fsSL "$ICON_URL" -o "$icons/temper.svg" 2>/dev/null || true
  cat >"$apps/temper.desktop" <<EOF
[Desktop Entry]
Type=Application
Name=Temper
Comment=Run, fine-tune and ship local AI models
Exec=$dest/temper
Icon=temper
Categories=Development;
Terminal=false
EOF
  say ""
  say "Temper $version is installed at $dest/temper and in your application menu."
  if ! ldconfig -p 2>/dev/null | grep -q 'libfuse\.so\.2'; then
    say "AppImages need FUSE 2. If Temper doesn't open, install it first:"
    say "  Ubuntu 24.04 and later: sudo apt install libfuse2t64"
    say "  Older Ubuntu and Debian: sudo apt install libfuse2"
    say "  Fedora: sudo dnf install fuse-libs"
  fi
  case ":$PATH:" in
    *":$dest:"*) ;;
    *) say "Add $dest to your PATH to start it with: temper" ;;
  esac
fi
