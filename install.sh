#!/bin/sh
# ghostframe installer for Linux / Steam Deck (desktop mode):
#   curl -fsSL https://raw.githubusercontent.com/Lyamon4/ghostframe/main/install.sh | sh
set -eu
repo="Lyamon4/ghostframe"
data="${XDG_DATA_HOME:-$HOME/.local/share}"
dir="$data/ghostframe/bin"
mkdir -p "$dir" "$HOME/.local/bin"

echo "Downloading the latest ghostframe release..."
curl -fsSL "https://github.com/$repo/releases/latest/download/ghostframe-linux-x64.tar.gz" | tar xz -C "$dir"
ln -sf "$dir/ghostframe" "$HOME/.local/bin/ghostframe"

"$dir/ghostframe" install

echo
echo "Ready."
echo "  Steam (Proton or native): set the game's launch options to   ENABLE_GHOSTFRAME=1 %command%"
echo "  Anything else:            ghostframe run <game> [args]"
case ":$PATH:" in
  *":$HOME/.local/bin:"*) ;;
  *) echo "  Note: add ~/.local/bin to your PATH to use the 'ghostframe' command." ;;
esac
