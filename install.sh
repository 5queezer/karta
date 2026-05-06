#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$repo_root"

usage() {
  cat <<'EOF'
Install the Karta CLI and update the global pi extension from this checkout.

Usage:
  ./install.sh [cargo-install-options]

Examples:
  ./install.sh
  ./install.sh --root ~/.local

Environment:
  PI_EXTENSION_DIR   Override pi extension install dir
                     (default: ~/.pi/agent/extensions)
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

if ! command -v cargo >/dev/null 2>&1; then
  echo "error: cargo is required to install Karta." >&2
  echo "Install Rust from https://rustup.rs/ and run this script again." >&2
  exit 1
fi

install_root="${CARGO_INSTALL_ROOT:-$HOME/.cargo}"
args=("$@")
for ((i = 0; i < ${#args[@]}; i++)); do
  case "${args[$i]}" in
    --root)
      if ((i + 1 < ${#args[@]})); then
        install_root="${args[$((i + 1))]}"
      fi
      ;;
    --root=*)
      install_root="${args[$i]#--root=}"
      ;;
  esac
done

printf 'Installing Karta CLI to %s/bin...\n' "$install_root"
cargo install --path crates/karta-cli --bin karta --locked --force "$@"

bin_path="$install_root/bin/karta"
pi_extension_src="$repo_root/.pi/extensions/karta.ts"
pi_extension_dir="${PI_EXTENSION_DIR:-$HOME/.pi/agent/extensions}"
pi_extension_dest="$pi_extension_dir/karta.ts"

if [[ -f "$pi_extension_src" ]]; then
  printf 'Installing/updating pi extension at %s...\n' "$pi_extension_dest"
  mkdir -p "$pi_extension_dir"
  install -m 0644 "$pi_extension_src" "$pi_extension_dest"
else
  echo "warning: pi extension source not found at $pi_extension_src; skipping pi extension install." >&2
fi

cat <<EOF

Karta CLI installed.
Pi extension installed/updated at:
  $pi_extension_dest

Run:
  karta --help

If 'karta' is not found, add this to your shell profile:
  export PATH=\"$install_root/bin:\$PATH\"

If pi cannot find 'karta', launch pi with:
  export KARTA_BIN=\"$bin_path\"
EOF

if [[ -x "$bin_path" ]]; then
  printf '\nInstalled binary: %s\n' "$bin_path"
fi
