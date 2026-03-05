#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

# Require scala-cli >= 1.5.0 for Scala Native 0.5.9 support.
# If /tmp/scala-cli exists (downloaded manually), prefer it.
if [[ -x /tmp/scala-cli ]]; then
  SCALA_CLI=/tmp/scala-cli
elif command -v scala-cli &>/dev/null; then
  SCALA_CLI=scala-cli
  # Check version is new enough
  VERSION=$($SCALA_CLI --version 2>&1 | grep -oE '[0-9]+\.[0-9]+\.[0-9]+' | head -1)
  MAJOR=$(echo "$VERSION" | cut -d. -f1)
  MINOR=$(echo "$VERSION" | cut -d. -f2)
  if (( MAJOR < 1 || (MAJOR == 1 && MINOR < 5) )); then
    echo "scala-cli $VERSION is too old. Need >= 1.5.0 for Scala Native 0.5.9."
    echo "Install: curl -fL https://github.com/VirtusLab/scala-cli/releases/latest/download/scala-cli-aarch64-apple-darwin.gz | gunzip > /tmp/scala-cli && chmod +x /tmp/scala-cli"
    exit 1
  fi
else
  echo "scala-cli not found. Install: brew install virtuslab/scala-cli/scala-cli"
  exit 1
fi

export PROJECT_DIR

exec "$SCALA_CLI" run "$SCRIPT_DIR" \
  --main-class ChatDemo \
  --native-version 0.5.9 \
  --native-linking "-L$PROJECT_DIR/.build/install/lib" \
  --native-linking "-L$PROJECT_DIR/.build" \
  --native-linking "-lmlxllm" \
  --native-linking "-lmlxc" \
  --native-linking "-lmlx" \
  --native-linking "-lc++" \
  --native-linking "-Wl,-rpath,$PROJECT_DIR/.build/install/lib" \
  --native-linking "-Wl,-rpath,$PROJECT_DIR/.build" \
  -- "$@"
