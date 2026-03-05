#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
TESTS_DIR="$SCRIPT_DIR/tests"

# Require scala-cli >= 1.5.0 for Scala Native 0.5.9 support.
if [[ -x /tmp/scala-cli ]]; then
  SCALA_CLI=/tmp/scala-cli
elif command -v scala-cli &>/dev/null; then
  SCALA_CLI=scala-cli
  VERSION=$($SCALA_CLI --version 2>&1 | grep -oE '[0-9]+\.[0-9]+\.[0-9]+' | head -1)
  MAJOR=$(echo "$VERSION" | cut -d. -f1)
  MINOR=$(echo "$VERSION" | cut -d. -f2)
  if (( MAJOR < 1 || (MAJOR == 1 && MINOR < 5) )); then
    echo "scala-cli $VERSION is too old. Need >= 1.5.0 for Scala Native 0.5.9."
    exit 1
  fi
else
  echo "scala-cli not found."
  exit 1
fi

# ── Parse arguments ──────────────────────────────────────────────────────

RUN_UNIT=false
RUN_INTEGRATION=false

if [[ $# -eq 0 ]]; then
  RUN_UNIT=true
fi

for arg in "$@"; do
  case "$arg" in
    --unit)        RUN_UNIT=true ;;
    --integration) RUN_INTEGRATION=true ;;
    --all)         RUN_UNIT=true; RUN_INTEGRATION=true ;;
    *)
      echo "Usage: ./run-tests.sh [--unit] [--integration] [--all]"
      echo ""
      echo "  --unit          Run unit tests (default, no model required)"
      echo "  --integration   Run integration tests (requires ./setup.sh)"
      echo "  --all           Run all tests"
      exit 1
      ;;
  esac
done

LINK_FLAGS=(
  --native-version 0.5.9
  --native-linking "-L$SCRIPT_DIR/.build/install/lib"
  --native-linking "-L$SCRIPT_DIR/.build"
  --native-linking "-lmlxllm"
  --native-linking "-lmlxc"
  --native-linking "-lmlx"
  --native-linking "-lc++"
  --native-linking "-Wl,-rpath,$SCRIPT_DIR/.build/install/lib"
  --native-linking "-Wl,-rpath,$SCRIPT_DIR/.build"
)

# ── Unit tests ───────────────────────────────────────────────────────────

if $RUN_UNIT; then
  echo "==> Running unit tests..."
  "$SCALA_CLI" test \
    "$TESTS_DIR/project.scala" \
    "$TESTS_DIR/LlamaConfigSpec.scala" \
    "$TESTS_DIR/TokenizerSpec.scala" \
    "$TESTS_DIR/SamplingSpec.scala" \
    "${LINK_FLAGS[@]}"
  echo ""
fi

# ── Integration tests (requires dylibs + model weights) ──────────────────

if $RUN_INTEGRATION; then
  echo "==> Running integration tests..."
  "$SCALA_CLI" test \
    "$TESTS_DIR/project.scala" \
    "$TESTS_DIR/IntegrationSpec.scala" \
    "${LINK_FLAGS[@]}"
  echo ""
fi

echo "==> Done."
