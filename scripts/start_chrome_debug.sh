#!/usr/bin/env bash
# start_chrome_debug.sh
#
# Launch Google Chrome in remote debugging mode for Ocularis CDP connect mode.
#
# Security guarantees:
#   --remote-debugging-address=127.0.0.1  → CDP socket binds to localhost only
#   --user-data-dir=~/.ocularis/chrome-profile  → isolated profile, won't touch your main Chrome
#
# Usage:
#   chmod +x scripts/start_chrome_debug.sh
#   ./scripts/start_chrome_debug.sh
#
# Then in config.yaml set:
#   browser_mode: connect
# And in your RunRequest set cdp_url to "http://127.0.0.1:9222"

set -euo pipefail

CDP_PORT="${OCULARIS_CDP_PORT:-9222}"
USE_SYSTEM_PROFILE="${OCULARIS_USE_SYSTEM_PROFILE:-0}"
PROFILE_NAME="${OCULARIS_CHROME_PROFILE_NAME:-Default}"
PROFILE_DIR="${HOME}/.ocularis/chrome-profile"
SYSTEM_CHROME_BASE=""

# Detect Chrome binary
if [[ "$(uname)" == "Darwin" ]]; then
    CHROME_CANDIDATES=(
        "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
        "/Applications/Chromium.app/Contents/MacOS/Chromium"
        "$(which google-chrome 2>/dev/null || true)"
        "$(which chromium 2>/dev/null || true)"
    )
elif [[ "$(uname)" == "Linux" ]]; then
    CHROME_CANDIDATES=(
        "/usr/bin/google-chrome"
        "/usr/bin/google-chrome-stable"
        "/usr/bin/chromium"
        "/usr/bin/chromium-browser"
        "$(which google-chrome 2>/dev/null || true)"
        "$(which chromium 2>/dev/null || true)"
    )
else
    echo "Unsupported OS: $(uname)" >&2
    exit 1
fi

if [[ "$USE_SYSTEM_PROFILE" == "1" ]]; then
    if [[ "$(uname)" == "Darwin" ]]; then
        SYSTEM_CHROME_BASE="${HOME}/Library/Application Support/Google/Chrome"
    else
        SYSTEM_CHROME_BASE="${HOME}/.config/google-chrome"
    fi
    SRC_PROFILE_DIR="${SYSTEM_CHROME_BASE}/${PROFILE_NAME}"
    PROFILE_DIR="${HOME}/.ocularis/chrome-profile-system"
    mkdir -p "${PROFILE_DIR}/${PROFILE_NAME}"

    if [[ ! -d "$SRC_PROFILE_DIR" ]]; then
        echo "ERROR: Chrome profile not found: $SRC_PROFILE_DIR" >&2
        echo "Set OCULARIS_CHROME_PROFILE_NAME to the right profile (Default, Profile 1, etc)." >&2
        exit 1
    fi

    echo "Cloning profile '${PROFILE_NAME}' into isolated CDP directory..."
    if [[ -f "${SYSTEM_CHROME_BASE}/Local State" ]]; then
        cp -f "${SYSTEM_CHROME_BASE}/Local State" "${PROFILE_DIR}/Local State"
    fi
    rsync -a --delete "${SRC_PROFILE_DIR}/" "${PROFILE_DIR}/${PROFILE_NAME}/"
else
    mkdir -p "$PROFILE_DIR"
fi

CHROME_BIN=""
for candidate in "${CHROME_CANDIDATES[@]}"; do
    if [[ -n "$candidate" && -x "$candidate" ]]; then
        CHROME_BIN="$candidate"
        break
    fi
done

if [[ -z "$CHROME_BIN" ]]; then
    echo "ERROR: Could not find Chrome or Chromium. Install Google Chrome and try again." >&2
    exit 1
fi

echo "Using Chrome: $CHROME_BIN"
echo "CDP port:     $CDP_PORT (localhost only)"
echo "Profile dir:  $PROFILE_DIR"
if [[ "$USE_SYSTEM_PROFILE" == "1" ]]; then
    echo "Profile name: $PROFILE_NAME (cloned from your real Chrome profile)"
    echo "Tip: close normal Chrome windows before starting to avoid stale copied state."
fi
echo ""
echo "Once Chrome launches, set cdp_url to: http://127.0.0.1:${CDP_PORT}"
echo "Press Ctrl+C to stop Chrome and disconnect Ocularis."
echo ""

# If something else is already listening on the target CDP port, reuse it.
if curl -fsS "http://127.0.0.1:${CDP_PORT}/json/version" >/dev/null 2>&1; then
    echo "CDP already available at http://127.0.0.1:${CDP_PORT}"
    echo "You can start Ocularis in connect mode now."
    exit 0
fi

chrome_args=(
    --remote-debugging-port="$CDP_PORT"
    --remote-debugging-address="127.0.0.1"
    --user-data-dir="$PROFILE_DIR"
    --no-first-run
    --no-default-browser-check
)

if [[ "$USE_SYSTEM_PROFILE" == "1" ]]; then
    chrome_args+=(--profile-directory="$PROFILE_NAME")
fi

if (( $# > 0 )); then
    "$CHROME_BIN" "${chrome_args[@]}" "$@" &
else
    "$CHROME_BIN" "${chrome_args[@]}" &
fi

CHROME_PID=$!
trap 'kill "$CHROME_PID" 2>/dev/null || true' EXIT INT TERM

# Wait for DevTools endpoint so connect mode does not race startup.
for _ in $(seq 1 40); do
    if curl -fsS "http://127.0.0.1:${CDP_PORT}/json/version" >/dev/null 2>&1; then
        echo "CDP is ready: http://127.0.0.1:${CDP_PORT}"
        echo "WebSocket endpoint:"
        curl -fsS "http://127.0.0.1:${CDP_PORT}/json/version" | sed -n 's/.*"webSocketDebuggerUrl":"\([^"]*\)".*/\1/p'
        wait "$CHROME_PID"
        exit 0
    fi
    sleep 0.25
done

echo "ERROR: Chrome started but CDP was not reachable on port ${CDP_PORT}." >&2
echo "Try closing all Chrome windows and rerun this script." >&2
exit 1
