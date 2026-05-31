#!/usr/bin/env bash
# combo_bot Web UI launcher (Linux / manual terminal use).
# macOS users: double-click 启动机器人.command instead.
#
# Same logic as 启动机器人.command but without the macOS `open`
# integration — prints the URL for you to open manually.

set -e
cd "$(dirname "$0")"

CONFIG_FILE="${CONFIG_FILE:-config.testnet.json}"
PROFILE_ARGS="${PROFILE_ARGS:---testnet --real}"
UI_HOST="${UI_HOST:-127.0.0.1}"
UI_PORT="${UI_PORT:-8765}"

if command -v python3 >/dev/null 2>&1; then PY=python3; else PY=python; fi

echo "[combo_bot] checking dependencies…"
if ! $PY -c "import combo_bot, fastapi, uvicorn, jinja2, dotenv" >/dev/null 2>&1; then
  echo "[combo_bot] installing deps…"
  $PY -m pip install -e ".[ui]"
fi

if [ ! -f ".env" ]; then
  cat > .env <<'ENVEOF'
BINANCE_API_KEY=
BINANCE_API_SECRET=
ENVEOF
  echo "[combo_bot] created .env template — fill in your API keys and re-run."
  exit 1
fi

echo "[combo_bot] UI → http://$UI_HOST:$UI_PORT"
exec $PY -m combo_bot.main ui \
  --config "$CONFIG_FILE" \
  $PROFILE_ARGS \
  --host "$UI_HOST" \
  --port "$UI_PORT"
