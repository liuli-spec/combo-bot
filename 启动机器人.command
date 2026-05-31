#!/usr/bin/env bash
# combo_bot 一键启动脚本（macOS 双击运行）
#
# 双击此文件即可：
#   1. 自动切到项目目录
#   2. 检查 / 安装依赖
#   3. 检查 .env 里的 API 密钥
#   4. 启动 Web 操作台并自动打开浏览器
#
# 默认连「测试网 + 真实下单」（--testnet --real）。要改成实盘，
# 把下面 PROFILE_ARGS 改成 "--real"（不带 --testnet）—— 但请先
# 在测试网跑够一周再说。

set -e

# ── 切到脚本所在目录（即项目根目录）──────────────────────────
cd "$(dirname "$0")"
PROJECT_DIR="$(pwd)"

echo "════════════════════════════════════════════════════════"
echo "  Combo 量化机器人 · 一键启动"
echo "  项目目录: $PROJECT_DIR"
echo "════════════════════════════════════════════════════════"
echo ""

# ── 配置（改这里切换测试网 / 实盘）──────────────────────────
CONFIG_FILE="config.testnet.json"
PROFILE_ARGS="--testnet --real"   # 测试网真实下单
UI_HOST="127.0.0.1"
UI_PORT="8765"

# ── 1. 找 python ───────────────────────────────────────────
if command -v python3 >/dev/null 2>&1; then
  PY=python3
elif command -v python >/dev/null 2>&1; then
  PY=python
else
  echo "❌ 没找到 Python。请先安装 Python 3.11 或更高版本："
  echo "   https://www.python.org/downloads/"
  echo ""
  read -p "按回车键退出…"
  exit 1
fi
echo "✅ 使用 Python: $($PY --version 2>&1)"

# ── 2. 检查 / 安装依赖 ──────────────────────────────────────
echo ""
echo "🔍 检查依赖…"
if ! $PY -c "import combo_bot, fastapi, uvicorn, jinja2, dotenv" >/dev/null 2>&1; then
  echo "📦 缺少依赖，正在安装（首次运行会慢一点）…"
  $PY -m pip install -e ".[ui]" || {
    echo "❌ 依赖安装失败。请手动运行： pip install -e \".[ui]\""
    read -p "按回车键退出…"
    exit 1
  }
  echo "✅ 依赖安装完成"
else
  echo "✅ 依赖齐全"
fi

# ── 3. 检查 .env 密钥 ───────────────────────────────────────
echo ""
if [ ! -f ".env" ]; then
  echo "⚠️  没找到 .env 文件 —— 里面要放交易所 API 密钥。"
  echo ""
  echo "   现在帮你创建一个空模板。请用文本编辑器打开 .env，"
  echo "   填入你的【测试网】API key 和 secret，保存后再运行本脚本。"
  echo ""
  cat > .env <<'ENVEOF'
# 在等号后面填入你的密钥（不要带引号、不要有多余空格）
BINANCE_API_KEY=
BINANCE_API_SECRET=
ENVEOF
  echo "   已创建 .env 模板：$PROJECT_DIR/.env"
  echo "   测试网注册 + 生成密钥： https://testnet.binancefuture.com"
  echo ""
  # 尝试用默认编辑器打开
  open .env 2>/dev/null || true
  read -p "填好 .env 并保存后，按回车键继续…"
fi

# 校验 .env 里两个 key 都非空
KEY_OK=$($PY - <<'PYEOF'
import os
from pathlib import Path
vals = {}
for line in Path(".env").read_text().splitlines():
    line = line.strip()
    if not line or line.startswith("#") or "=" not in line:
        continue
    k, _, v = line.partition("=")
    vals[k.strip()] = v.strip()
ok = bool(vals.get("BINANCE_API_KEY")) and bool(vals.get("BINANCE_API_SECRET"))
print("yes" if ok else "no")
PYEOF
)
if [ "$KEY_OK" != "yes" ]; then
  echo "❌ .env 里的 BINANCE_API_KEY 或 BINANCE_API_SECRET 还是空的。"
  echo "   请填好两个密钥再运行。"
  open .env 2>/dev/null || true
  read -p "按回车键退出…"
  exit 1
fi
echo "✅ API 密钥已配置"

# ── 4. 启动 UI + 自动开浏览器 ───────────────────────────────
echo ""
echo "🚀 启动操作台…"
echo "   浏览器地址： http://$UI_HOST:$UI_PORT"
echo "   （关闭此窗口或按 Ctrl-C 即可停止操作台）"
echo ""

# 延迟 3 秒后自动打开浏览器（等服务起来）
( sleep 3 && open "http://$UI_HOST:$UI_PORT" 2>/dev/null ) &

# 前台运行 UI 服务
exec $PY -m combo_bot.main ui \
  --config "$CONFIG_FILE" \
  $PROFILE_ARGS \
  --host "$UI_HOST" \
  --port "$UI_PORT"
