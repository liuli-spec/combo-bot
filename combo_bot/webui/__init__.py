"""Local operator web UI for combo_bot.

Single-page FastAPI app exposing trader process control + read-only
state snapshot + live log stream. Zero npm/build pipeline — server-
rendered Jinja2 + vanilla CSS + Chart.js via CDN. Operator runs:

    combo-futures ui --config config.testnet.json --testnet

and opens http://localhost:8765 in a browser.
"""
