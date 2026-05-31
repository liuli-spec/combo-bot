# combo_bot operational runbook

Round-29 baseline. Targets the testnet shadow run and first real-money
deployment. Update this file when something in production behaviour
changes — don't rely on memory.

---

## TL;DR — first-day playbook

```bash
# Terminal 1: start the trader against testnet
combo-futures live --config config.testnet.json --testnet --real

# Terminal 2: live monitor (30s refresh)
python -m combo_bot.monitor --config config.testnet.json --testnet

# Anytime: one-shot status snapshot
combo-futures status --config config.testnet.json --testnet

# Emergency: cancel all + flat all + lock the trader
combo-futures kill --config config.testnet.json --testnet --reason="...why"
# After review, to resume:
rm state.testnet.STOPPED
combo-futures live --config config.testnet.json --testnet --real
```

---

## Starting the trader

`combo-futures live` enters the main loop. The `--testnet` flag selects
the testnet exchange via `combo_bot.data.create_exchange`; `--real`
means "send actual orders" (NOT "use the real exchange" — `--real` +
`--testnet` together is the right combo for a testnet shadow run).

State file naming:

| Profile | Default state file |
|---|---|
| `--testnet` | `state.testnet.json` |
| `--real` without `--testnet` | `state.real.json` |
| (neither) | `state.dryrun.json` |

The state file, intent journal (`*.intent_journal.jsonl`), and STOPPED
sentinel (`*.STOPPED`) all sit beside each other. Each profile is
fully isolated — testnet activity NEVER touches the real-money state.

### Startup checklist

1. `combo-futures status --config <c> --testnet` — confirm balance,
   no leftover positions, no STOPPED sentinel.
2. `git log --oneline -3` — confirm the deployed commit.
3. `python -m pytest tests/ -q | tail -3` — confirm green local suite.
4. Start the trader (foreground in tmux; output also goes to stdout).
5. Start the monitor in a second pane.
6. Watch the first 3-5 ticks for `tick | balance=... equity=...`
   output. If equity diverges from balance materially on the first
   tick, stop immediately and investigate.

---

## What the monitor shows

```
=== combo_bot monitor @ 2026-05-31 14:23:01Z (config=config.testnet.json) ===
  state.balance=1000.00  equity=1000.00  peak=1000.00  dd=0.00%
  risk.tier=green  red_latched=False  cooldown_until=0
  BTC/USDT:USDT  last=68450.00  grid_long=flat  grid_short=flat  ...
  --- exchange poll ---
  exchange.balance=1000.00
  BTC/USDT:USDT  open_orders=2
  BTC/USDT:USDT  positions=flat
```

Things to watch:

| Field | Healthy | Investigate when |
|---|---|---|
| `dd` | < 5% on a settled run | climbs without retracing across many ticks |
| `risk.tier` | `green` or `yellow` | sees `orange` (entries scaled by 0.5) or `red` (panic close + cooldown) |
| `red_latched` | `False` | `True` — operator action required to clear |
| `stuck_symbols` | empty | non-empty — fills polling is broken on that symbol |
| `unknown_overlay` | empty (or transient) | persistent — overlay attribution lost |
| `persistence_failed` | `False` | `True` — state save broken; entries blocked |
| `open_orders` | matches what state says (per side) | mismatch ≥ 2 — reconcile loop is drifting |
| `ex_position` | matches `state.grid_long + state.trend_long` etc | mismatch — buckets diverged from exchange |

Red banner ANYWHERE in the output → operator intervention required.

---

## Emergency stop (kill switch)

`combo-futures kill --config <c> [--testnet]` does THREE things, in
order:

1. Cancel every open order per symbol.
2. Submit market reduce-only orders to flat-close every position
   (long AND short, ground truth from `fetch_positions`).
3. Write `<state>.STOPPED` with reason + per-symbol summary.

The trader REFUSES to start while the sentinel exists. To resume:

```bash
# Inspect what kill recorded
cat state.testnet.STOPPED
# After review, remove
rm state.testnet.STOPPED
combo-futures live --config config.testnet.json --testnet --real
```

Exit codes:

* `0` — all cancels + flattens succeeded
* `1` — at least one cancel or flatten failed (sentinel still written)
* `2` — config error (no symbols)

`kill` is idempotent — running twice is safe; the second run sees no
orders / no positions and just rewrites the sentinel.

### When to kill

* You see `persistence_failed=True` for > 1 minute.
* `stuck_symbols` is populated and not clearing across multiple ticks.
* `ex_position` and the local bucket diverge by anything material.
* Equity dropped > 10% in < 1 hour without a clear market reason.
* You're going to lose network for an extended period.
* Anything in the logs you don't immediately understand and the bot
  is still actively placing orders.

### When NOT to kill

* RED latch fired and panic-close already ran — the bot already
  flattened itself. Wait for cooldown or `reset_red_latch()` rather
  than killing.
* You're spooked by a single losing trade. Killing closes ALL
  positions including grids that might recover — only kill if you
  can articulate why "no positions" is safer than "current positions".

---

## Recovery from a kill switch event

1. **Read the sentinel:** `cat state.testnet.STOPPED` — captures
   reason + per-symbol cancel/flatten counts.
2. **Read the last ~200 log lines** for the lead-up.
3. **Compare local state to exchange state:**
   ```bash
   combo-futures status --config config.testnet.json --testnet
   ```
   `state.balance` vs `exchange.balance`, `state.position_long` vs
   `ex_position long`. Any drift → investigate before resuming.
4. **Check the intent journal** for unresolved cIDs:
   ```bash
   jq '. | select(.kind == "submit" or .kind == "open")' \
     state.testnet.intent_journal.jsonl | tail -20
   ```
   Any cID that's not paired with a terminal row (filled / rejected /
   canceled / resolved) is a lost-attribution risk. The trader's
   replay at start re-claims these.
5. **Remove the sentinel** only after you understand what caused the
   stop.

---

## Common operational scenarios

### "I just deployed a new commit — is anything different?"

`git log --oneline HEAD~5..HEAD`. Diff vs the last running commit
recorded in your deploy log (you ARE keeping a deploy log, right?).
For any change in `risk.py` / `live.py` / `backtest.py` / `fill_*` —
treat the deploy as a config change and run testnet for at least
4h before promoting.

### "RED latch fired — now what?"

Round-28 default: `red_latch_auto_release_minutes=0` means latch is
manual-only. Steps:

1. `combo-futures status` — confirm `red_latched=True`.
2. Read the last 100 log lines around the latch event.
3. If the cause is identified and not recurring, in a Python REPL:
   ```python
   import json
   path = "state.testnet.json"
   state = json.loads(open(path).read())
   state["risk"]["red_latched"] = False
   state["risk"]["red_cooldown_until"] = 0
   state["risk"]["tier"] = "green"
   open(path, "w").write(json.dumps(state, indent=2))
   ```
   then restart the trader.
4. If cause is unclear, kill switch and don't restart until you
   understand.

### "I forced a kill -9 — did the journal survive?"

The intent journal is `fsync`-per-append and uses atomic compact via
tempfile + rename. A SIGKILL between `create_order` ack and
`_save_state` is the exact scenario the journal was designed for —
replay at start re-claims any cID that wasn't terminal. The state
file uses atomic `write-tmp + os.replace` so it's never half-written.

What CAN go wrong:

* OS crash before fsync flushed pages (rare on modern fs).
* Disk full on the journal volume → `persistence_failed=True` set
  at replay; trader refuses new exposure until cleared.

### "fetch_my_trades is failing on one symbol for hours"

`stuck_symbols` accumulates. `_risk_increasing_blocked()` returns
True for that symbol → no new entries. Reduce-only flows still work.

To clear after fixing the exchange issue: edit state file's
`stuck_symbols` list to remove the symbol, restart trader.

---

## Test plan for the testnet shadow run

Minimum 7 days. Each day, end-of-day, do this checklist:

- [ ] `combo-futures status` — capture snapshot, save to a daily log.
- [ ] Diff today's `state.testnet.json` against yesterday's; look for
      any field that changed unexpectedly (latch state, stuck symbols).
- [ ] `jq '.kind' state.testnet.intent_journal.jsonl | sort | uniq -c`
      — count of submit / open / filled / canceled / rejected /
      resolved. Submit count ≈ open count; terminal kinds should
      cover everything except in-flight.
- [ ] Spot-check 5-10 fills on the exchange UI vs the equivalent
      `Fill` entries in the trader's logs. The realized PnL, fee,
      and `source` attribution should match what you see on-exchange.
- [ ] Note any anomaly in a daily log; don't trust your memory.

Stretch tests across the week:

- [ ] Day 2: kill -9 the trader, restart, verify replay restored
      pending state.
- [ ] Day 3: disconnect the network for 5 min, watch recovery.
- [ ] Day 4: manually place a non-bot order on the testnet account
      via the exchange UI, verify the bot doesn't try to cancel /
      manage it.
- [ ] Day 5: temporarily set `red_threshold=0.001` in config and
      restart — verify panic-close fires and latch holds until
      manually cleared.
- [ ] Day 6-7: leave running untouched.

Promotion criteria from testnet → real:

* Zero `persistence_failed=True` events across the week.
* Zero unresolved `unknown_overlay` entries lasting > 1 hour.
* Zero divergence between local bucket state and exchange positions.
* Every fill seen on-exchange matches a Fill in the local ledger
  (no missing fills, no duplicates).
* Backtest replay of the same period matches testnet PnL within
  combined fee + slippage tolerance.

If ANY of these criteria fail, stay on testnet until fixed.

---

## First real-money deployment

Use a SEPARATE config file with a SEPARATE state file. Never share
state between testnet and real — the state-file segregation by profile
exists specifically to prevent cross-contamination.

Suggested very-first-run config (conservative even by combo-bot's
"high risk" standard):

```jsonc
{
  "symbols": ["BTC/USDT:USDT"],     // one symbol only
  "starting_balance": 200.0,         // "lose-it-all" capital
  "leverage": 2,                     // not 10
  "grid": {
    "n_positions": 1,
    "wallet_exposure_limit": 0.05,   // 5% of balance per stuck grid
    "total_wallet_exposure_limit": 0.05
  },
  "risk": {
    "trend_wallet_exposure_limit": 0  // trend overlay OFF for first run
  },
  "unfilledtimeout_entry_seconds": 600,
  "unfilledtimeout_exit_seconds": 1800
}
```

Run this for 48h with full monitoring before touching ANY knob upward.

---

## Files this runbook references

* `state.{profile}.json` — durable trader state (atomic write).
* `state.{profile}.intent_journal.jsonl` — append-only intent log.
* `state.{profile}.STOPPED` — kill-switch sentinel (operator review
  required to remove).
* `config.testnet.json` — testnet-shadow config template.
* `combo_bot/kill_switch.py` — emergency stop entry point.
* `combo_bot/monitor.py` — read-only health snapshot.
* `combo_bot/live.py` — main trader; `start()` checks STOPPED first.
