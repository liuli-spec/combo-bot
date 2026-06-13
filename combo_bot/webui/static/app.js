// combo_bot operator UI — client-side glue.
//
// Polls /api/status every 2s, streams /api/logs/stream via SSE,
// renders KPIs / orders / positions / fills / regime, drives the
// start/stop/kill control buttons. Zero framework — vanilla DOM +
// fetch + EventSource.

const POLL_MS = 2000;
const $ = (id) => document.getElementById(id);

let chart = null;
let lastEquityLen = 0;
let logAutoScroll = true;
let activeSymbol = '__all__'; // fills table filter

const traderStatusMap = {
  stopped:  { dot: 'dot-stopped',  label: '已停止' },
  starting: { dot: 'dot-starting', label: '启动中…' },
  running:  { dot: 'dot-running',  label: '运行中' },
  stopping: { dot: 'dot-stopping', label: '停止中…' },
  exited:   { dot: 'dot-exited',   label: '已退出' },
  crashed:  { dot: 'dot-crashed',  label: '已崩溃' },
};

const regimeBadgeMap = {
  strong_bull:  { cls: 'regime-strong-bull',  label: '强牛' },
  bull:         { cls: 'regime-bull',          label: '牛' },
  neutral:      { cls: 'regime-neutral',       label: '中性' },
  bear:         { cls: 'regime-bear',          label: '熊' },
  strong_bear:  { cls: 'regime-strong-bear',   label: '强熊' },
};

const modeBadgeMap = {
  normal:        { cls: 'mode-normal',        label: '正常' },
  aggressive:    { cls: 'mode-aggressive',    label: '激进' },
  tp_only:       { cls: 'mode-tp-only',       label: '仅平仓' },
  graceful_stop: { cls: 'mode-graceful-stop', label: '优雅退出' },
  panic:         { cls: 'mode-panic',         label: '熔断' },
};

// ─── status polling ──────────────────────────────────────────────

async function pollStatus() {
  try {
    const res = await fetch('/api/status', { cache: 'no-store' });
    if (!res.ok) throw new Error('HTTP ' + res.status);
    const data = await res.json();
    renderStatus(data);
  } catch (e) {
    console.warn('status poll failed', e);
  } finally {
    setTimeout(pollStatus, POLL_MS);
  }
}

function renderStatus(d) {
  // Trader pill
  const ts = d.trader.state;
  const meta = traderStatusMap[ts] || traderStatusMap.stopped;
  const dotEl = document.querySelector('#trader-status .status-dot');
  const txtEl = $('trader-status-text');
  if (dotEl) {
    dotEl.className = 'status-dot ' + meta.dot;
  }
  if (txtEl) {
    let label = meta.label;
    if (d.trader.exit_code !== null && (ts === 'exited' || ts === 'crashed')) {
      label += ` (退出码 ${d.trader.exit_code})`;
    }
    txtEl.textContent = label;
  }

  // Buttons
  $('btn-start').disabled = d.trader.is_running || d.sentinel_present;
  $('btn-stop').disabled = !d.trader.is_running;
  $('btn-kill').disabled = false;
  const clearBtn = $('btn-clear-sentinel');
  clearBtn.hidden = !d.sentinel_present;

  // KPIs
  const state = d.state || {};
  const balance = num(state.balance);
  const equity = num(state.equity);
  const peak = num(state.equity_peak);
  const dd = peak > 0 ? (peak - equity) / peak : 0;
  $('m-equity').textContent = fmtUsd(equity);
  $('m-equity-sub').textContent = peak > 0 ? `峰值 ${fmtUsd(peak)}` : '暂无峰值';
  $('m-equity').classList.toggle('up', equity >= balance);
  $('m-equity').classList.toggle('down', equity < balance);
  $('m-balance').textContent = fmtUsd(balance);
  $('m-balance-sub').textContent = d.exchange && d.exchange.balance != null
    ? `交易所 ${fmtUsd(d.exchange.balance)}`
    : '交易所数据不可用';
  $('m-dd').textContent = `${(dd * 100).toFixed(2)}%`;
  $('m-dd').className = 'kpi-value' + (dd >= 0.10 ? ' down' : '');
  $('m-dd-sub').textContent = peak > 0 ? `距峰值 ${fmtUsd(peak)}` : '—';

  const tier = (state.risk_tier || 'green').toString().toLowerCase();
  const tierLabels = {
    green: '绿色 · 安全', yellow: '黄色 · 减仓',
    orange: '橙色 · 仅平仓', red: '红色 · 熔断',
  };
  const tEl = $('m-tier');
  tEl.textContent = tierLabels[tier] || tier.toUpperCase();
  tEl.classList.remove('tier-green', 'tier-yellow', 'tier-orange', 'tier-red');
  tEl.classList.add('tier-' + tier);
  const subBits = [];
  if (state.risk_red_latched) subBits.push('已锁定');
  if (state.risk_red_cooldown_until && state.risk_red_cooldown_until > d.now_ms) {
    const secs = Math.round((state.risk_red_cooldown_until - d.now_ms) / 1000);
    subBits.push(`冷却 ${secs}秒`);
  }
  $('m-tier-sub').textContent = subBits.length ? subBits.join(' · ') : '一切正常';

  // ── new: PnL split (Grid / Trend) ──────────────────────────────
  renderPnlSplit(d);
  // ── new: regime mini + grid ─────────────────────────────────────
  renderRegime(d);
  // ── new: fills table ─────────────────────────────────────────────
  renderFills(d);
  // Orders
  renderOrders(d);
  // Positions
  renderPositions(d);
  // Warnings
  renderWarnings(d);

  // Equity chart
  if (equity > 0) {
    pushChartPoint(d.now_ms, equity);
  }
}

// ─── PnL split ──────────────────────────────────────────────────

function renderPnlSplit(d) {
  const pnl = d.pnl || {};
  const gridEq = num(pnl.grid_equity);
  const trendEq = num(pnl.trend_equity);
  const splitEl = $('pnl-split');
  if (gridEq === 0 && trendEq === 0) {
    splitEl.hidden = true;
    return;
  }
  splitEl.hidden = false;
  $('pnl-grid').textContent = fmtUsd(gridEq);
  $('pnl-trend').textContent = fmtUsd(trendEq);
  $('pnl-grid').classList.toggle('up', gridEq >= 0);
  $('pnl-grid').classList.toggle('down', gridEq < 0);
  $('pnl-trend').classList.toggle('up', trendEq >= 0);
  $('pnl-trend').classList.toggle('down', trendEq < 0);
}

// ─── regime display ──────────────────────────────────────────────

function renderRegime(d) {
  // Mini indicator in risk KPI
  const regime = d.regime || {};
  const primary = (regime.primary || 'neutral').toString().toLowerCase();
  const conviction = num(regime.conviction);
  const miniEl = $('regime-mini');
  if (primary === 'neutral' && conviction < 0.1) {
    miniEl.hidden = true;
    return;
  }
  miniEl.hidden = false;
  const bad = regimeBadgeMap[primary] || { cls: 'regime-neutral', label: primary };
  const badgeEl = $('regime-badge');
  badgeEl.textContent = bad.label;
  badgeEl.className = 'regime-badge ' + bad.cls;
  $('regime-strength').textContent = `conv ${(conviction * 100).toFixed(0)}%`;

  // Regime grid: per-symbol mode cards
  const detail = d.symbols_detail || {};
  const symbols = Object.keys(detail);
  if (!symbols.length) {
    $('regime-grid').innerHTML = '<div class="empty">等待状态数据…</div>';
    return;
  }
  $('regime-updated').textContent = fmtTime(d.now_ms);
  let html = '';
  for (const sym of symbols) {
    const sd = detail[sym] || {};
    const longMode = (sd.mode_long || 'normal').toString().toLowerCase();
    const shortMode = (sd.mode_short || 'normal').toString().toLowerCase();
    const lm = modeBadgeMap[longMode] || { cls: 'mode-normal', label: longMode };
    const sm = modeBadgeMap[shortMode] || { cls: 'mode-normal', label: shortMode };
    const symShort = sym.split(':')[0].replace('/USDT', '');
    html += `<div class="regime-symbol-card">
      <div class="regime-sym-name">${symShort}</div>
      <div class="regime-sym-modes">
        <span class="mode-badge ${lm.cls}">L ${lm.label}</span>
        <span class="mode-badge ${sm.cls}">S ${sm.label}</span>
      </div>
    </div>`;
  }
  $('regime-grid').innerHTML = html;
}

// ─── fills table ─────────────────────────────────────────────────

let _fillsAll = [];

async function loadFills() {
  try {
    const res = await fetch('/api/fills?limit=200', { cache: 'no-store' });
    if (!res.ok) return;
    const data = await res.json();
    _fillsAll = data.fills || [];
    renderFillsTable();
  } catch (e) {
    console.warn('fills fetch failed', e);
  }
}

function mergeFills(fills) {
  if (!fills || !fills.length) return;
  const seen = new Set(_fillsAll.map(f => f.trade_id || f.timestamp + '_' + f.price));
  for (const f of fills) {
    const key = f.trade_id || (f.timestamp || '') + '_' + (f.price || '');
    if (!seen.has(key)) {
      _fillsAll.unshift(f);
      seen.add(key);
    }
  }
  if (_fillsAll.length > 500) _fillsAll.length = 500;
}

function renderFills(d) {
  // Merge fills from status response (they come via state.fill_events.recent_fills)
  const state = d.state || {};
  const fe = state.fill_events || {};
  const recent = fe.recent_fills || [];
  mergeFills(recent);
  renderFillsTable();
  renderSymbolChips();
}

function renderSymbolChips() {
  const syms = new Set();
  for (const f of _fillsAll) {
    const s = (f.symbol || '').split(':')[0];
    if (s) syms.add(s);
  }
  const chips = ['__all__', ...Array.from(syms).sort()];
  let html = '';
  for (const s of chips) {
    const label = s === '__all__' ? '全部' : s;
    const active = s === activeSymbol ? ' active' : '';
    html += `<button class="chip${active}" data-sym="${s}">${label}</button>`;
  }
  $('symbol-chips').innerHTML = html;
  // Attach listeners
  $('symbol-chips').querySelectorAll('.chip').forEach(btn => {
    btn.addEventListener('click', () => {
      activeSymbol = btn.dataset.sym;
      renderSymbolChips();
      renderFillsTable();
    });
  });
}

function renderFillsTable() {
  let fills = _fillsAll;
  if (activeSymbol !== '__all__') {
    fills = fills.filter(f => {
      const fsym = (f.symbol || '').split(':')[0];
      return fsym === activeSymbol || (f.symbol || '') === activeSymbol;
    });
  }
  $('fills-count').textContent = `${fills.length} 条`;
  const body = $('fills-body');
  if (!fills.length) {
    body.innerHTML = '<tr><td colspan="8" class="empty">暂无成交记录</td></tr>';
    return;
  }
  body.innerHTML = fills.slice(0, 200).map(f => {
    const side = (f.side || '').toLowerCase();
    const sideCls = side === 'long' ? 'side-long' : (side === 'short' ? 'side-short' : 'side-buy');
    const srcCls = f.source === 'trend' ? 'source-trend' : 'source-grid';
    const srcLabel = f.source === 'trend' ? 'Trend' : 'Grid';
    const pnl = f.realized_pnl;
    const pnlCls = pnl >= 0 ? 'side-long' : 'side-short';
    const symShort = (f.symbol || '').split(':')[0];
    return `<tr>
      <td class="dim">${fmtTime(f.timestamp)}</td>
      <td>${symShort || f.symbol || '—'}</td>
      <td><span class="${sideCls}">${side}</span></td>
      <td><span class="${srcCls}">${srcLabel}</span></td>
      <td>${fmtQty(f.qty)}</td>
      <td>${fmtPrice(f.price)}</td>
      <td class="dim">${fmtUsd(f.fee)}</td>
      <td class="${pnlCls}">${fmtUsd(pnl)}</td>
    </tr>`;
  }).join('');
}

// CSV export
function exportFillsCSV() {
  if (!_fillsAll.length) return;
  const header = 'timestamp,symbol,side,source,qty,price,fee,realized_pnl';
  const rows = _fillsAll.map(f =>
    [f.timestamp, f.symbol, f.side, f.source, f.qty, f.price, f.fee, f.realized_pnl].join(',')
  );
  const csv = header + '\n' + rows.join('\n');
  const blob = new Blob([csv], { type: 'text/csv' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = `fills_${new Date().toISOString().slice(0, 10)}.csv`;
  a.click();
  URL.revokeObjectURL(url);
}

// ─── orders ──────────────────────────────────────────────────────

function renderOrders(d) {
  const symbolMap = (d.exchange && d.exchange.open_orders_by_symbol) || {};
  let rows = [];
  let count = 0;
  for (const [sym, entries] of Object.entries(symbolMap)) {
    if (Array.isArray(entries)) {
      for (const o of entries) {
        const symShort = sym.split(':')[0];
        rows.push(orderRow(symShort, o));
        count++;
      }
    }
  }
  $('orders-count').textContent = count.toString();
  $('orders-list').innerHTML = rows.length
    ? rows.join('')
    : `<div class="empty">暂无挂单</div>`;
}

function orderRow(sym, o) {
  const side = (o.side || '').toLowerCase();
  const sideLabel = side === 'buy' ? 'B' : (side === 'sell' ? 'S' : side);
  const sideCls = side === 'buy' ? 'side-buy' : (side === 'sell' ? 'side-sell' : '');
  const reduceTag = o.reduceOnly ? '<span class="reduce-tag">只减</span>' : '';
  return `<div class="order-row">
    <span class="dim">${sym}</span>
    <span class="${sideCls}">${sideLabel}</span>
    <span>${fmtQty(o.amount)}</span>
    <span>${fmtPrice(o.price)}</span>
    <span>${reduceTag}</span>
  </div>`;
}

// ─── positions ───────────────────────────────────────────────────

function renderPositions(d) {
  const symbolMap = (d.exchange && d.exchange.positions_by_symbol) || {};
  let rows = [];
  let count = 0;
  for (const [sym, entries] of Object.entries(symbolMap)) {
    if (Array.isArray(entries)) {
      for (const p of entries) {
        const symShort = sym.split(':')[0];
        rows.push(positionRow(symShort, p));
        count++;
      }
    }
  }
  $('positions-count').textContent = count.toString();
  $('positions-list').innerHTML = rows.length
    ? rows.join('')
    : `<div class="empty">暂无持仓</div>`;
}

function positionRow(sym, p) {
  const side = (p.side || '').toLowerCase();
  const sideLabel = side === 'long' ? 'L' : (side === 'short' ? 'S' : side);
  const upnl = p.unrealizedPnl || 0;
  const upnlCls = upnl >= 0 ? 'side-long' : 'side-short';
  return `<div class="position-row">
    <span class="dim">${sym}</span>
    <span class="side-${side}">${sideLabel}</span>
    <span>${fmtQty(p.contracts)}</span>
    <span>${fmtPrice(p.entryPrice)} → ${fmtPrice(p.markPrice)}</span>
    <span class="${upnlCls}">${fmtUsd(upnl)}</span>
  </div>`;
}

// ─── warnings ────────────────────────────────────────────────────

function renderWarnings(d) {
  const fragments = [];
  const state = d.state || {};
  const fe = state.fill_events || {};

  if (d.sentinel_present) {
    fragments.push(`<div>检测到停止锁文件 <code>${d.sentinel_path}</code> —— 机器人已被锁定，需手动解除后才能启动。</div>`);
  }

  // STUCK symbols — distinguish the two very different causes (see
  // FillEventManager._stuck_reason): "cursor" is a real pagination
  // stall (ledger-integrity risk, needs operator), "fetch" is a
  // transient connectivity error (a restart auto-retries).
  const stuck = fe.stuck_symbols || [];
  if (stuck.length) {
    const lastTs = fe.last_ts_ms || {};   // 最后一笔成功拉到的成交时间戳
    const counts = fe.stuck_count || {};
    const reasons = fe.stuck_reason || {};
    for (const sym of stuck) {
      const cnt = counts[sym] || '?';
      const wm = lastTs[sym];
      const wmStr = wm ? `最后成交回报 ${fmtTime(wm)}` : '尚无成功回报记录';
      const reason = reasons[sym] || 'unknown';
      const symSafe = sym.replace(/'/g, "\\'");

      let icon, title, detail, explain, btnLabel, hint, rowCls;
      if (reason === 'cursor') {
        rowCls = 'stuck-cursor';
        icon = '⛔';
        title = `${sym} 成交分页卡死`;
        detail = `${cnt} 次同毫秒满页 · ${wmStr}`;
        explain = '交易所在同一毫秒有大量成交，分页游标无法前进，可能漏记成交。' +
                  '请先在交易所核对该时刻成交是否完整，确认无误后再清除。';
        btnLabel = '已核实交易所，清除';
        hint = '清除后需重启机器人';
      } else if (reason === 'fetch') {
        rowCls = 'stuck-fetch';
        icon = '⚠';
        title = `${sym} 成交回报连续失败`;
        detail = `${cnt} 次拉取异常 · ${wmStr}`;
        explain = 'fetch_my_trades 连续失败（多为网络/接口波动）。' +
                  '重启机器人会自动重试，通常无需手动清除。';
        btnLabel = '立即清除';
        hint = '或直接重启机器人自动重试';
      } else {
        rowCls = 'stuck-fetch';
        icon = '⚠';
        title = `${sym} 成交回报暂停`;
        detail = `${cnt} 次连续失败 · ${wmStr}`;
        explain = '历史遗留的暂停标记（无原因记录）。重启机器人会自动重试。';
        btnLabel = '立即清除';
        hint = '或直接重启机器人自动重试';
      }

      fragments.push(`
        <div class="stuck-row ${rowCls}">
          <div class="stuck-info">
            <span class="stuck-sym">${icon} ${title}</span>
            <span class="stuck-detail">${detail}</span>
            <span class="stuck-explain">${explain}</span>
          </div>
          <div class="stuck-actions">
            <button class="ghost-btn stuck-clear-btn"
                    onclick="clearStuck('${symSafe}')"
                    title="${reason === 'cursor' ? '务必先在交易所核对该时刻成交' : '清除后重启生效'}">
              ${btnLabel}
            </button>
            <span class="stuck-hint">${hint}</span>
          </div>
        </div>`);
    }
  }

  // Transient fill-poll failures — non-blocking, self-healing. Shown
  // only when sustained (>=2 consecutive) to avoid flicker on a single
  // blip. No clear button: a fresh poll auto-recovers; the only effect
  // is that new entries are paused on each failing tick.
  const pollFailed = fe.last_poll_failed || [];
  const failCounts = fe.fetch_fail_count || {};
  const stuckSet = new Set(stuck);
  for (const sym of pollFailed) {
    if (stuckSet.has(sym)) continue;          // already shown as STUCK
    const cnt = failCounts[sym] || 0;
    if (cnt < 2) continue;                     // ignore single transient blip
    fragments.push(`
      <div class="poll-fail-row">
        <span class="poll-fail-sym">⏳ ${sym} 成交回报暂时失败</span>
        <span class="poll-fail-detail">连续 ${cnt} 次 · 本轮已暂停加仓，正在自动重试，无需手动处理</span>
      </div>`);
  }

  const unknown = state.unknown_overlay || [];
  if (Array.isArray(unknown) && unknown.length) {
    fragments.push(`<div>存在状态未知的挂单认领：<code>${JSON.stringify(unknown)}</code></div>`);
  }
  if (d.trader.state === 'crashed') {
    fragments.push(`<div>机器人进程已崩溃（退出码 ${d.trader.exit_code}），请查看实时日志。</div>`);
  }

  const card = $('warn-card');
  if (fragments.length) {
    card.hidden = false;
    $('warn-list').innerHTML = fragments.join('');
  } else {
    card.hidden = true;
  }
}

async function clearStuck(symbol) {
  const btn = document.querySelector(`.stuck-clear-btn[onclick*="${symbol.replace(/'/g, "\\'")}"]`);
  if (btn) { btn.disabled = true; btn.textContent = '清除中…'; }
  try {
    const res = await fetch('/api/control/clear_stuck', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ symbol }),
    });
    const data = await res.json();
    if (data.ok) {
      // Replace the row with a success message prompting restart
      const row = btn ? btn.closest('.stuck-row') : null;
      if (row) {
        row.innerHTML = `<div class="stuck-cleared">
          ✅ <strong>${symbol}</strong> 已从状态文件移除 ——
          点击下方<strong>停止</strong>再<strong>启动</strong>使变更生效
        </div>`;
      }
    } else {
      alert('清除失败: ' + (data.error || '未知错误'));
      if (btn) { btn.disabled = false; btn.textContent = '已排查，清除 STUCK'; }
    }
  } catch (e) {
    alert('请求失败: ' + e.message);
    if (btn) { btn.disabled = false; btn.textContent = '已排查，清除 STUCK'; }
  }
}

// ─── chart ────────────────────────────────────────────────────────

function initChart() {
  const ctx = document.getElementById('equity-chart').getContext('2d');
  chart = new Chart(ctx, {
    type: 'line',
    data: { labels: [], datasets: [{
      label: 'equity',
      data: [],
      borderColor: '#10b981',
      backgroundColor: 'rgba(16, 185, 129, 0.10)',
      borderWidth: 2,
      tension: 0.25,
      pointRadius: 0,
      fill: true,
    }] },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      animation: false,
      interaction: { mode: 'index', intersect: false },
      plugins: { legend: { display: false }, tooltip: {
        backgroundColor: '#0a0e1a', borderColor: '#10b981', borderWidth: 1,
        callbacks: { label: (ctx) => `$${ctx.parsed.y.toFixed(2)}` },
      } },
      scales: {
        x: { display: false },
        y: {
          ticks: { color: '#6e7689', font: { family: 'JetBrains Mono' } },
          grid: { color: 'rgba(255,255,255,0.05)' },
        },
      },
    },
  });
}

function pushChartPoint(ts, equity) {
  if (!chart) return;
  const labels = chart.data.labels;
  const series = chart.data.datasets[0].data;
  const lastTs = labels.length ? labels[labels.length - 1] : 0;
  if (ts - lastTs < 1500) return;
  labels.push(ts);
  series.push(equity);
  if (labels.length > 720) { labels.shift(); series.shift(); }
  chart.update('none');
  const rangeEl = $('chart-range');
  if (rangeEl && labels.length >= 2) {
    const spanMs = labels[labels.length - 1] - labels[0];
    rangeEl.textContent = `最近 ${fmtSpan(spanMs)}`;
  }
}

// ─── equity history (load on startup) ─────────────────────────────

async function loadEquityHistory() {
  try {
    const res = await fetch('/api/equity', { cache: 'no-store' });
    if (!res.ok) return;
    const data = await res.json();
    const samples = data.samples || [];
    let lastTs = 0;
    for (const s of samples) {
      if (s.ts > lastTs) {
        pushChartPoint(s.ts, s.equity);
        lastTs = s.ts;
      }
    }
  } catch (e) {
    console.warn('equity history load failed', e);
  }
}

// ─── log SSE ──────────────────────────────────────────────────────

function startLogStream() {
  const body = $('log-body');
  body.textContent = '';
  const es = new EventSource('/api/logs/stream');
  es.onmessage = (ev) => {
    appendLogLine(ev.data);
  };
  es.onerror = () => {
    appendLogLine('[界面] 日志连接已断开，正在重连…');
    es.close();
    setTimeout(startLogStream, 2000);
  };
}

function appendLogLine(line) {
  const body = $('log-body');
  const cls = classifyLogLine(line);
  const span = document.createElement('span');
  if (cls) span.className = cls;
  span.textContent = line + '\n';
  body.appendChild(span);
  while (body.childNodes.length > 800) {
    body.removeChild(body.firstChild);
  }
  if (logAutoScroll) {
    body.scrollTop = body.scrollHeight;
  }
}

function classifyLogLine(line) {
  if (/ERROR|Failed|Traceback/i.test(line)) return 'log-line-error';
  if (/WARNING|WARN|⚠/i.test(line)) return 'log-line-warn';
  if (/INFO|tick|Created|Cancelled/i.test(line)) return 'log-line-info';
  return null;
}

// ─── controls ────────────────────────────────────────────────────

async function postControl(path) {
  try {
    const res = await fetch(path, { method: 'POST' });
    return await res.json();
  } catch (e) {
    return { ok: false, message: String(e) };
  }
}

function attachControls() {
  $('btn-start').addEventListener('click', async () => {
    $('btn-start').disabled = true;
    await postControl('/api/control/start');
  });
  $('btn-stop').addEventListener('click', async () => {
    $('btn-stop').disabled = true;
    await postControl('/api/control/stop');
  });
  $('btn-kill').addEventListener('click', () => openKillModal());
  $('btn-clear-sentinel').addEventListener('click', async () => {
    await postControl('/api/control/clear_sentinel');
  });
  $('log-autoscroll').addEventListener('change', (e) => {
    logAutoScroll = e.target.checked;
  });
  $('log-clear').addEventListener('click', () => {
    $('log-body').textContent = '';
  });
  $('fills-export').addEventListener('click', exportFillsCSV);

  // Kill confirmation modal.
  const modal = $('kill-modal');
  const input = $('kill-confirm-input');
  const confirmBtn = $('kill-confirm');
  $('kill-cancel').addEventListener('click', () => closeKillModal());
  input.addEventListener('input', () => {
    confirmBtn.disabled = input.value !== 'KILL';
  });
  confirmBtn.addEventListener('click', async () => {
    confirmBtn.disabled = true;
    confirmBtn.textContent = '清仓中…';
    await postControl('/api/control/kill');
    closeKillModal();
  });

  function openKillModal() {
    input.value = '';
    confirmBtn.disabled = true;
    confirmBtn.textContent = '确认清仓';
    modal.hidden = false;
    setTimeout(() => input.focus(), 50);
  }
  function closeKillModal() {
    modal.hidden = true;
  }
}

// ─── formatters ───────────────────────────────────────────────────

function num(v) { const n = Number(v); return Number.isFinite(n) ? n : 0; }
function fmtUsd(n) {
  if (!Number.isFinite(n)) return '—';
  const sign = n < 0 ? '-' : '';
  const abs = Math.abs(n);
  return `${sign}$${abs.toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`;
}
function fmtPrice(n) {
  if (!n) return '—';
  return n.toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 4 });
}
function fmtQty(n) {
  if (!n) return '0';
  return Number(n).toFixed(6);
}
function fmtTime(ts) {
  if (!ts) return '—';
  const d = new Date(Number(ts));
  return d.toLocaleTimeString('zh-CN', { hour: '2-digit', minute: '2-digit', second: '2-digit' });
}
function fmtSpan(ms) {
  const s = ms / 1000;
  if (s < 60) return `${Math.round(s)}s`;
  if (s < 3600) return `${Math.round(s / 60)}m`;
  if (s < 86400) return `${(s / 3600).toFixed(1)}h`;
  return `${(s / 86400).toFixed(1)}d`;
}

// ─── tabs ─────────────────────────────────────────────────────────

function initTabs() {
  document.querySelectorAll('.tab-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      const target = btn.dataset.tab;
      document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
      document.querySelectorAll('.tab-pane').forEach(p => p.classList.remove('active'));
      btn.classList.add('active');
      document.getElementById('tab-' + target).classList.add('active');
    });
  });
}

// ─── backtest / optimize ──────────────────────────────────────────

let btChart = null;
let btJobId = null;
let optJobId = null;

async function submitBacktest() {
  const body = { config: {} };
  const balance = parseFloat($('bt-balance').value);
  if (Number.isFinite(balance) && balance > 0) body.config.starting_balance = balance;

  const spacing = parseFloat($('bt-grid-spacing').value);
  const emaDist = parseFloat($('bt-ema-dist').value);
  const wel = parseFloat($('bt-wel').value);
  if (Number.isFinite(spacing))  body.config.grid = { ...(body.config.grid || {}), entry_grid_spacing_pct: spacing };
  if (Number.isFinite(emaDist))  body.config.grid = { ...(body.config.grid || {}), entry_initial_ema_dist: emaDist };
  if (Number.isFinite(wel))      body.config.grid = { ...(body.config.grid || {}), wallet_exposure_limit: wel };

  $('btn-bt-run').disabled = true;
  $('bt-status').textContent = '提交中…';
  $('bt-result').hidden = true;

  try {
    const res = await fetch('/api/backtest/run', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || 'HTTP ' + res.status);
    btJobId = data.job_id;
    showJobProgress('bt', true, 0, '已加入队列…');
    pollJob(btJobId, 'bt', onBacktestDone);
  } catch (e) {
    $('bt-status').textContent = '错误: ' + e.message;
    $('btn-bt-run').disabled = false;
  }
}

async function submitOptimize() {
  const nTrials = parseInt($('opt-trials').value) || 100;
  const sampler = $('opt-sampler').value;
  const wfSplits = parseInt($('opt-wf-splits').value) || 3;
  // "adg:max,max_drawdown:min" → ["adg:max","max_drawdown:min"]; "" → [] (scalar)
  const objRaw = ($('opt-objectives') ? $('opt-objectives').value : '').trim();
  const objectives = objRaw ? objRaw.split(',').map(s => s.trim()).filter(Boolean) : [];

  $('btn-opt-run').disabled = true;
  $('opt-status').textContent = '提交中…';
  $('opt-result').hidden = true;

  try {
    const res = await fetch('/api/optimize/run', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ n_trials: nTrials, sampler, walk_forward_splits: wfSplits, objectives }),
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || 'HTTP ' + res.status);
    optJobId = data.job_id;
    showJobProgress('opt', true, 0, '已加入队列…');
    pollJob(optJobId, 'opt', onOptimizeDone);
  } catch (e) {
    $('opt-status').textContent = '错误: ' + e.message;
    $('btn-opt-run').disabled = false;
  }
}

function showJobProgress(prefix, visible, pct, msg) {
  $(`${prefix}-progress-wrap`).hidden = !visible;
  const bar = $(`${prefix}-progress-bar`);
  if (bar) bar.style.width = pct + '%';
  const msgEl = $(`${prefix}-progress-msg`);
  if (msgEl) msgEl.textContent = msg;
}

function pollJob(jobId, prefix, onDone) {
  const tick = async () => {
    try {
      const res = await fetch('/api/job/' + jobId, { cache: 'no-store' });
      if (!res.ok) return;
      const job = await res.json();
      showJobProgress(prefix, true, job.progress, job.progress_msg || '运行中…');
      $(`${prefix}-status`).textContent = job.status === 'running' ? '运行中' : '';
      if (job.status === 'done' || job.status === 'error') {
        onDone(job);
        return;
      }
      setTimeout(tick, 800);
    } catch (e) {
      setTimeout(tick, 2000);
    }
  };
  tick();
}

function onBacktestDone(job) {
  $('btn-bt-run').disabled = false;
  if (job.status === 'error') {
    $('bt-status').textContent = '失败: ' + (job.error || '未知错误');
    showJobProgress('bt', false, 0, '');
    return;
  }
  $('bt-status').textContent = '完成';
  showJobProgress('bt', true, 100, '回测完成');
  renderBacktestResult(job.result);
}

function renderBacktestResult(r) {
  if (!r) return;
  const el = $('bt-result');
  el.hidden = false;

  const fmt = (v, dec=2) => Number.isFinite(v) ? v.toFixed(dec) : '—';
  const pct = (v) => Number.isFinite(v) ? (v * 100).toFixed(2) + '%' : '—';

  $('bt-metrics').innerHTML = `
    <div class="metric-card"><div class="metric-val ${r.adg >= 0 ? 'up' : 'down'}">${pct(r.adg)}</div><div class="metric-label">ADG（日收益）</div></div>
    <div class="metric-card"><div class="metric-val down">${pct(r.max_drawdown)}</div><div class="metric-label">最大回撤</div></div>
    <div class="metric-card"><div class="metric-val">${fmt(r.sharpe_ratio)}</div><div class="metric-label">Sharpe</div></div>
    <div class="metric-card"><div class="metric-val">${fmt(r.sortino_ratio)}</div><div class="metric-label">Sortino</div></div>
    <div class="metric-card"><div class="metric-val">${fmt(r.calmar_ratio)}</div><div class="metric-label">Calmar</div></div>
    <div class="metric-card"><div class="metric-val">${r.n_trades}</div><div class="metric-label">总成交笔数</div></div>
    <div class="metric-card"><div class="metric-val">${pct(r.win_rate)}</div><div class="metric-label">胜率</div></div>
    <div class="metric-card"><div class="metric-val">${fmt(r.duration_days, 0)} 天</div><div class="metric-label">回测时长</div></div>
    <div class="metric-card"><div class="metric-val ${r.total_pnl >= 0 ? 'up' : 'down'}">${fmtUsd(r.total_pnl)}</div><div class="metric-label">总盈亏</div></div>
    <div class="metric-card"><div class="metric-val">${fmtUsd(r.total_fees)}</div><div class="metric-label">手续费</div></div>
    <div class="metric-card"><div class="metric-val ${r.grid_pnl >= 0 ? 'up' : 'down'}">${fmtUsd(r.grid_pnl)}</div><div class="metric-label">Grid 盈亏</div></div>
    <div class="metric-card"><div class="metric-val ${r.trend_pnl >= 0 ? 'up' : 'down'}">${fmtUsd(r.trend_pnl)}</div><div class="metric-label">Trend 盈亏</div></div>
  `;

  // Equity chart
  if (r.equity_curve && r.equity_curve.length > 1) {
    const tss = r.equity_curve.map(p => p[0]);
    const eqs = r.equity_curve.map(p => p[1]);
    if (btChart) btChart.destroy();
    const ctx = document.getElementById('bt-equity-chart').getContext('2d');
    btChart = new Chart(ctx, {
      type: 'line',
      data: { labels: tss, datasets: [{
        label: 'equity', data: eqs,
        borderColor: '#10b981', backgroundColor: 'rgba(16,185,129,0.10)',
        borderWidth: 2, tension: 0.2, pointRadius: 0, fill: true,
      }] },
      options: {
        responsive: true, maintainAspectRatio: false, animation: false,
        plugins: { legend: { display: false }, tooltip: {
          backgroundColor: '#0a0e1a', borderColor: '#10b981', borderWidth: 1,
          callbacks: { label: ctx => '$' + ctx.parsed.y.toFixed(2),
                       title: ctx => new Date(ctx[0].parsed.x).toLocaleDateString('zh-CN') },
        }},
        scales: {
          x: { display: false },
          y: { ticks: { color: '#6e7689', font: { family: 'JetBrains Mono' } },
               grid: { color: 'rgba(255,255,255,0.05)' } },
        },
      },
    });
  }
}

function onOptimizeDone(job) {
  $('btn-opt-run').disabled = false;
  if (job.status === 'error') {
    $('opt-status').textContent = '失败: ' + (job.error || '未知错误');
    showJobProgress('opt', false, 0, '');
    return;
  }
  $('opt-status').textContent = '完成';
  showJobProgress('opt', true, 100, '优化完成');
  renderOptimizeResult(job.result);
}

function renderOptimizeResult(r) {
  if (!r) return;
  $('opt-result').hidden = false;
  const fmt = (v, dec=4) => Number.isFinite(Number(v)) ? Number(v).toFixed(dec) : String(v);

  if (r.pareto_front) {
    renderParetoFront(r);
    return;
  }

  // Legacy single-scalar result.
  $('opt-pareto-wrap').hidden = true;
  $('opt-metrics').innerHTML = `
    <div class="metric-card"><div class="metric-val up">${fmt(r.best_value)}</div><div class="metric-label">最优得分</div></div>
    <div class="metric-card"><div class="metric-val">${r.n_trials}</div><div class="metric-label">完成试验</div></div>
    <div class="metric-card"><div class="metric-val dim">${r.study_name || '—'}</div><div class="metric-label">Study 名称</div></div>
  `;
  $('opt-params-title').textContent = '最优参数（可复制到 config.json）';
  $('opt-params-json').textContent = JSON.stringify(
    { grid: r.grid || {}, trend: r.trend || {}, merger: r.merger || {} }, null, 2
  );
}

function renderParetoFront(r) {
  const fmt = (v, dec=4) => Number.isFinite(Number(v)) ? Number(v).toFixed(dec) : String(v);
  const front = r.pareto_front || [];
  const objKeys = (r.objectives || []).map(o => o.split(':')[0]);

  $('opt-metrics').innerHTML = `
    <div class="metric-card"><div class="metric-val up">${front.length}</div><div class="metric-label">帕累托解数</div></div>
    <div class="metric-card"><div class="metric-val">${r.n_trials}</div><div class="metric-label">完成试验</div></div>
    <div class="metric-card"><div class="metric-val dim">${(r.objectives || []).join(' · ')}</div><div class="metric-label">优化目标</div></div>
  `;

  // Table: one row per non-dominated solution, columns = objective values.
  $('opt-pareto-wrap').hidden = false;
  $('opt-pareto-head').innerHTML =
    '<tr><th>#</th>' + objKeys.map(k => `<th>${k}</th>`).join('') + '</tr>';
  $('opt-pareto-body').innerHTML = front.map((sol, idx) => {
    const cells = objKeys.map(k => {
      const v = sol.objectives[k];
      const pctLike = k.includes('drawdown') || k === 'adg' || k.includes('win');
      return `<td>${pctLike ? (Number(v) * 100).toFixed(2) + '%' : fmt(v)}</td>`;
    }).join('');
    return `<tr class="pareto-row" data-idx="${idx}" style="cursor:pointer">
      <td class="dim">${idx + 1}</td>${cells}</tr>`;
  }).join('');

  // Click a row → show that solution's config blocks.
  const showSolution = (idx) => {
    const sol = front[idx];
    if (!sol) return;
    $('opt-params-title').textContent =
      `解 #${idx + 1} 参数（可复制到 config.json）`;
    $('opt-params-json').textContent = JSON.stringify(
      { grid: sol.grid || {}, trend: sol.trend || {}, merger: sol.merger || {} }, null, 2
    );
    $('opt-pareto-body').querySelectorAll('.pareto-row').forEach(row => {
      row.classList.toggle('pareto-row-active', Number(row.dataset.idx) === idx);
    });
  };
  $('opt-pareto-body').querySelectorAll('.pareto-row').forEach(row => {
    row.addEventListener('click', () => showSolution(Number(row.dataset.idx)));
  });
  // Default: show the first solution.
  if (front.length) showSolution(0);
}

function attachLabControls() {
  $('btn-bt-run').addEventListener('click', submitBacktest);
  $('btn-opt-run').addEventListener('click', submitOptimize);
}

// ─── boot ─────────────────────────────────────────────────────────

window.addEventListener('DOMContentLoaded', () => {
  initTabs();
  initChart();
  attachControls();
  attachLabControls();
  loadEquityHistory();
  loadFills();
  pollStatus();
  startLogStream();
  initDeepSeek();
});

// ═══════════════════════════════════════════════════════════════════
// DeepSeek AI 分析 (v3)
// ═══════════════════════════════════════════════════════════════════

let _dsKey = '';         // 内存中暂存的 API key
let _dsAvailable = false;
let _lastBtJobId = null; // 最近回测 job id
let _chatHistory = [];   // 对话历史 [{role, content}]

function initDeepSeek() {
  // 尝试自动加载环境变量 key（如果后端已配置则无需手动输入）
  checkDSHealth();

  // 配置按钮
  $('btn-ds-config').addEventListener('click', configureDS);

  // 分析按钮
  $('btn-ds-analyze').addEventListener('click', () => analyzeBacktest(false));
  $('btn-ds-analyze-stream').addEventListener('click', () => analyzeBacktest(true));

  // 聊天
  $('btn-ds-send').addEventListener('click', sendChat);
  $('ds-chat-input').addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); sendChat(); }
  });

  // 快捷提问按钮
  document.querySelectorAll('.ai-quick-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      $('ds-chat-input').value = btn.dataset.q;
      sendChat();
    });
  });

  // 市场分析按钮
  const mktBtn = $('btn-ds-market');
  if (mktBtn) mktBtn.addEventListener('click', analyzeMarket);
}

async function checkDSHealth() {
  try {
    const res = await fetch('/api/deepseek/health');
    const data = await res.json();
    _dsAvailable = data.available;
    renderDSStatus(data);
  } catch (e) {
    _dsAvailable = false;
    renderDSStatus({ available: false, message: '无法连接后端' });
  }
}

function renderDSStatus(data) {
  const el = $('ds-health');
  if (!el) return;
  if (data.available) {
    el.innerHTML = `<span class="ai-status ai-status-ok">● 已连接</span><span class="ai-model-select">${data.model || 'deepseek-chat'}</span>`;
    $('ds-key').placeholder = '已从环境变量加载（无需手动输入）';
    $('btn-ds-config').textContent = '重新配置';
  } else {
    el.innerHTML = `<span class="ai-status ai-status-err">○ 未配置</span>`;
    $('ds-key').placeholder = '粘贴 DeepSeek API key (sk-…)';
  }
}

async function configureDS() {
  const key = $('ds-key').value.trim();
  if (!key) {
    appendChatMsg('assistant', '请先在输入框中粘贴 DeepSeek API key。');
    return;
  }
  $('btn-ds-config').disabled = true;
  $('btn-ds-config').textContent = '配置中…';
  try {
    const res = await fetch('/api/deepseek/config', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ api_key: key }),
    });
    const data = await res.json();
    if (data.ok) {
      _dsAvailable = true;
      _dsKey = key;
      $('ds-health').innerHTML = `<span class="ai-status ai-status-ok">● 已连接</span><span class="ai-model-select">${data.model}</span>`;
      $('btn-ds-config').textContent = '✓ 已配置';
      appendChatMsg('assistant', 'DeepSeek 客户端配置成功！现在可以使用 AI 分析功能了。');
    } else {
      $('ds-health').innerHTML = `<span class="ai-status ai-status-err">✕ ${data.message || '配置失败'}</span>`;
      $('btn-ds-config').textContent = '重试';
      appendChatMsg('error', '配置失败: ' + (data.message || '未知错误'));
    }
  } catch (e) {
    $('ds-health').innerHTML = `<span class="ai-status ai-status-err">✕ 网络错误</span>`;
    $('btn-ds-config').textContent = '重试';
    appendChatMsg('error', '请求失败: ' + e.message);
  } finally {
    $('btn-ds-config').disabled = false;
  }
}

async function analyzeBacktest(stream) {
  const resultEl = $('ds-analysis-result');
  if (!_dsAvailable) {
    resultEl.innerHTML = '<div class="empty">请先配置 DeepSeek API key。</div>';
    return;
  }

  // 优先用最近的回测 job
  let jobId = _lastBtJobId;
  if (!jobId) {
    // 尝试从 job list 找最近的 backtest
    try {
      const res = await fetch('/api/jobs');
      const data = await res.json();
      const btJobs = (data.jobs || []).filter(j => j.kind === 'backtest' && j.status === 'done');
      if (btJobs.length > 0) jobId = btJobs[0].id;
    } catch (e) {}
  }
  if (!jobId) {
    resultEl.innerHTML = '<div class="empty">请先在「回测实验室」运行一次回测。</div>';
    return;
  }

  if (!stream) {
    // 非流式：fetch 完整结果
    resultEl.innerHTML = '<div class="ai-dot-pulse"><span></span><span></span><span></span></div><span style="color:var(--text-dim)">AI 正在分析…</span>';
    try {
      // 获取 job 的 result
      const jobRes = await fetch('/api/job/' + jobId);
      const job = await jobRes.json();
      if (!job.result) throw new Error('回测结果不可用');

      const res = await fetch('/api/deepseek/analyze', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ result: job.result }),
      });
      const data = await res.json();
      if (data.error) {
        resultEl.innerHTML = `<div class="empty" style="color:var(--rose)">${data.error}</div>`;
      } else {
        resultEl.innerHTML = simpleMarkdown(data.analysis || '无分析结果');
        resultEl.scrollTop = 0;
      }
    } catch (e) {
      resultEl.innerHTML = `<div class="empty" style="color:var(--rose)">分析失败: ${e.message}</div>`;
    }
    return;
  }

  // 流式 SSE
  resultEl.innerHTML = '<span class="typing-cursor"></span>';
  try {
    const evtSrc = new EventSource(
      `/api/deepseek/analyze/stream?job_id=${encodeURIComponent(jobId)}`
    );
    let firstChunk = true;
    evtSrc.onmessage = (ev) => {
      try {
        const d = JSON.parse(ev.data);
        if (d.content) {
          if (firstChunk) { resultEl.innerHTML = ''; firstChunk = false; }
          resultEl.innerHTML += d.content.replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/\n/g, '<br>');
          resultEl.scrollTop = resultEl.scrollHeight;
        } else if (d.status) {
          resultEl.innerHTML = `<span style="color:var(--text-dim)">${d.msg || d.status}</span>`;
        } else if (d.error) {
          resultEl.innerHTML = `<span style="color:var(--rose)">错误: ${d.error}</span>`;
        }
      } catch {}
    };
    evtSrc.onerror = () => {
      evtSrc.close();
      // 如果已有内容，用 simpleMarkdown 重新渲染
      if (resultEl.textContent && resultEl.textContent.length > 20) {
        const raw = resultEl.textContent;
        resultEl.innerHTML = simpleMarkdown(raw);
      }
    };
  } catch (e) {
    resultEl.innerHTML = `<span style="color:var(--rose)">SSE 连接失败: ${e.message}</span>`;
  }
}

function simpleMarkdown(text) {
  if (!text) return '';
  return text
    .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
    // 标题
    .replace(/^### (.+)$/gm, '<h3>$1</h3>')
    .replace(/^## (.+)$/gm, '<h2>$1</h2>')
    .replace(/^# (.+)$/gm, '<h1>$1</h1>')
    // 粗体
    .replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>')
    // 行内代码
    .replace(/`([^`]+)`/g, '<code>$1</code>')
    // 列表
    .replace(/^- (.+)$/gm, '<li>$1</li>')
    .replace(/^(\d+)\. (.+)$/gm, '<li>$2</li>')
    // 区块引用
    .replace(/^> (.+)$/gm, '<blockquote>$1</blockquote>')
    // 段落（连续非空行）
    .replace(/\n\n/g, '</p><p>')
    // 单换行
    .replace(/\n/g, '<br>')
    // wrap
    .replace(/^(.+)$/m, (m) => m.startsWith('<') ? m : `<p>${m}</p>`);
}

// ─── 聊天 ─────────────────────────────────────────────────────────

function sendChat() {
  const input = $('ds-chat-input');
  const prompt = input.value.trim();
  if (!prompt) return;

  if (!_dsAvailable) {
    appendChatMsg('error', '请先配置 DeepSeek API key。');
    return;
  }

  input.value = '';
  appendChatMsg('user', prompt);
  appendChatMsg('assistant', '<span class="ai-dot-pulse"><span></span><span></span><span></span></span>', true);
  _chatHistory.push({ role: 'user', content: prompt });

  const evtSrc = new EventSource(
    `/api/deepseek/freeform/stream?prompt=${encodeURIComponent(prompt)}`
  );

  let lastMsg = document.querySelector('#ds-chat-body .ai-chat-msg-assistant:last-child');
  let content = '';
  let firstChunk = true;

  evtSrc.onmessage = (ev) => {
    try {
      const d = JSON.parse(ev.data);
      if (d.content) {
        if (firstChunk && lastMsg) {
          lastMsg.innerHTML = '';
          firstChunk = false;
        }
        content += d.content;
        if (lastMsg) {
          lastMsg.innerHTML = simpleMarkdown(content);
          $('ds-chat-body').scrollTop = $('ds-chat-body').scrollHeight;
        }
      } else if (d.error) {
        if (lastMsg) lastMsg.innerHTML = `<span style="color:var(--rose)">${d.error}</span>`;
      }
    } catch {}
  };

  evtSrc.onerror = () => {
    evtSrc.close();
    if (content) _chatHistory.push({ role: 'assistant', content });
  };
}

function appendChatMsg(role, html, isStreaming) {
  const body = $('ds-chat-body');
  const cls = role === 'user' ? 'ai-chat-msg-user' :
              role === 'error' ? 'ai-chat-msg-error' : 'ai-chat-msg-assistant';
  const div = document.createElement('div');
  div.className = `ai-chat-msg ${cls}`;
  if (isStreaming) div.dataset.streaming = '1';
  div.innerHTML = html;
  body.appendChild(div);
  body.scrollTop = body.scrollHeight;
}

// ─── 市场分析 ─────────────────────────────────────────────────────

async function analyzeMarket() {
  if (!_dsAvailable) {
    appendChatMsg('error', '请先配置 DeepSeek API key。');
    return;
  }
  const resultEl = $('ds-analysis-result');
  resultEl.innerHTML = '<div class="ai-dot-pulse"><span></span><span></span><span></span></div><span style="color:var(--text-dim)">AI 正在分析市场…</span>';
  try {
    const res = await fetch('/api/deepseek/market');
    const data = await res.json();
    if (data.error) {
      resultEl.innerHTML = `<div class="empty" style="color:var(--rose)">${data.error}</div>`;
    } else {
      resultEl.innerHTML = simpleMarkdown(data.analysis || '无分析结果');
    }
  } catch (e) {
    resultEl.innerHTML = `<div class="empty" style="color:var(--rose)">市场分析失败: ${e.message}</div>`;
  }
}

// ─── 跟踪最近回测 job ─────────────────────────────────────────────

// Monkey-patch onBacktestDone to record the job id
const _origOnBacktestDone = onBacktestDone;
onBacktestDone = function(job) {
  _lastBtJobId = job.id;
  _origOnBacktestDone(job);
};
