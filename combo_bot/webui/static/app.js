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
  const warnings = [];
  if (d.sentinel_present) {
    warnings.push(`检测到停止锁文件 <code>${d.sentinel_path}</code> —— 机器人已被锁定，需手动解除后才能启动。`);
  }
  const state = d.state || {};
  const stuck = ((state.fill_events || {}).stuck_symbols) || [];
  if (stuck.length) {
    warnings.push(`成交回报轮询卡住的交易对：<code>${stuck.join('、')}</code>（需人工排查交易所接口）`);
  }
  const unknown = state.unknown_overlay || [];
  if (Array.isArray(unknown) && unknown.length) {
    warnings.push(`存在状态未知的挂单认领：<code>${JSON.stringify(unknown)}</code>`);
  }
  if (d.trader.state === 'crashed') {
    warnings.push(`机器人进程已崩溃（退出码 ${d.trader.exit_code}），请查看实时日志。`);
  }
  const card = $('warn-card');
  if (warnings.length) {
    card.hidden = false;
    $('warn-list').innerHTML = warnings.map(w => `<div>${w}</div>`).join('');
  } else {
    card.hidden = true;
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

// ─── boot ─────────────────────────────────────────────────────────

window.addEventListener('DOMContentLoaded', () => {
  initChart();
  attachControls();
  loadEquityHistory();
  loadFills();
  pollStatus();
  startLogStream();
});
