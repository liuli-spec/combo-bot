// combo_bot operator UI — client-side glue.
//
// Polls /api/status every 2s, streams /api/logs/stream via SSE,
// renders KPIs / orders / positions, drives the start/stop/kill
// control buttons. Zero framework — vanilla DOM + fetch + EventSource.

const POLL_MS = 2000;
const $ = (id) => document.getElementById(id);

let chart = null;
let lastEquityLen = 0;
let logAutoScroll = true;

const traderStatusMap = {
  stopped:  { dot: 'dot-stopped',  label: 'STOPPED' },
  starting: { dot: 'dot-starting', label: 'STARTING…' },
  running:  { dot: 'dot-running',  label: 'RUNNING' },
  stopping: { dot: 'dot-stopping', label: 'STOPPING…' },
  exited:   { dot: 'dot-exited',   label: 'EXITED' },
  crashed:  { dot: 'dot-crashed',  label: 'CRASHED' },
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
      label += ` (code ${d.trader.exit_code})`;
    }
    txtEl.textContent = label;
  }

  // Buttons
  $('btn-start').disabled = d.trader.is_running || d.sentinel_present;
  $('btn-stop').disabled = !d.trader.is_running;
  $('btn-kill').disabled = false;
  const clearBtn = $('btn-clear-sentinel');
  if (d.sentinel_present) {
    clearBtn.hidden = false;
  } else {
    clearBtn.hidden = true;
  }

  // KPIs
  const state = d.state || {};
  const balance = num(state.balance);
  const equity = num(state.equity);
  const peak = num(state.equity_peak);
  const dd = peak > 0 ? (peak - equity) / peak : 0;
  $('m-equity').textContent = fmtUsd(equity);
  $('m-equity-sub').textContent = peak > 0 ? `peak ${fmtUsd(peak)}` : 'no peak yet';
  const equityCard = $('kpi-equity');
  equityCard.classList.remove('kpi-equity');
  equityCard.classList.add('kpi-equity');
  $('m-equity').classList.toggle('up', equity >= balance);
  $('m-equity').classList.toggle('down', equity < balance);
  $('m-balance').textContent = fmtUsd(balance);
  $('m-balance-sub').textContent = d.exchange && d.exchange.balance != null
    ? `exchange ${fmtUsd(d.exchange.balance)}`
    : 'exchange n/a';
  $('m-dd').textContent = `${(dd * 100).toFixed(2)}%`;
  $('m-dd').className = 'kpi-value' + (dd >= 0.10 ? ' down' : '');
  $('m-dd-sub').textContent = peak > 0 ? `from ${fmtUsd(peak)}` : '—';

  const tier = (state.risk_tier || 'green').toString().toLowerCase();
  const tEl = $('m-tier');
  tEl.textContent = tier.toUpperCase();
  tEl.classList.remove('tier-green', 'tier-yellow', 'tier-orange', 'tier-red');
  tEl.classList.add('tier-' + tier);
  const subBits = [];
  if (state.risk_red_latched) subBits.push('LATCHED');
  if (state.risk_red_cooldown_until && state.risk_red_cooldown_until > d.now_ms) {
    const secs = Math.round((state.risk_red_cooldown_until - d.now_ms) / 1000);
    subBits.push(`cooldown ${secs}s`);
  }
  $('m-tier-sub').textContent = subBits.length ? subBits.join(' · ') : 'all clear';

  // Orders
  renderOrders(d);

  // Positions
  renderPositions(d);

  // Warnings
  renderWarnings(d);

  // Equity chart (separate endpoint to keep status payload small,
  // but we already have the equity value in state — push it into
  // the chart history client-side too so the first sample shows up
  // without waiting for /api/equity).
  if (equity > 0) {
    pushChartPoint(d.now_ms, equity);
  }
}

function renderOrders(d) {
  const symbolMap = (d.exchange && d.exchange.open_orders_by_symbol) || {};
  let rows = [];
  let count = 0;
  for (const [sym, entries] of Object.entries(symbolMap)) {
    if (Array.isArray(entries)) {
      for (const o of entries) {
        rows.push(orderRow(sym, o));
        count++;
      }
    }
  }
  $('orders-count').textContent = count.toString();
  $('orders-list').innerHTML = rows.length
    ? rows.join('')
    : `<div class="empty">no open orders</div>`;
}

function orderRow(sym, o) {
  const side = (o.side || '').toLowerCase();
  const reduceTag = o.reduceOnly ? '<span class="reduce-tag">REDUCE</span>' : '';
  return `<div class="order-row">
    <span class="side-${side}">${side.toUpperCase()}</span>
    <span>${fmtQty(o.amount)}</span>
    <span>@ ${fmtPrice(o.price)}</span>
    <span>${reduceTag}</span>
  </div>`;
}

function renderPositions(d) {
  const symbolMap = (d.exchange && d.exchange.positions_by_symbol) || {};
  let rows = [];
  let count = 0;
  for (const [sym, entries] of Object.entries(symbolMap)) {
    if (Array.isArray(entries)) {
      for (const p of entries) {
        rows.push(positionRow(sym, p));
        count++;
      }
    }
  }
  $('positions-count').textContent = count.toString();
  $('positions-list').innerHTML = rows.length
    ? rows.join('')
    : `<div class="empty">no open positions</div>`;
}

function positionRow(sym, p) {
  const side = (p.side || '').toLowerCase();
  const upnl = p.unrealizedPnl || 0;
  const upnlCls = upnl >= 0 ? 'side-buy' : 'side-sell';
  return `<div class="position-row">
    <span class="side-${side}">${side.toUpperCase()}</span>
    <span>${fmtQty(p.contracts)}</span>
    <span>@ ${fmtPrice(p.entryPrice)} → ${fmtPrice(p.markPrice)}</span>
    <span class="${upnlCls}">${fmtUsd(upnl)}</span>
  </div>`;
}

function renderWarnings(d) {
  const warnings = [];
  if (d.sentinel_present) {
    warnings.push(`STOPPED sentinel at <code>${d.sentinel_path}</code> — trader is halted until you clear it.`);
  }
  const state = d.state || {};
  const stuck = ((state.fill_events || {}).stuck_symbols) || [];
  if (stuck.length) {
    warnings.push(`stuck symbols (fill-event polling failing): <code>${stuck.join(', ')}</code>`);
  }
  const unknown = state.unknown_overlay || [];
  if (Array.isArray(unknown) && unknown.length) {
    warnings.push(`unknown overlay claims: <code>${JSON.stringify(unknown)}</code>`);
  }
  if (d.trader.state === 'crashed') {
    warnings.push(`trader process CRASHED (exit ${d.trader.exit_code}). Check the live log.`);
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
  if (ts - lastTs < 1500) return;  // throttle: at most one point per ~1.5s
  labels.push(ts);
  series.push(equity);
  if (labels.length > 720) { labels.shift(); series.shift(); }
  chart.update('none');
  const rangeEl = $('chart-range');
  if (rangeEl && labels.length >= 2) {
    const spanMs = labels[labels.length - 1] - labels[0];
    rangeEl.textContent = `last ${fmtSpan(spanMs)}`;
  }
}

function fmtSpan(ms) {
  const s = ms / 1000;
  if (s < 60) return `${Math.round(s)}s`;
  if (s < 3600) return `${Math.round(s / 60)}m`;
  if (s < 86400) return `${(s / 3600).toFixed(1)}h`;
  return `${(s / 86400).toFixed(1)}d`;
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
    appendLogLine('[ui] log stream disconnected, retrying…');
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
  // Trim if oversize.
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
    confirmBtn.textContent = 'killing…';
    await postControl('/api/control/kill');
    closeKillModal();
  });

  function openKillModal() {
    input.value = '';
    confirmBtn.disabled = true;
    confirmBtn.textContent = 'confirm kill';
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
  return n.toFixed(6);
}

// ─── boot ─────────────────────────────────────────────────────────

window.addEventListener('DOMContentLoaded', () => {
  initChart();
  attachControls();
  pollStatus();
  startLogStream();
});
