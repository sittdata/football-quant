/**
 * Cook45 Quantitative Hedge Fund Dashboard Client Logic
 * Real-time Chart.js Equity Curve, KPI Stream & Interactive Ledger
 */

let equityChart = null;
let allBets = [];
let currentFilter = 'ALL';

document.addEventListener('DOMContentLoaded', () => {
  initDashboard();
  // Auto refresh every 30 seconds
  setInterval(loadDashboard, 30000);
});

async function initDashboard() {
  await loadDashboard();
  await loadBets();
}

/**
 * Fetch and render core KPIs & Equity Curve
 */
async function loadDashboard() {
  const timestampEl = document.getElementById('headerTimestamp');
  if (timestampEl) {
    const now = new Date();
    timestampEl.innerText = `SYNCED • ${now.toLocaleTimeString()}`;
  }

  try {
    const res = await fetch('/api/dashboard');
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();

    updateKpis(data.metrics || {});
    renderEquityChart(data.equity_curve || []);
    renderRecommendations(data.recommendations || []);
  } catch (err) {
    console.error('Failed to load dashboard metrics:', err);
  }
}

/**
 * Update top KPI cards
 */
function updateKpis(m) {
  const pnlEl = document.getElementById('kpiPnl');
  const pnlVal = m.net_pnl !== undefined ? m.net_pnl : 0;
  const pnlSign = pnlVal >= 0 ? '+' : '';
  pnlEl.innerHTML = `${pnlSign}${pnlVal.toFixed(2)} <span class="kpi-unit">Units</span>`;
  pnlEl.className = `kpi-value ${pnlVal >= 0 ? 'text-emerald' : 'text-rose'}`;

  const pnlTrend = document.getElementById('kpiPnlTrend');
  if (pnlTrend) {
    pnlTrend.innerText = pnlVal >= 0 ? '▲ Active Realized Alpha' : '▼ Temporary Drawdown';
    pnlTrend.className = `kpi-trend ${pnlVal >= 0 ? 'text-emerald' : 'text-rose'}`;
  }

  const winRateEl = document.getElementById('kpiWinRate');
  const winRate = m.win_rate !== undefined ? m.win_rate : 0;
  winRateEl.innerHTML = `${(winRate * 100).toFixed(1)}<span class="kpi-unit">%</span>`;

  const recordEl = document.getElementById('kpiRecord');
  recordEl.innerText = `${m.wins || 0}W - ${m.draws || 0}D - ${m.losses || 0}L`;

  const roiEl = document.getElementById('kpiRoi');
  const roi = m.roi !== undefined ? m.roi : 0;
  const roiSign = roi >= 0 ? '+' : '';
  roiEl.innerHTML = `${roiSign}${(roi * 100).toFixed(1)}<span class="kpi-unit">%</span>`;
  roiEl.className = `kpi-value ${roi >= 0 ? 'text-gold' : 'text-rose'}`;

  const pfEl = document.getElementById('kpiProfitFactor');
  const pf = m.profit_factor !== undefined ? m.profit_factor : 0;
  pfEl.innerText = `Profit Factor: ${pf > 50 ? '∞' : pf.toFixed(2)}`;

  const maxDdEl = document.getElementById('kpiMaxDd');
  const maxDd = m.max_drawdown !== undefined ? m.max_drawdown : 0;
  maxDdEl.innerHTML = `${maxDd.toFixed(2)} <span class="kpi-unit">Units</span>`;

  const sharpeEl = document.getElementById('kpiSharpe');
  sharpeEl.innerText = `Sharpe: ${(m.sharpe_ratio || 0).toFixed(2)} | Sortino: ${(m.sortino_ratio || 0).toFixed(2)}`;

  const totalBetsEl = document.getElementById('kpiTotalBets');
  totalBetsEl.innerHTML = `${m.total_bets || 0} <span class="kpi-unit">Picks</span>`;

  const openCountEl = document.getElementById('kpiOpenCount');
  openCountEl.innerText = `${m.open_bets || 0} Open | ${m.settled_bets || 0} Settled`;
}

/**
 * Render High-Performance Chart.js Equity Trajectory
 */
function renderEquityChart(curveData) {
  const ctx = document.getElementById('equityChart');
  if (!ctx) return;

  // Default curve if empty
  if (!curveData || curveData.length === 0) {
    curveData = [
      { step: 0, pnl: 0, label: 'Baseline' }
    ];
  }

  const labels = curveData.map((d, i) => d.label || `#${i}`);
  const values = curveData.map(d => d.pnl);

  const isNetPositive = values[values.length - 1] >= 0;
  const strokeColor = isNetPositive ? '#10b981' : '#f43f5e';
  const fillGradient = ctx.getContext('2d').createLinearGradient(0, 0, 0, 280);
  fillGradient.addColorStop(0, isNetPositive ? 'rgba(16, 185, 129, 0.25)' : 'rgba(244, 63, 94, 0.25)');
  fillGradient.addColorStop(1, 'rgba(16, 185, 129, 0.0)');

  if (equityChart) {
    equityChart.data.labels = labels;
    equityChart.data.datasets[0].data = values;
    equityChart.data.datasets[0].borderColor = strokeColor;
    equityChart.data.datasets[0].backgroundColor = fillGradient;
    equityChart.update();
    return;
  }

  equityChart = new Chart(ctx, {
    type: 'line',
    data: {
      labels: labels,
      datasets: [{
        label: 'Cumulative P&L (Units)',
        data: values,
        borderColor: strokeColor,
        backgroundColor: fillGradient,
        fill: true,
        borderWidth: 2.5,
        tension: 0.25,
        pointRadius: values.length > 50 ? 0 : 3,
        pointHoverRadius: 6,
        pointBackgroundColor: strokeColor,
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { display: false },
        tooltip: {
          backgroundColor: 'rgba(15, 23, 42, 0.95)',
          titleFont: { family: 'Outfit', size: 13 },
          bodyFont: { family: 'JetBrains Mono', size: 13 },
          borderColor: 'rgba(255, 255, 255, 0.1)',
          borderWidth: 1,
          padding: 12,
          callbacks: {
            label: function(context) {
              const val = context.parsed.y;
              return ` Cumulative P&L: ${val >= 0 ? '+' : ''}${val.toFixed(2)} Units`;
            }
          }
        }
      },
      scales: {
        x: {
          grid: { color: 'rgba(255, 255, 255, 0.03)' },
          ticks: {
            color: '#64748b',
            font: { family: 'JetBrains Mono', size: 10 },
            maxTicksLimit: 12
          }
        },
        y: {
          grid: { color: 'rgba(255, 255, 255, 0.04)' },
          ticks: {
            color: '#64748b',
            font: { family: 'JetBrains Mono', size: 11 },
            callback: function(value) {
              return (value >= 0 ? '+' : '') + value.toFixed(1) + ' U';
            }
          }
        }
      }
    }
  });
}

/**
 * Render Recommended Alpha Fixtures list
 */
function renderRecommendations(recs) {
  const container = document.getElementById('recommendationsList');
  const countBadge = document.getElementById('openPicksCount');
  if (!container) return;

  if (countBadge) {
    countBadge.innerText = `${recs.length} PICKS`;
  }

  if (recs.length === 0) {
    container.innerHTML = `
      <div style="text-align: center; color: var(--text-muted); padding: 50px 0; font-size: 13px;">
        <span>🎯 No live high-edge fixtures currently pending.<br>Click <b>Scan Fixtures</b> to ingest upcoming matches.</span>
      </div>
    `;
    return;
  }

  container.innerHTML = recs.map(rec => {
    const edgeVal = (rec.edge * 100).toFixed(1);
    const probVal = (rec.predicted_win_rate * 100).toFixed(1);
    const isSteam = rec.pick && rec.pick.includes('🔥');
    const isRlm = rec.pick && rec.pick.includes('⚡');
    
    let steamBadge = '';
    if (isSteam) {
      steamBadge = '<span class="badge badge-steam" title="Aligned with Sharp Money Movement">🔥 STEAM</span>';
    } else if (isRlm) {
      steamBadge = '<span class="badge badge-rlm" title="Reverse Line Movement Detected">⚡ RLM</span>';
    }

    return `
      <div class="rec-item">
        <div class="rec-header">
          <div style="display: flex; align-items: center; gap: 6px;">
            <span class="rec-league">${escapeHtml(rec.league || 'LEAGUE')}</span>
            ${steamBadge}
          </div>
          <span class="rec-edge-badge">+${edgeVal}% Edge</span>
        </div>
        <div class="rec-match">${escapeHtml(rec.home_team)} <span style="color:var(--text-muted)">vs</span> ${escapeHtml(rec.away_team)}</div>
        <div class="rec-footer">
          <div class="rec-pick">
            <span class="badge ${rec.market === 'AH_dog' ? 'badge-open' : (rec.market === 'AH_fav' ? 'badge-steam' : 'badge-draw')}">${escapeHtml(rec.pick)}</span>
            <span>@ ${Number(rec.odds).toFixed(2)}</span>
            <span style="font-size: 11px; color: var(--text-muted);">(${probVal}% win)</span>
          </div>
          <div class="rec-stake">Kelly: ${Number(rec.stake_units || 1.0).toFixed(2)}U</div>
        </div>
      </div>
    `;
  }).join('');

}

/**
 * Fetch and render all paper ledger bets
 */
async function loadBets() {
  try {
    const res = await fetch('/api/bets');
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    allBets = await res.json();
    renderLedgerTable();
  } catch (err) {
    console.error('Failed to load ledger bets:', err);
  }
}

/**
 * Filter ledger by tab (ALL, OPEN, SETTLED)
 */
function filterLedger(status) {
  currentFilter = status;
  renderLedgerTable();
}

/**
 * Render the HTML Ledger Table
 */
function renderLedgerTable() {
  const tbody = document.getElementById('ledgerTableBody');
  if (!tbody) return;

  let filtered = allBets;
  if (currentFilter === 'OPEN') {
    filtered = allBets.filter(b => b.status === 'OPEN');
  } else if (currentFilter === 'SETTLED') {
    filtered = allBets.filter(b => b.status !== 'OPEN');
  }

  if (filtered.length === 0) {
    tbody.innerHTML = `
      <tr>
        <td colspan="13" style="text-align: center; color: var(--text-muted); padding: 30px;">
          No paper trading records found for filter: [${currentFilter}]
        </td>
      </tr>
    `;
    return;
  }

  tbody.innerHTML = filtered.map(b => {
    let statusBadge = '<span class="badge badge-open">OPEN</span>';
    let pnlText = '<span style="color: var(--text-muted)">-</span>';
    
    if (b.status === 'WON' || b.status === 'HALF_WON') {
      statusBadge = `<span class="badge badge-won">${b.status}</span>`;
      pnlText = `<span class="text-emerald" style="font-weight:700; font-family:var(--font-mono);">+${Number(b.profit_loss).toFixed(2)} U</span>`;
    } else if (b.status === 'LOST' || b.status === 'HALF_LOST') {
      statusBadge = `<span class="badge badge-lost">${b.status}</span>`;
      pnlText = `<span class="text-rose" style="font-weight:700; font-family:var(--font-mono);">${Number(b.profit_loss).toFixed(2)} U</span>`;
    } else if (b.status === 'PUSH' || b.status === 'DRAW') {
      statusBadge = `<span class="badge badge-draw">${b.status}</span>`;
      pnlText = `<span style="font-family:var(--font-mono);">0.00 U</span>`;
    }

    const kickoffStr = b.kickoff_time ? b.kickoff_time.replace('T', ' ').slice(0, 16) : '-';
    const scoreStr = (b.home_goals !== null && b.away_goals !== null) ? `${b.home_goals} - ${b.away_goals}` : (b.actual_score || '-');
    const edgeVal = b.edge ? `+${(Number(b.edge) * 100).toFixed(1)}%` : '-';
    const probVal = b.predicted_win_rate ? `${(Number(b.predicted_win_rate) * 100).toFixed(1)}%` : '-';
    const stakeVal = `${Number(b.stake_units || 1.0).toFixed(2)}U`;

    return `
      <tr>
        <td style="font-family: var(--font-mono); font-size: 11px;">#${b.id}</td>
        <td style="font-family: var(--font-mono); font-size: 11px;">${kickoffStr}</td>
        <td><span class="rec-league">${escapeHtml(b.league || 'LEAGUE')}</span></td>
        <td style="font-weight: 600;">${escapeHtml(b.home_team)} <span style="color:var(--text-muted)">vs</span> ${escapeHtml(b.away_team)}</td>
        <td><span class="badge badge-draw">${escapeHtml(b.market_type || 'AH')}</span></td>
        <td><b>${escapeHtml(b.pick)}</b></td>
        <td style="font-family: var(--font-mono);">${Number(b.odds).toFixed(2)}</td>
        <td style="font-family: var(--font-mono);">${probVal}</td>
        <td style="font-family: var(--font-mono); color: var(--accent-emerald);">${edgeVal}</td>
        <td style="font-family: var(--font-mono); color: var(--accent-cyan);">${stakeVal}</td>
        <td style="font-family: var(--font-mono); font-weight: 700;">${scoreStr}</td>
        <td>${statusBadge}</td>
        <td>${pnlText}</td>
      </tr>
    `;
  }).join('');
}

/**
 * Trigger backend fixture scan
 */
async function triggerScan() {
  const btn = document.getElementById('btnScan');
  const originalHtml = btn.innerHTML;
  btn.disabled = true;
  btn.innerHTML = '<span>⏳</span> Scanning 12 Leagues...';

  try {
    const res = await fetch('/api/scan', { method: 'POST' });
    const data = await res.json();
    alert(`Scan Completed!\nEvaluated: ${data.evaluated || 0} fixtures\nNew Picks Logged: ${data.logged || 0}`);
    await loadDashboard();
    await loadBets();
  } catch (err) {
    alert(`Scan error: ${err.message}`);
  } finally {
    btn.disabled = false;
    btn.innerHTML = originalHtml;
  }
}

/**
 * Trigger backend auto-settle feed
 */
async function triggerSettle() {
  const btn = document.getElementById('btnSettle');
  const originalHtml = btn.innerHTML;
  btn.disabled = true;
  btn.innerHTML = '<span>⚡</span> Settling Live Results...';

  try {
    const res = await fetch('/api/settle', { method: 'POST' });
    const data = await res.json();
    alert(`Auto-Settle Finished!\nMatches Synced: ${data.matches_synced || 0}\nBets Settled: ${data.bets_settled || 0}`);
    await loadDashboard();
    await loadBets();
  } catch (err) {
    alert(`Settle error: ${err.message}`);
  } finally {
    btn.disabled = false;
    btn.innerHTML = originalHtml;
  }
}

/**
 * Escape HTML utility
 */
function escapeHtml(str) {
  if (!str) return '';
  return String(str)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#039;');
}
