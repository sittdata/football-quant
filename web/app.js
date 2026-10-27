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

    try { updateKpis(data.metrics || {}); } catch (e) { console.error('updateKpis error:', e); }
    try { renderEquityChart(data.equity_curve || []); } catch (e) { console.error('renderEquityChart error:', e); }
    try { renderRecommendations(data.recommendations || []); } catch (e) { console.error('renderRecommendations error:', e); }
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
  if (maxDdEl) {
    const maxDd = m.max_drawdown !== undefined ? m.max_drawdown : 0;
    maxDdEl.innerHTML = `${maxDd.toFixed(2)} <span class="kpi-unit">Units</span>`;
  }

  const sharpeEl = document.getElementById('kpiSharpe');
  if (sharpeEl) {
    sharpeEl.innerText = `Sharpe: ${(m.sharpe_ratio || 0).toFixed(2)} | Sortino: ${(m.sortino_ratio || 0).toFixed(2)}`;
  }

  const totalBetsEl = document.getElementById('kpiTotalBets');
  if (totalBetsEl) {
    totalBetsEl.innerHTML = `${m.total_bets || 0} <span class="kpi-unit">Picks</span>`;
  }

  const openCountEl = document.getElementById('kpiOpenCount');
  if (openCountEl) {
    openCountEl.innerText = `${m.open_bets || 0} Open | ${m.settled_bets || 0} Settled`;
  }
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
          grid: { color: 'rgba(255, 255, 255, 0.04)' },
          ticks: {
            color: '#8e9db3',
            font: { family: 'JetBrains Mono', size: 10 },
            maxTicksLimit: 12
          }
        },
        y: {
          grid: { color: 'rgba(255, 255, 255, 0.04)' },
          ticks: {
            color: '#8e9db3',
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
    const baseStake = Number(rec.stake_units || 1.0);
    const synergyStake = Number(rec.synergy_stake || baseStake);

    let steamBadge = '';
    if (rec.steam) {
      const steamDesc = rec.steam.detail || rec.steam.description || 'Steam Signal';
      if (rec.steam.is_confirmed) {
        steamBadge = `<span class="badge badge-steam-confirmed" title="${escapeHtml(steamDesc)}">🔥 STEAM CONFIRMED</span>`;
      } else if (rec.steam.is_rlm) {
        steamBadge = `<span class="badge badge-steam-rlm" title="${escapeHtml(steamDesc)}">⚡ RLM ALERT</span>`;
      } else if (rec.steam.is_conflict) {
        steamBadge = `<span class="badge badge-steam-conflict" title="${escapeHtml(steamDesc)}">⚠️ STEAM CONFLICT</span>`;
      } else {
        steamBadge = `<span class="badge badge-steam-neutral" title="Line Movement Stable">STABLE</span>`;
      }
    } else if (rec.pick && rec.pick.includes('🔥')) {
      steamBadge = '<span class="badge badge-steam-confirmed">🔥 STEAM CONFIRMED</span>';
    } else if (rec.pick && rec.pick.includes('⚡')) {
      steamBadge = '<span class="badge badge-steam-rlm">⚡ RLM ALERT</span>';
    }

    let stakeHtml = `<div class="rec-stake">Kelly: ${baseStake.toFixed(2)}U</div>`;
    if (synergyStake > baseStake + 0.01) {
      stakeHtml = `<div class="rec-stake stake-boost" title="Steam Confirmed: +25% Stake Boost">Kelly: ${synergyStake.toFixed(2)}U <span style="font-size:10px;">(+25% 🔥)</span></div>`;
    } else if (synergyStake < baseStake - 0.01) {
      stakeHtml = `<div class="rec-stake stake-cut" title="Steam Conflict: -50% Defense Reduction">Kelly: ${synergyStake.toFixed(2)}U <span style="font-size:10px;">(-50% ⚠️)</span></div>`;
    }

    let xgHtml = '';
    if (rec.xg) {
      const hXg = Number(rec.xg.home_xg || 1.25).toFixed(2);
      const aXg = Number(rec.xg.away_xg || 1.25).toFixed(2);
      const diffVal = Number(rec.xg.xg_diff || 0.0);
      const diffStr = diffVal >= 0 ? `+${diffVal.toFixed(2)}` : diffVal.toFixed(2);
      const topScore = rec.xg.top_score || '1-1';
      const scoreProb = (Number(rec.xg.top_score_prob || 0.12) * 100).toFixed(0);

      const hPpda = Number(rec.xg.home_ppda || 11.0);
      let pressTag = '';
      if (hPpda < 9.0) {
        pressTag = '<span class="tactical-tag-press" title="High-Pressing Intensity">⚡ High-Press</span>';
      } else if (hPpda > 14.0) {
        pressTag = '<span class="tactical-tag-block" title="Low-Block Defensive Line">🛡️ Low-Block</span>';
      }

      let luckBadge = '';
      const hLuck = Number(rec.xg.home_luck || 0.0);
      if (hLuck > 0.6) {
        luckBadge = '<span class="badge-luck-warn" title="Overperforming xG: Unsustainable finishing variance">🎯 Luck Alert</span>';
      }

      xgHtml = `
        <div class="rec-xg-card">
          <div class="xg-dominance-row">
            <span>xG Dominance: <b>${hXg}</b> vs <b>${aXg}</b> <small style="color:var(--text-muted)">(${diffStr})</small></span>
            <span class="xg-score-tag">Sim: ${topScore} (${scoreProb}%)</span>
          </div>
          <div style="display:flex; gap:6px; align-items:center;">
            ${pressTag}
            ${luckBadge}
          </div>
        </div>
      `;
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
        ${xgHtml}
        <div class="rec-footer">
          <div class="rec-pick">
            <span class="badge ${rec.market === 'AH_dog' ? 'badge-open' : (rec.market === 'AH_fav' ? 'badge-steam' : 'badge-draw')}">${escapeHtml(rec.pick)}</span>
            <span>@ ${Number(rec.odds).toFixed(2)}</span>
            <span style="font-size: 11px; color: var(--text-muted);">(${probVal}% win)</span>
          </div>
          ${stakeHtml}
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
        <td style="font-family: var(--font-mono); font-size: 12px;">#${b.id}</td>
        <td style="font-family: var(--font-mono); font-size: 12px;">${kickoffStr}</td>
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
  btn.innerHTML = '<svg class="spin-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" width="14" height="14"><circle cx="12" cy="12" r="10" stroke-dasharray="32" stroke-dashoffset="12"/><path d="M12 2v4"/></svg> <span>Syncing Portfolio...</span>';

  try {
    // 1. Auto-settle any completed fixtures first
    await fetch('/api/settle', { method: 'POST' }).catch(() => {});
    // 2. Scan fresh odds & detect steam signals
    const res = await fetch('/api/scan', { method: 'POST' });
    const data = await res.json();
    await loadDashboard();
    await loadBets();
    const evaluated = data.evaluated || 0;
    const logged = data.logged || 0;
    showToast(`⚡ Sync complete: ${evaluated} fixtures priced, ${logged} actionable picks.`);
  } catch (err) {
    showToast(`❌ Sync error: ${err.message}`);
  } finally {
    btn.disabled = false;
    btn.innerHTML = originalHtml;
  }
}

function showToast(msg) {
  let toast = document.getElementById('quantToast');
  if (!toast) {
    toast = document.createElement('div');
    toast.id = 'quantToast';
    toast.style.cssText = 'position:fixed;bottom:24px;right:24px;background:#202020;border:1px solid #444;color:#fff;padding:10px 18px;border-radius:8px;font-family:var(--font-mono);font-size:12px;z-index:9999;box-shadow:0 6px 20px rgba(0,0,0,0.6);transition:opacity 0.3s ease;';
    document.body.appendChild(toast);
  }
  toast.innerText = msg;
  toast.style.opacity = '1';
  clearTimeout(window.__toastTimer);
  window.__toastTimer = setTimeout(() => { toast.style.opacity = '0'; }, 3500);
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

/* ══════════════════════════════════════════════════
   DUAL-MODEL TAB SYSTEM
══════════════════════════════════════════════════ */

function switchTab(model) {
  // Update tab buttons
  document.getElementById('tab-legacy').className = 'model-tab' + (model === 'legacy' ? ' active-legacy' : '');
  document.getElementById('tab-syndicate').className = 'model-tab' + (model === 'syndicate' ? ' active-syndicate' : '');

  // Show/hide panels
  document.getElementById('panel-legacy').classList.toggle('active', model === 'legacy');
  document.getElementById('panel-syndicate').classList.toggle('active', model === 'syndicate');

  // Auto-load syndicate data on first switch
  if (model === 'syndicate') {
    const list = document.getElementById('syndicateList');
    if (list && list.dataset.loaded !== 'true') {
      loadSyndicateData();
    }
  }
}

async function loadSyndicateData() {
  const btn = document.getElementById('btnScanSyn');
  const list = document.getElementById('syndicateList');
  if (btn) { btn.disabled = true; btn.textContent = '⏳ Scanning...'; }
  if (list) list.innerHTML = '<div style="text-align:center;color:#888;padding:36px 0;font-family:monospace;font-size:13px;">Connecting to Pinnacle + SofaScore feeds...</div>';

  try {
    const res = await fetch('/api/syndicate');
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();

    if (data.status === 'error') throw new Error(data.message);

    const signals = data.signals || [];
    const actionable = signals.filter(s => s.verdict && s.verdict !== 'NEUTRAL');

    // Update stats
    const el = (id) => document.getElementById(id);
    if (el('synScanned')) el('synScanned').textContent = data.scanned ?? signals.length;
    if (el('synActionable')) el('synActionable').textContent = data.actionable ?? actionable.length;
    if (el('synPickBadge')) el('synPickBadge').textContent = `${data.actionable ?? actionable.length} ALPHA`;

    if (!list) return;
    list.dataset.loaded = 'true';

    if (signals.length === 0) {
      list.innerHTML = '<div style="text-align:center;color:#888;padding:36px 0;font-family:monospace;">No fixtures available right now. Try again later.</div>';
      return;
    }

    // Sort: actionable first, then neutral
    const sorted = [...actionable, ...signals.filter(s => !s.verdict || s.verdict === 'NEUTRAL')];

    list.innerHTML = sorted.map(s => {
      const isAlpha = s.verdict && s.verdict !== 'NEUTRAL';
      const vig = s.market_vig_percent != null ? s.market_vig_percent.toFixed(1) : '–';
      const z = s.shin_insider_z != null ? (s.shin_insider_z * 100).toFixed(1) : '–';
      const edge = s.edge_percent != null ? s.edge_percent.toFixed(1) : '–';
      const kelly = s.quarter_kelly_stake_percent != null ? s.quarter_kelly_stake_percent.toFixed(2) : '–';
      const homeXg = s.total_xg_home != null ? s.total_xg_home.toFixed(2) : '–';
      const awayXg = s.total_xg_away != null ? s.total_xg_away.toFixed(2) : '–';
      const status = s.is_live
        ? `<span style="color:#34d399;font-weight:700;">● LIVE ${s.current_minute ? s.current_minute + "'" : ''}</span>`
        : `<span style="color:#888;">○ UPCOMING</span>`;

      return `
        <div class="syndicate-card${isAlpha ? '' : ' neutral'}">
          <div class="syndicate-header">
            <div>
              <div class="syndicate-match">${escapeHtml(s.home_team)} <span style="color:#555;">vs</span> ${escapeHtml(s.away_team)}</div>
              <div class="syndicate-league">${escapeHtml(s.league)} &nbsp;•&nbsp; ${status}</div>
            </div>
            ${isAlpha
              ? `<span class="badge-alpha">🔥 ALPHA SIGNAL</span>`
              : `<span class="badge-neutral-syn">NEUTRAL</span>`
            }
          </div>

          <div class="syndicate-metrics">
            <div><div class="syn-m-label">Vig (ค่าน้ำโต๊ะ)</div><div class="syn-m-value" style="${isAlpha ? '' : 'color:#666;'}">${vig}%</div></div>
            <div><div class="syn-m-label">Insider z (เงินโต๊ะใน)</div><div class="syn-m-value" style="${isAlpha ? '' : 'color:#666;'}">${z}%</div></div>
            <div><div class="syn-m-label">xG Home (สด)</div><div class="syn-m-value" style="color:#60a5fa;">${homeXg}</div></div>
            <div><div class="syn-m-label">xG Away (สด)</div><div class="syn-m-value" style="color:#f87171;">${awayXg}</div></div>
          </div>

          ${s.pinnacle_quoted_odds && s.pinnacle_quoted_odds.length >= 3 ? `
            <div style="margin-top:8px;font-size:11px;color:#888;font-family:monospace;background:rgba(0,0,0,0.2);padding:4px 8px;border-radius:4px;">
              📊 ราคาเปิด 1X2 Pinnacle: 
              เจ้าบ้าน <strong>${s.pinnacle_quoted_odds[0].toFixed(2)}</strong> | 
              เสมอ <strong>${s.pinnacle_quoted_odds[1].toFixed(2)}</strong> | 
              เยือน <strong>${s.pinnacle_quoted_odds[2].toFixed(2)}</strong>
            </div>
          ` : ''}

          ${isAlpha ? `
            <div style="background:rgba(245,158,11,0.18);border:1.5px solid #f59e0b;border-radius:8px;padding:12px 14px;margin-top:12px;">
              <div style="font-size:11px;color:#f59e0b;font-weight:800;letter-spacing:0.05em;text-transform:uppercase;margin-bottom:4px;">
                🎯 คำแนะนำการแทง (SYNDICATE PICK)
              </div>
              <div style="font-size:15px;font-weight:800;color:#fff;font-family:'JetBrains Mono',monospace;">
                แทงฝั่ง: <span style="color:#fcd34d;background:rgba(245,158,11,0.25);padding:2px 8px;border-radius:4px;">${escapeHtml(s.recommended_market || '')}</span>
              </div>
              <div style="margin-top:6px;font-size:12px;color:#bbb;font-family:monospace;">
                Edge (ความได้เปรียบ): <strong style="color:#34d399;">+${edge}%</strong> &nbsp;|&nbsp;
                Kelly Stake: <strong style="color:#60a5fa;">${kelly}% ของทุน</strong>
              </div>
            </div>
            <div class="syn-rationale" style="margin-top:8px;color:#aaa;">💡 <strong>เหตุผล:</strong> ${escapeHtml(s.rationale || '')}</div>
          ` : `
            <div style="margin-top:8px;padding:6px 10px;background:rgba(255,255,255,0.03);border-radius:6px;font-size:11px;color:#777;font-family:monospace;">
              ⛔ <strong>สถานะ: ข้ามคู่นี้ (NO BET)</strong> — ตลาดตั้งราคาได้สมดุลแล้ว ไม่มีแต้มต่อทางคณิตศาสตร์
            </div>
          `}
        </div>
      `;
    }).join('');

  } catch (err) {
    console.error('Syndicate load error:', err);
    if (list) list.innerHTML = `<div style="text-align:center;color:#f87171;padding:36px 0;font-family:monospace;">Error: ${escapeHtml(err.message)}</div>`;
  } finally {
    if (btn) { btn.disabled = false; btn.textContent = '⚡ REFRESH'; }
  }
}

