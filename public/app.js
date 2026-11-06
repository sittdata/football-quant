// State
let modelData = null;
let teams = {};

// DOM Elements
const leagueFilter = document.getElementById('leagueFilter');
const homeTeamSelect = document.getElementById('homeTeamSelect');
const awayTeamSelect = document.getElementById('awayTeamSelect');
const homeLeagueBadge = document.getElementById('homeLeagueBadge');
const awayLeagueBadge = document.getElementById('awayLeagueBadge');

const homeXgInput = document.getElementById('homeXgInput');
const homeXgaInput = document.getElementById('homeXgaInput');
const awayXgInput = document.getElementById('awayXgInput');
const awayXgaInput = document.getElementById('awayXgaInput');

const homeXgVal = document.getElementById('homeXgVal');
const homeXgaVal = document.getElementById('homeXgaVal');
const awayXgVal = document.getElementById('awayXgVal');
const awayXgaVal = document.getElementById('awayXgaVal');

const handicapSlider = document.getElementById('handicapSlider');
const handicapDisplay = document.getElementById('handicapDisplay');
const handicapLabel = document.getElementById('handicapLabel');

const oddsSlider = document.getElementById('oddsSlider');
const oddsDisplay = document.getElementById('oddsDisplay');
const oddsLabel = document.getElementById('oddsLabel');

const predictBtn = document.getElementById('predictBtn');

// Result Elements
const winnerBadge = document.getElementById('winnerBadge');
const probHomeName = document.getElementById('probHomeName');
const probAwayName = document.getElementById('probAwayName');
const probHomeVal = document.getElementById('probHomeVal');
const probDrawVal = document.getElementById('probDrawVal');
const probAwayVal = document.getElementById('probAwayVal');

const barHome = document.getElementById('barHome');
const barDraw = document.getElementById('barDraw');
const barAway = document.getElementById('barAway');

const cardHomeProb = document.getElementById('cardHomeProb');
const cardDrawProb = document.getElementById('cardDrawProb');
const cardAwayProb = document.getElementById('cardAwayProb');
const cardHomeTeam = document.getElementById('cardHomeTeam');
const cardAwayTeam = document.getElementById('cardAwayTeam');

// Load model data
async function init() {
  try {
    const res = await fetch('model_data.json');
    modelData = await res.json();
    teams = modelData.teams;
    populateTeams('ALL');
    setupEvents();
    runPrediction();
  } catch (err) {
    console.error("Error loading model_data.json:", err);
  }
}

// Populate team dropdowns
function populateTeams(selectedLeague) {
  const teamNames = Object.keys(teams).sort();
  homeTeamSelect.innerHTML = '';
  awayTeamSelect.innerHTML = '';

  teamNames.forEach(t => {
    const tData = teams[t];
    if (selectedLeague === 'ALL' || tData.league === selectedLeague) {
      const optH = new Option(`${t} (${tData.league.split(' ')[0]})`, t);
      const optA = new Option(`${t} (${tData.league.split(' ')[0]})`, t);
      homeTeamSelect.add(optH);
      awayTeamSelect.add(optA);
    }
  });

  // Default selections
  if (teams['Arsenal']) homeTeamSelect.value = 'Arsenal';
  if (teams['Chelsea']) awayTeamSelect.value = 'Chelsea';

  updateTeamStats();
}

function updateTeamStats() {
  const hName = homeTeamSelect.value;
  const aName = awayTeamSelect.value;

  if (teams[hName]) {
    const h = teams[hName];
    homeXgInput.value = h.xg_for;
    homeXgaInput.value = h.xg_against;
    homeXgVal.textContent = h.xg_for;
    homeXgaVal.textContent = h.xg_against;
    homeLeagueBadge.textContent = h.league;
  }

  if (teams[aName]) {
    const a = teams[aName];
    awayXgInput.value = a.xg_for;
    awayXgaInput.value = a.xg_against;
    awayXgVal.textContent = a.xg_for;
    awayXgaVal.textContent = a.xg_against;
    awayLeagueBadge.textContent = a.league;
  }

  cardHomeTeam.textContent = hName;
  cardAwayTeam.textContent = aName;
  probHomeName.textContent = hName;
  probAwayName.textContent = aName;
}

function setupEvents() {
  leagueFilter.addEventListener('change', e => {
    populateTeams(e.target.value);
    runPrediction();
  });

  homeTeamSelect.addEventListener('change', () => {
    updateTeamStats();
    runPrediction();
  });

  awayTeamSelect.addEventListener('change', () => {
    updateTeamStats();
    runPrediction();
  });

  [homeXgInput, homeXgaInput, awayXgInput, awayXgaInput].forEach(inp => {
    inp.addEventListener('input', () => {
      homeXgVal.textContent = homeXgInput.value;
      homeXgaVal.textContent = homeXgaInput.value;
      awayXgVal.textContent = awayXgInput.value;
      awayXgaVal.textContent = awayXgaInput.value;
      runPrediction();
    });
  });

  handicapSlider.addEventListener('input', e => {
    const val = parseFloat(e.target.value).toFixed(2);
    handicapDisplay.textContent = val;
    handicapLabel.textContent = val;
    runPrediction();
  });

  oddsSlider.addEventListener('input', e => {
    const val = parseFloat(e.target.value).toFixed(2);
    oddsDisplay.textContent = val;
    oddsLabel.textContent = val;
    runPrediction();
  });

  predictBtn.addEventListener('click', runPrediction);
}

// Tree evaluation helper
function evalNode(node, feats) {
  if (node.l !== undefined) return node.l;
  const val = feats[node.s];
  const target = val < node.c ? node.y : node.n;
  if (node.ch && node.ch[target]) {
    return evalNode(node.ch[target], feats);
  }
  return 0.0;
}

// Predict function using exact XGBoost tree evaluations
function runPrediction() {
  if (!modelData) return;

  const rawFeatures = [
    parseFloat(homeXgInput.value) || 1.5,
    parseFloat(awayXgInput.value) || 1.2,
    parseFloat(homeXgaInput.value) || 1.2,
    parseFloat(awayXgaInput.value) || 1.5,
    parseFloat(handicapSlider.value) || 0.0,
    parseFloat(oddsSlider.value) || 1.90
  ];

  // Scale features using MinMaxScaler min/max from dataset
  const minV = modelData.scaler.data_min;
  const maxV = modelData.scaler.data_max;
  const scaled = rawFeatures.map((val, i) => {
    const denom = maxV[i] - minV[i];
    return denom === 0 ? 0.0 : (val - minV[i]) / denom;
  });

  // Calculate raw margins across 300 trees
  const margins = [...modelData.base_scores];
  modelData.trees.forEach((tree, idx) => {
    const cls = idx % 3;
    margins[cls] += evalNode(tree, scaled);
  });

  // Softmax
  const maxM = Math.max(...margins);
  const expM = margins.map(m => Math.exp(m - maxM));
  const sumExp = expM.reduce((a, b) => a + b, 0);
  const probs = expM.map(e => e / sumExp);

  const pAway = probs[0] * 100.0;
  const pDraw = probs[1] * 100.0;
  const pHome = probs[2] * 100.0;

  // Update UI elements
  probAwayVal.textContent = pAway.toFixed(1) + '%';
  probDrawVal.textContent = pDraw.toFixed(1) + '%';
  probHomeVal.textContent = pHome.toFixed(1) + '%';

  cardAwayProb.textContent = pAway.toFixed(1) + '%';
  cardDrawProb.textContent = pDraw.toFixed(1) + '%';
  cardHomeProb.textContent = pHome.toFixed(1) + '%';

  barAway.style.width = pAway.toFixed(1) + '%';
  barDraw.style.width = pDraw.toFixed(1) + '%';
  barHome.style.width = pHome.toFixed(1) + '%';

  const hName = homeTeamSelect.value;
  const aName = awayTeamSelect.value;

  if (pHome >= pAway && pHome >= pDraw) {
    winnerBadge.textContent = `🏆 ${hName} Win Predicted`;
    winnerBadge.className = 'inline-flex items-center px-3 py-1 rounded-full text-xs font-bold uppercase tracking-wider bg-emerald-500/10 border border-emerald-500/20 text-emerald-400';
  } else if (pAway >= pHome && pAway >= pDraw) {
    winnerBadge.textContent = `🏆 ${aName} Win Predicted`;
    winnerBadge.className = 'inline-flex items-center px-3 py-1 rounded-full text-xs font-bold uppercase tracking-wider bg-rose-500/10 border border-rose-500/20 text-rose-400';
  } else {
    winnerBadge.textContent = `🤝 Draw Predicted`;
    winnerBadge.className = 'inline-flex items-center px-3 py-1 rounded-full text-xs font-bold uppercase tracking-wider bg-amber-500/10 border border-amber-500/20 text-amber-400';
  }
}

// Start
init();
