/**
 * KUBET Quant Auto-Bet Bridge Content Script
 * Author: cook45
 */

(function () {
  // Prevent duplicate injection in the same frame
  if (window.__quant_autobet_injected) return;
  window.__quant_autobet_injected = true;

  const host = window.location.hostname.toLowerCase();
  if (!host.includes("jjsskktt") && !host.includes("sbo") && !host.includes("cuu") && !host.includes("hdfr") && !host.includes("heratven") && !host.includes("ku") && !host.includes("sport") && !host.includes("saba") && !host.includes("cmd") && !host.includes("bet")) {
    return;
  }

  console.log("[Cook45 Quant AI] Sportsbook Bridge initializing on:", window.location.href);

  // Config State
  const state = {
    connected: false,
    autoBetEnabled: true,
    dryRunMode: true, // Default to Safe Simulation mode
    stakeAmount: 50,
    ws: null,
  };

  // ── 1. Create Floating HUD ───────────────────────────────────────────────
  function createHUD() {
    if (document.getElementById("quant-hud-container")) return;

    const isTop = (window.top === window.self);
    const href = window.location.href.toLowerCase();
    const isSportsFrame = href.includes("sbomaind") || href.includes("jjsskktt") || href.includes("games.aspx") || href.includes("sports");

    // Inject in top window OR directly into the sports iframe/frame
    if (!isTop && !isSportsFrame) return;

    const container = document.body || document.documentElement;
    if (!container) return;

    const hud = document.createElement("div");
    hud.id = "quant-hud-container";
    hud.style.zIndex = "2147483647";
    hud.style.position = "fixed";
    hud.innerHTML = `
      <div id="quant-hud-header">
        <div id="quant-hud-title">
          <span id="quant-hud-status-badge"></span>
          <span>COOK45 QUANT AUTO-BET</span>
        </div>
        <span id="quant-hud-toggle-btn" style="cursor: pointer; font-size: 14px;">▼</span>
      </div>
      <div id="quant-hud-body">
        <div class="quant-hud-row">
          <span class="quant-hud-label">ระบบ Auto-Bet อัตโนมัติ:</span>
          <label class="quant-hud-switch">
            <input type="checkbox" id="quant-autobet-toggle" checked>
            <span class="quant-hud-slider"></span>
          </label>
        </div>
        <div class="quant-hud-row">
          <span class="quant-hud-label">โหมดทดสอบ (Dry-Run ปลอดภัย):</span>
          <label class="quant-hud-switch">
            <input type="checkbox" id="quant-dryrun-toggle" checked>
            <span class="quant-hud-slider"></span>
          </label>
        </div>
        <div class="quant-hud-row">
          <span class="quant-hud-label">เงินเดิมพันต่อไม้ (บาท):</span>
          <input type="number" id="quant-stake-input" class="quant-hud-input" value="50" min="10" max="5000" step="10">
        </div>
        <button id="quant-hud-test-btn">⚡ ทดสอบจำลองบิล (Test Celtic)</button>
        <button id="quant-hud-inspect-btn" style="width: 100%; background: #1e293b; border: 1px solid #38bdf8; color: #38bdf8; font-weight: bold; border-radius: 6px; padding: 6px; margin-top: 5px; cursor: pointer; font-size: 11px; transition: 0.2s;">🔍 สแกนชื่อทีม & Data Source (Deep Inspect)</button>
        <div id="quant-hud-log">
          <div class="quant-log-item signal">[System] รอเชื่อมต่อ Local Python Bridge...</div>
        </div>
      </div>
    `;

    container.appendChild(hud);

    // Event listeners
    const toggleHeader = document.getElementById("quant-hud-header");
    const body = document.getElementById("quant-hud-body");
    const toggleBtn = document.getElementById("quant-hud-toggle-btn");
    let isCollapsed = false;

    toggleHeader.addEventListener("click", (e) => {
      if (e.target.id === "quant-hud-toggle-btn" || e.target.closest("#quant-hud-header")) {
        isCollapsed = !isCollapsed;
        body.style.display = isCollapsed ? "none" : "flex";
        toggleBtn.textContent = isCollapsed ? "▲" : "▼";
      }
    });

    document.getElementById("quant-autobet-toggle").addEventListener("change", (e) => {
      state.autoBetEnabled = e.target.checked;
      addLog(`Auto-Bet: ${state.autoBetEnabled ? "เปิดใช้งาน (ON)" : "ปิดชั่วคราว (OFF)"}`, state.autoBetEnabled ? "success" : "warning");
    });

    document.getElementById("quant-dryrun-toggle").addEventListener("change", (e) => {
      state.dryRunMode = e.target.checked;
      addLog(`โหมดทดสอบ: ${state.dryRunMode ? "เปิด (จำลองคลิก ไม่หักเงิน)" : "ปิด (ยิงบิลเงินจริง!)"}`, state.dryRunMode ? "warning" : "error");
    });

    document.getElementById("quant-stake-input").addEventListener("change", (e) => {
      state.stakeAmount = parseInt(e.target.value, 10) || 50;
      addLog(`ตั้งค่าเงินเดิมพัน: ${state.stakeAmount} บาท`, "success");
    });

    document.getElementById("quant-hud-test-btn").addEventListener("click", () => {
      addLog("⚡ ส่งสัญญาณทดสอบจำลองบิล (เซลติก vs ฮิเบอร์เนียน)...", "signal");
      executeBetJob({
        match: "ฮิเบอร์เนียน vs เซลติก",
        home: "ฮิเบอร์เนียน",
        away: "เซลติก",
        pick: "ต่อ เซลติก (-1.00)",
        team_target: "เซลติก",
        amount: state.stakeAmount,
        test_mode: true
      });
    });

    document.getElementById("quant-hud-inspect-btn").addEventListener("click", () => {
      runDeepInspection();
    });
  }

  function runDeepInspection() {
    addLog("🔍 เริ่มการสแกนเจาะลึกโครงสร้าง KUBET...", "signal");
    const docs = getAllAccessibleDocuments();
    addLog(`📄 พบเอกสาร/เฟรมทั้งหมด: ${docs.length} เฟรม`, "signal");

    const foundThai = [];
    for (let i = 0; i < docs.length; i++) {
      const doc = docs[i];
      if (!doc || !doc.body) continue;

      const walker = doc.createTreeWalker(doc.body, NodeFilter.SHOW_TEXT, null, false);
      let node;
      while ((node = walker.nextNode())) {
        const text = (node.nodeValue || "").trim();
        // Look for Thai text between 3 and 25 characters
        if (/[\u0E00-\u0E7F]{3,}/.test(text) && text.length <= 25) {
          if (!foundThai.includes(text) && !text.includes("COOK45") && !text.includes("เดิมพัน") && !text.includes("ทดสอบ")) {
            foundThai.push(text);
          }
        }
      }
    }

    console.log("[Cook45 Inspect] Found Thai Strings on Page:", foundThai);

    if (foundThai.length > 0) {
      addLog(`🎯 พบข้อความภาษาไทย ${foundThai.length} รายการบนจอ!`, "success");
      const preview = foundThai.slice(0, 6).join(", ");
      addLog(`📋 ตัวอย่างชื่อที่ตรวจพบ: ${preview}`, "success");
    }

    // Send full DOM dump to Python Bridge so Agent can read every HTML element directly!
    try {
      const dumpPayload = {
        type: "DOM_DUMP",
        url: window.location.href,
        html: document.body.innerHTML
      };
      chrome.runtime.sendMessage(dumpPayload);
      addLog("📤 ส่งโครงสร้าง HTML (DOM Dump) ไปยัง Local Bridge เรียบร้อย!", "success");
    } catch (e) {
      addLog(`⚠️ ไม่สามารถส่ง DOM Dump: ${e.message}`, "warning");
    }

    // Inspect window globals
    const suspiciousKeys = Object.keys(window).filter(k => 
      /game|match|odd|sport|data|list|bbin|jz/i.test(k) && typeof window[k] !== "function"
    );
    if (suspiciousKeys.length > 0) {
      console.log("[Cook45 Inspect] Global Data Keys on window:", suspiciousKeys);
      addLog(`💾 พบตัวแปรข้อมูล Global: ${suspiciousKeys.slice(0, 4).join(", ")}`, "signal");
    }
  }

  function addLog(text, type = "") {
    const logBox = document.getElementById("quant-hud-log");
    if (!logBox) return;
    const time = new Date().toLocaleTimeString("th-TH", { hour12: false });
    const item = document.createElement("div");
    item.className = `quant-log-item ${type}`;
    item.textContent = `[${time}] ${text}`;
    logBox.appendChild(item);
    logBox.scrollTop = logBox.scrollHeight;
  }

  function updateStatus(connected) {
    state.connected = connected;
    const badge = document.getElementById("quant-hud-status-badge");
    if (badge) {
      if (connected) {
        badge.classList.add("connected");
      } else {
        badge.classList.remove("connected");
      }
    }
  }

  // ── 2. Chrome Extension Background Bridge (Zero Browser Popups!) ────────
  function initExtensionBridge() {
    // Check initial connection status from background service worker
    try {
      chrome.runtime.sendMessage({ type: "GET_STATUS" }, (res) => {
        if (chrome.runtime.lastError) return;
        if (res && res.connected) {
          updateStatus(true);
          addLog("🟢 เชื่อมต่อกับ Python Quant AI สำเร็จ!", "success");
        }
      });
    } catch (e) {}

    // Listen for messages forwarded by background.js
    chrome.runtime.onMessage.addListener((data) => {
      console.log("[Content Script] Message from Background:", data);

      if (data.type === "BRIDGE_STATUS") {
        updateStatus(data.connected);
        if (data.connected) {
          addLog("🟢 เชื่อมต่อกับ Python Quant AI สำเร็จ!", "success");
        } else {
          addLog("🔴 หลุดการเชื่อมต่อกับ Python AI", "error");
        }
      } else if (data.type === "BET_SIGNAL") {
        // Find if this frame has the match odds table
        const res = findKubetMatchAndOdds(data);
        if (!res || !res.success) {
          // Not in this frame, skip
          return;
        }

        // Prevent duplicate execution across frames within 10s
        const betKey = `${data.match}_${data.pick}`;
        if (window.__last_exec_key === betKey && (Date.now() - (window.__last_exec_time || 0) < 10000)) {
          return;
        }
        window.__last_exec_key = betKey;
        window.__last_exec_time = Date.now();

        addLog(`🚨 สัญญาณเข้า: ${data.match} | ${data.pick} (Edge +${data.edge}%)`, "signal");
        if (!state.autoBetEnabled) {
          addLog("⚠️ ข้ามบิลนี้: Auto-Bet ถูกปิดไว้", "warning");
          return;
        }
        executeBetJob(data);
      } else if (data.type === "CONFIG_UPDATE") {
        if (data.auto_bet !== undefined) {
          state.autoBetEnabled = data.auto_bet;
          const el = document.getElementById("quant-autobet-toggle");
          if (el) el.checked = state.autoBetEnabled;
        }
        if (data.dry_run !== undefined) {
          state.dryRunMode = data.dry_run;
          const el = document.getElementById("quant-dryrun-toggle");
          if (el) el.checked = state.dryRunMode;
        }
        if (data.amount !== undefined) {
          state.stakeAmount = data.amount;
          const el = document.getElementById("quant-stake-input");
          if (el) el.value = state.stakeAmount;
        }
      }
    });
  }

  // ── Smart Team & Match Fuzzy Matching Engine for KUBET ─────────────────
  function normalizeThai(str) {
    if (!str) return "";
    return str
      .toLowerCase()
      .replace(/<[^>]+>/g, "")
      .replace(/\([^\)]+\)/g, "") // strip (ญ), (U20), (N)
      .replace(/(ต่อ|รอง|เลือก|สูง|ต่ำ|over|under|ลูก|เสมอ)/gi, "")
      .replace(/(เอฟซี|เอฟ ซี|ยูไนเต็ด|สโมสร|คลับ|ซีตี้|fc|united|club|city)/gi, "")
      .replace(/[\u0E48-\u0E4C]/g, "") // ตัดวรรณยุกต์ไทยออกทั้งหมด (่ ้ ๊ ๋ ์) -> "ฮิเบอร์เนี่ยน" กลายเป็น "ฮิเบอร์เนียน" ตรงกับเว็บ 100%!
      .replace(/[0-9.\-_+\s()]/g, "")
      .trim();
  }

  // ── Multi-Frame & TreeWalker DOM Search Engine for KUBET ───────────────
  function getAllAccessibleDocuments() {
    const docs = [document];
    function extractFrames(w) {
      try {
        if (w.document && !docs.includes(w.document)) {
          docs.push(w.document);
        }
      } catch (e) {}
      try {
        for (let i = 0; i < w.frames.length; i++) {
          try {
            extractFrames(w.frames[i]);
          } catch (e) {}
        }
      } catch (e) {}
    }
    extractFrames(window);
    try {
      const iframes = document.querySelectorAll("iframe, frame");
      for (const f of iframes) {
        try {
          const doc = f.contentDocument || (f.contentWindow && f.contentWindow.document);
          if (doc && !docs.includes(doc)) {
            docs.push(doc);
          }
        } catch (e) {}
      }
    } catch (e) {}
    return docs;
  }

  function triggerRealClick(el) {
    if (!el) return;
    const target = el.querySelector("a, span, font, b, div, button, [onclick]") || el;
    const opts = { bubbles: true, cancelable: true, view: window };
    try { target.dispatchEvent(new PointerEvent("pointerdown", opts)); } catch (e) {}
    try { target.dispatchEvent(new MouseEvent("mousedown", opts)); } catch (e) {}
    try { target.dispatchEvent(new PointerEvent("pointerup", opts)); } catch (e) {}
    try { target.dispatchEvent(new MouseEvent("mouseup", opts)); } catch (e) {}
    try { target.dispatchEvent(new MouseEvent("click", opts)); } catch (e) {}
    if (typeof target.click === "function") {
      try { target.click(); } catch (e) {}
    }
    if (target !== el) {
      try { el.dispatchEvent(new MouseEvent("click", opts)); } catch (e) {}
      if (typeof el.click === "function") {
        try { el.click(); } catch (e) {}
      }
    }
    if (target.parentElement && typeof target.parentElement.click === "function") {
      try { target.parentElement.click(); } catch (e) {}
    }

    // CRITICAL FIX: If inline onclick handler exists, execute it in the Main Page Context
    const onclickCode = target.getAttribute("onclick") || el.getAttribute("onclick") || (target.parentElement && target.parentElement.getAttribute("onclick"));
    if (onclickCode) {
      try {
        const s = document.createElement("script");
        s.textContent = onclickCode;
        (document.head || document.documentElement).appendChild(s);
        s.remove();
      } catch (e) {}
    }
  }

  function findKubetMatchAndOdds(job) {
    const targetTeam = normalizeThai(job.team_target || "");
    const homeTarget = normalizeThai(job.home || "");
    const awayTarget = normalizeThai(job.away || "");
    const pickStr = (job.pick || "").toLowerCase();
    const isOU = pickStr.includes("สูง") || pickStr.includes("ต่ำ") || pickStr.includes("over") || pickStr.includes("under");
    const isOver = pickStr.includes("สูง") || pickStr.includes("over");
    const is1X2 = pickStr.includes("1x2");
    const isDraw = pickStr.includes("เสมอ") || (job.team_target || "").includes("เสมอ");

    console.log("[Kubet Matcher] Targets -> Target:", targetTeam, "Home:", homeTarget, "Away:", awayTarget, "Pick:", pickStr);

    const docs = getAllAccessibleDocuments();

    for (const doc of docs) {
      if (!doc || !doc.body) continue;

      const allRows = Array.from(doc.querySelectorAll("tr"));
      for (let rIdx = 0; rIdx < allRows.length; rIdx++) {
        const row = allRows[rIdx];
        if (row.closest("#quant-hud-container")) continue;

        const rowText = (row.textContent || "").trim();
        // Ignore search banners, headers, and tooltips
        if (
          rowText.includes("ผลการค้นหา") ||
          rowText.includes("การค้นหานี้พบ") ||
          rowText.includes("เปรียบเทียบราคา") ||
          rowText.includes("COOK45")
        ) {
          continue;
        }

        const normRow = normalizeThai(rowText);
        const hasTarget = targetTeam && normRow.includes(targetTeam);
        const hasHome = homeTarget && normRow.includes(homeTarget);
        const hasAway = awayTarget && normRow.includes(awayTarget);

        if (!hasTarget && !hasHome && !hasAway) continue;

        // Verify this row or neighboring row has decimal odds
        const cells = Array.from(row.querySelectorAll("td"));
        if (cells.length < 2) continue;

        const nextRow = row.nextElementSibling;
        const nextNextRow = nextRow ? nextRow.nextElementSibling : null;

        const hasDecimalOdds = cells.some((td) => /\d+\.\d{2}/.test(td.textContent));
        const nextHasOdds = nextRow && Array.from(nextRow.querySelectorAll("td")).some((td) => /\d+\.\d{2}/.test(td.textContent));

        if (!hasDecimalOdds && !nextHasOdds) continue;

        console.log("[Kubet Matcher] Found genuine Match Row:", rowText.slice(0, 50));

        // Determine which row belongs to the target team (Home vs Away)
        let targetRow = row;
        let isAway = false;

        if (targetTeam) {
          if (normRow.includes(targetTeam)) {
            targetRow = row;
          } else if (nextRow && normalizeThai(nextRow.textContent).includes(targetTeam)) {
            targetRow = nextRow;
            isAway = true;
          }
        } else if (awayTarget && normRow.includes(awayTarget)) {
          targetRow = row;
          isAway = true;
        } else if (nextRow && awayTarget && normalizeThai(nextRow.textContent).includes(awayTarget)) {
          targetRow = nextRow;
          isAway = true;
        }

        // Get all td cells with decimal odds inside the target row
        const targetCells = Array.from(targetRow.querySelectorAll("td"));
        const oddsCells = targetCells.filter((td) => /\d+\.\d{2}/.test(td.textContent));

        console.log(`[Kubet Matcher] TargetRow (isAway=${isAway}) found ${oddsCells.length} odds cells:`, oddsCells.map(c => c.textContent.trim()));

        if (oddsCells.length === 0) continue;

        let targetOddsCell = null;

        if (isOU) {
          // Over/Under
          if (isOver) {
            targetOddsCell = oddsCells.find((c) => /สูง|over/i.test(c.textContent)) || (oddsCells.length >= 2 ? oddsCells[1] : oddsCells[0]);
          } else {
            targetOddsCell = oddsCells.find((c) => /ต่ำ|under/i.test(c.textContent)) || (oddsCells.length >= 2 ? oddsCells[1] : oddsCells[0]);
          }
        } else if (is1X2) {
          if (isDraw) {
            const drawRow = isAway ? nextRow : nextNextRow;
            if (drawRow) {
              const dCells = Array.from(drawRow.querySelectorAll("td")).filter((td) => /\d+\.\d{2}/.test(td.textContent));
              targetOddsCell = dCells[0] || null;
            }
          } else {
            // 1X2 is typically the 3rd odds column (or last)
            targetOddsCell = oddsCells.length >= 3 ? oddsCells[2] : oddsCells[oddsCells.length - 1];
          }
        } else {
          // Asian Handicap (AH) is ALWAYS the 1st column with decimal odds!
          targetOddsCell = oddsCells[0];
        }

        if (targetOddsCell) {
          // Find the clickable button or decimal span inside the cell
          let clickableBtn = targetOddsCell.querySelector("a, span.odds, span.rate, [onclick], b, font");
          if (!clickableBtn) {
            const innerElements = Array.from(targetOddsCell.querySelectorAll("*"));
            clickableBtn = innerElements.find((el) => /\d+\.\d{2}/.test(el.textContent)) || targetOddsCell;
          }

          console.log("[Kubet Matcher] 🎯 SELECTED EXACT CELL:", targetOddsCell.textContent.trim(), clickableBtn);

          return {
            success: true,
            score: 1.0,
            tr: targetRow,
            teamEl: targetRow,
            matchedText: targetRow.textContent.trim().slice(0, 35),
            oddsCell: targetOddsCell,
            oddsBtn: clickableBtn || targetOddsCell,
            doc: doc,
            isFallback: false
          };
        }
      }
    }

    return { success: false, score: 0 };
  }

  async function autoSearchKubet(query) {
    if (!query) return false;
    const cleanQuery = query.replace(/(ต่อ|รอง|เลือก|สูง|ต่ำ|\(.*\))/g, "").trim();
    if (cleanQuery.length < 2) return false;

    const docs = getAllAccessibleDocuments();
    for (const d of docs) {
      const inputs = d.querySelectorAll("input[placeholder*='ค้นหา'], input[type='search'], input.search-input, #txtSearch");
      for (const inp of inputs) {
        if (inp.offsetParent !== null) { // visible
          inp.focus();
          inp.value = cleanQuery;
          inp.dispatchEvent(new Event("input", { bubbles: true }));
          inp.dispatchEvent(new Event("change", { bubbles: true }));

          // Look for search icon or button
          const btn = inp.parentElement ? inp.parentElement.querySelector("button, i, span, svg") : null;
          if (btn && btn !== inp) {
            triggerRealClick(btn);
          }
          inp.dispatchEvent(new KeyboardEvent("keydown", { key: "Enter", keyCode: 13, bubbles: true }));
          inp.dispatchEvent(new KeyboardEvent("keyup", { key: "Enter", keyCode: 13, bubbles: true }));
          await sleep(700);
          return true;
        }
      }
    }
    return false;
  }

  // ── 10 Ironclad Capital Protection Shields (ระบบเกราะป้องกันเงินทุน 10 ชั้น 100%) ──
  let consecutiveFailures = 0;
  const MAX_CONSECUTIVE_FAILURES = 3;
  const MAX_HARD_STAKE = 100; // ล็อกเพดานห้ามแทงเกิน 100 บาทในโหมดออโต้เด็ดขาด!
  let isJobExecuting = false; // Re-entrancy Mutex Lock

  // SHIELD 2: ระบบตรวจสอบการแทงซ้ำข้ามหน้า & ข้าม Session (Persistent Storage Guard)
  async function isBetDuplicate(betKey) {
    return new Promise((resolve) => {
      try {
        if (!chrome.storage || !chrome.storage.local) {
          resolve(false);
          return;
        }
        chrome.storage.local.get(["ku_executed_bets"], (res) => {
          if (chrome.runtime.lastError) {
            resolve(false);
            return;
          }
          const history = res.ku_executed_bets || {};
          const now = Date.now();
          const timestamp = history[betKey];
          // ล็อคห้ามแทงซ้ำคู่และราคานี้เป็นเวลา 4 ชั่วโมง (14,400,000 ms)
          if (timestamp && (now - timestamp < 14400000)) {
            resolve(true);
          } else {
            resolve(false);
          }
        });
      } catch (e) {
        resolve(false);
      }
    });
  }

  async function recordExecutedBet(betKey) {
    return new Promise((resolve) => {
      try {
        if (!chrome.storage || !chrome.storage.local) {
          resolve();
          return;
        }
        chrome.storage.local.get(["ku_executed_bets"], (res) => {
          const history = res.ku_executed_bets || {};
          history[betKey] = Date.now();
          chrome.storage.local.set({ ku_executed_bets: history }, () => resolve());
        });
      } catch (e) {
        resolve();
      }
    });
  }

  // SHIELD 3: ดักจับป๊อปอัปเตือนของ KUBET (Session Expired / Maintenance / Suspended)
  function checkKubetSystemModals() {
    const docs = getAllAccessibleDocuments();
    for (const d of docs) {
      const errorModals = d.querySelectorAll(".modal, .popup, .alert, .dialog, .msg-box, .swal2-container, div[class*='dialog'], div[class*='modal']");
      for (const m of errorModals) {
        if (m.offsetParent !== null) { // visible
          const txt = (m.textContent || "").toLowerCase();
          if (
            txt.includes("หมดเวลา") ||
            txt.includes("เข้าสู่ระบบ") ||
            txt.includes("login") ||
            txt.includes("ระงับ") ||
            txt.includes("ปิดรับ") ||
            txt.includes("ไม่สำเร็จ") ||
            txt.includes("เงินคงเหลือไม่พอ") ||
            txt.includes("เครดิตไม่พอ")
          ) {
            return txt.slice(0, 80);
          }
        }
      }
    }
    return null;
  }

  // SHIELD 4: ดักจับและยกเลิกป๊อปอัป "ราคาต่อรองมีการเปลี่ยนแปลง" (Reject Worse Odds Prompt)
  function checkAndRejectOddsChangePrompt() {
    const docs = getAllAccessibleDocuments();
    for (const d of docs) {
      const alerts = d.querySelectorAll(".dialog, .modal, .popup, .alert-box, .msg, .confirm-dialog, div[class*='dialog']");
      for (const a of alerts) {
        if (a.offsetParent !== null) {
          const t = (a.textContent || "").toLowerCase();
          if (t.includes("อัตราต่อรองเปลี่ยน") || t.includes("ราคาเปลี่ยนแปลง") || t.includes("odds change") || t.includes("price change")) {
            // Find cancel / reject button to prevent blindly accepting worse price
            const cancelBtn = a.querySelector("button.btn-cancel, button:not([class*='confirm']):not([class*='accept']), .btn-close, a.cancel");
            if (cancelBtn) triggerRealClick(cancelBtn);
            return t.slice(0, 80);
          }
        }
      }
    }
    return null;
  }

  // SHIELD 5: ตรวจจับโหมดเดิมพันด่วน (Quick Bet Guard) ในโหมดทดสอบ
  function checkKubetQuickBetEnabled() {
    const docs = getAllAccessibleDocuments();
    for (const d of docs) {
      const quickInputs = d.querySelectorAll("input[type=checkbox]#chkQuickBet, input[type=checkbox].quick-bet, input[name*='quick']");
      for (const q of quickInputs) {
        if (q.checked) return true;
      }
      // Check labels/buttons that indicate Quick Bet is ACTIVE
      const quickBadges = d.querySelectorAll(".quick-bet.active, .btn-quick.on, [data-quick='1']");
      if (quickBadges.length > 0) return true;
    }
    return false;
  }

  // SHIELD 6: ล้างบิลเก่าค้างในรถเข็น/สลิปเดิมพัน เพื่อป้องกันบิลกลายเป็นสเต็ป/พาร์เลย์ (Cart Isolation Guard)
  async function clearOldSelectionsInCart() {
    const docs = getAllAccessibleDocuments();
    let cleared = false;
    for (const d of docs) {
      const clearBtns = d.querySelectorAll(
        "button.btn-clear, a.btn-clear, .delete-all, .clear-all, [data-action='clear-all'], #btnClearAll, .icon-trash, i.fa-trash, .btn-del-all, a.delAll, #delAll, .cart-clear, .btn-delete-all"
      );
      for (const btn of clearBtns) {
        if (btn.offsetParent !== null) {
          triggerRealClick(btn);
          cleared = true;
          await sleep(300);
          break;
        }
      }
    }
    return cleared;
  }

  // ── 3. DOM Automation Logic for KUBET with 10 Capital Protection Shields ─
  async function executeBetJob(job) {
    // SHIELD 1: Re-entrancy & Concurrency Mutex Lock
    if (isJobExecuting) {
      addLog("⚠️ มีบิลเดิมพันกำลังดำเนินการอยู่ ป้องกัน Race Condition!", "warning");
      return;
    }
    isJobExecuting = true;

    try {
      // SHIELD 2: ป้องกันการแทงซ้ำคู่เดิมข้ามหน้า / ข้ามรีเฟรช (Storage & Memory Deduplication)
      const betKey = `${job.match || ""}_${job.pick || ""}`;
      if (!job.test_mode) {
        const isDup = await isBetDuplicate(betKey);
        if (isDup) {
          addLog(`🛡️ [เซฟทุน] ข้ามบิลนี้: คู่นี้เคยวางเดิมพันไปแล้วใน 4 ชม. ที่ผ่านมา ป้องกันแทงซ้ำ 100%!`, "warning");
          return;
        }
      }

      // SHIELD 3: สวิตช์ตัดไฟฉุกเฉิน (Circuit Breaker)
      if (consecutiveFailures >= MAX_CONSECUTIVE_FAILURES) {
        state.autoBetEnabled = false;
        const el = document.getElementById("quant-autobet-toggle");
        if (el) el.checked = false;
        addLog(`🚨 [ตัดไฟฉุกเฉิน] ตรวจพบข้อผิดพลาดติดกัน 3 ครั้ง ปิดระบบ Auto-Bet อัตโนมัติเพื่อเซฟเงิน!`, "error");
        return;
      }

      // SHIELD 4: ดักจับป๊อปอัปเตือนของ KUBET (Session Expired / Market Suspended)
      const modalAlert = checkKubetSystemModals();
      if (modalAlert) {
        addLog(`⚠️ [หยุดฉุกเฉิน] ตรวจพบป๊อปอัปเตือนจากเว็บ KUBET: "${modalAlert}" ป้องกันเงินรั่วไหล!`, "error");
        return;
      }

      // SHIELD 5: ล็อกเพดานเงินเดิมพันสูงสุด (Hard Stake Ceiling Protection <= 100 บาท)
      const safeAmount = Math.max(10, Math.min(job.amount || state.stakeAmount || 10, MAX_HARD_STAKE));
      job.amount = safeAmount;

      addLog(`🔍 กำลังจับคู่: ${job.team_target || job.match}... (ตรวจชื่อทีม & ราคา)`, "signal");

      let res = findKubetMatchAndOdds(job);

      // If not immediately visible on screen, use KUBET's search bar to filter directly to this match!
      if (!res.success) {
        const searched = await autoSearchKubet(job.team_target || job.match);
        if (searched) {
          addLog(`🔎 ค้นหาคู่ "${job.team_target || job.match}" ผ่านแถบค้นหา KUBET...`, "signal");
          await sleep(800);
          res = findKubetMatchAndOdds(job);
        }
      }

      if (!res.success) {
        // Collect what teams are actually on screen to show the user
        const visibleTeams = [];
        const docs = getAllAccessibleDocuments();
        for (const d of docs) {
          if (!d || !d.body) continue;
          const walker = d.createTreeWalker(d.body, NodeFilter.SHOW_TEXT, null, false);
          let n;
          while ((n = walker.nextNode())) {
            if (n.parentElement && n.parentElement.closest("#quant-hud-container")) continue;
            const t = (n.nodeValue || "").trim();
            if (/[\u0E00-\u0E7F]{3,}/.test(t) && t.length >= 3 && t.length <= 25) {
              if (!visibleTeams.includes(t) && !t.includes("เดิมพัน") && !t.includes("ทดสอบ") && !t.includes("ระบบ")) {
                visibleTeams.push(t);
              }
            }
          }
        }

        addLog(`❌ ไม่พบคู่ ${job.team_target || job.match} บนตารางหน้าจอ!`, "error");
        if (visibleTeams.length > 0) {
          addLog(`📋 ทีมที่พบบนหน้านี้ขณะนี้: ${visibleTeams.slice(0, 4).join(" | ")}`, "warning");
        } else {
          addLog(`⚠️ ไม่พบชื่อทีมใดๆ ใน Frame หลัก (อาจอยู่ใน Frame ย่อยของ KUBET)`, "warning");
        }
        return;
      }

      // SHIELD 6: In-Play / Live Match Lock (ห้ามแทงบอลสดเด็ดขาด!)
      const rowFullText = (res.tr.textContent || "").toLowerCase();
      const isLiveMatch = /กำลังแข่ง|สด|live|in-play|ครึ่งแรก|ครึ่งหลัง|1h|2h|\b\d{1,2}'/.test(rowFullText);
      if (isLiveMatch) {
        addLog(`🚨 [ยกเลิกเพื่อเซฟเงิน] คู่นี้เริ่มแข่งไปแล้ว (บอลสด Live)! โมเดล Quant ปฏิเสธการแทงบอลสด 100%!`, "error");
        return;
      }

      if (res.isFallback) {
        addLog(`ℹ️ เซลติกไม่อยู่บนจอ -> ล็อคเป้าคู่จริงบนจอ: "${res.matchedText}"!`, "signal");
      } else {
        addLog(`🎯 พบคู่บนตาราง: "${res.matchedText}" ในตารางจริง!`, "success");
      }

      // Flash neon blue border and bright highlight on the matched row and scroll to it
      res.tr.style.outline = "4px solid #00f2fe";
      res.tr.style.backgroundColor = "rgba(0, 242, 254, 0.25)";
      res.tr.style.boxShadow = "0 0 30px rgba(0, 242, 254, 1)";
      res.tr.scrollIntoView({ behavior: "smooth", block: "center" });

      await sleep(500);

      const clickTarget = res.oddsBtn || res.oddsCell;

      // SHIELD 7: Odds Slippage & Payout Verification (ตรวจจับราคาน้ำร่วงเกิน 12%)
      const oddsMatch = (clickTarget.textContent || "").match(/\d+\.\d{2}/);
      const liveOdds = oddsMatch ? parseFloat(oddsMatch[0]) : null;
      if (job.expected_odds && liveOdds) {
        const slippage = (job.expected_odds - liveOdds) / job.expected_odds;
        if (slippage > 0.12) {
          addLog(`🚨 [ยกเลิกบิลเพื่อเซฟเงิน] ราคาน้ำตกลงเกินเกณฑ์! คาดหวัง: ${job.expected_odds} แต่ราคาจริงบนจอเหลือ: ${liveOdds} (ร่วงลง ${(slippage * 100).toFixed(1)}%)`, "error");
          consecutiveFailures++;
          return;
        }
      }

      // SHIELD 8: Quick-Bet Guard in Safe Dry-Run Mode
      const isQuickBetActive = checkKubetQuickBetEnabled();
      if (state.dryRunMode && isQuickBetActive) {
        addLog(`⚠️ [เซฟตี้โหมดทดสอบ] เว็บ KUBET กำลังเปิดโหมด "เดิมพันด่วน"! จำลองการไฮไลท์ราคาโดยไม่คลิกจริง เพื่อป้องกันเงินจริงตัด!`, "warning");
        if (clickTarget) {
          clickTarget.style.outline = "4px solid #10b981";
          clickTarget.style.backgroundColor = "rgba(16, 185, 129, 0.4)";
        }
        await sleep(1000);
        addLog(`🧪 [DRY RUN ปลอดภัย] จำลองบิลสำเร็จ 100% (จำลองเงิน ${job.amount} บาท)`, "success");
        try {
          chrome.runtime.sendMessage({
            type: "BET_CONFIRMATION",
            status: "SUCCESS_DRY_RUN",
            match: job.match || res.matchedText,
            pick: job.pick || "ทดสอบ",
            amount: job.amount || state.stakeAmount,
            receipt_id: "DRY-" + Math.floor(Math.random() * 900000 + 100000)
          });
        } catch (e) {}
        return;
      }

      // Pre-clean cart of any old selections before clicking new odds
      await clearOldSelectionsInCart();

      // Click the EXACT odds button element with full synthetic pointer/mouse events
      if (clickTarget) {
        clickTarget.style.outline = "4px solid #10b981";
        clickTarget.style.backgroundColor = "rgba(16, 185, 129, 0.4)";
        clickTarget.style.boxShadow = "0 0 20px rgba(16, 185, 129, 0.9)";
        triggerRealClick(clickTarget);
        if (res.oddsCell && res.oddsCell !== clickTarget) {
          triggerRealClick(res.oddsCell);
        }
        addLog(`✅ สั่งคลิกช่องราคาต่อรอง (${clickTarget.textContent.trim() || "ราคา"}) เรียบร้อย!`, "success");
      }

      await sleep(1000);

      // Check if bookmaker showed an immediate "odds changed" prompt
      const oddsPrompt = checkAndRejectOddsChangePrompt();
      if (oddsPrompt) {
        addLog(`⚠️ [ยกเลิกบิล] ตรวจพบการเปลี่ยนราคาจากโต๊ะ KUBET ("${oddsPrompt}") ปฏิเสธราคาใหม่เพื่อเซฟทุน!`, "error");
        return;
      }

      // Helper: Find real stake input strictly outside the HUD
      function findRealStakeInput() {
        const docs = getAllAccessibleDocuments();
        for (const d of docs) {
          const inputs = d.querySelectorAll(
            "input#txtStake, input#stake, input.stake, input[name=stake], input.bet-amount, input.amount, input#betAmount, input[type=number], input[name=amount], input.input-amount, .stake-input, input[placeholder*='เดิมพัน'], input[placeholder*='จำนวน'], input.txt-stake, input"
          );
          for (const inp of inputs) {
            if (inp.closest("#quant-hud-container")) continue; // NEVER match HUD!
            if (inp.offsetParent === null) continue; // Must be visible
            const id = (inp.id || "").toLowerCase();
            const cls = (inp.className || "").toLowerCase();
            const name = (inp.name || "").toLowerCase();
            const holder = (inp.placeholder || "").toLowerCase();
            if (
              id.includes("stake") || id.includes("amount") || id.includes("bet") ||
              cls.includes("stake") || cls.includes("amount") || cls.includes("bet") ||
              name.includes("stake") || name.includes("amount") ||
              holder.includes("เดิมพัน") || holder.includes("จำนวน") ||
              inp.type === "number" || inp.type === "tel"
            ) {
              return inp;
            }
          }
        }
        return null;
      }

      // Helper: Find real submit button strictly outside the HUD
      function findRealSubmitBtn() {
        const docs = getAllAccessibleDocuments();
        for (const d of docs) {
          const btns = d.querySelectorAll(
            "button#btnBet, button#btnSubmit, input#btnBet, input#btnSubmit, button.btn-bet, button.btn-confirm, button#confirmBet, .submit-btn, [data-action=submit], .btn-submit, a.btn-bet, button[class*='bet'], button[class*='confirm'], [class*='submit'], button, a, div[role='button'], input[type='button'], input[type='submit']"
          );
          for (const b of btns) {
            if (b.closest("#quant-hud-container")) continue; // NEVER match HUD!
            if (b.offsetParent === null) continue;
            
            const txt = (b.textContent || b.value || "").trim();
            const id = (b.id || "").toLowerCase();
            const cls = (b.className || "").toLowerCase();
            if (
              txt === "วางเดิมพัน" || txt === "ยืนยันการเดิมพัน" || txt === "ยืนยันเดิมพัน" ||
              txt === "เดิมพัน" || txt === "Place Bet" || txt === "Bet Now" ||
              id === "btnbet" || id === "btnsubmit" || id === "confirmbet" ||
              cls.includes("btn-bet") || cls.includes("btn-submit") || cls.includes("btn-confirm")
            ) {
              return b;
            }
          }
        }
        return null;
      }

      // Look for the Bet Slip input box across all frames/documents
      let stakeInput = findRealStakeInput();

      // If stake input not visible, open the yellow shopping cart icon in top right!
      if (!stakeInput) {
        const cartIcons = document.querySelectorAll("a, button, div, span, i");
        for (const el of cartIcons) {
          if (el.closest("#quant-hud-container")) continue;
          const cls = (el.className || "").toLowerCase();
          const title = (el.getAttribute("title") || "").toLowerCase();
          if (
            cls.includes("cart") ||
            cls.includes("order") ||
            cls.includes("slip") ||
            title.includes("รถเข็น") ||
            title.includes("เดิมพัน")
          ) {
            triggerRealClick(el);
            await sleep(600);
            break;
          }
        }

        stakeInput = findRealStakeInput();
      }

      // SHIELD 9: Stake Input Value Sanitization & Native Property Setter
      if (stakeInput) {
        stakeInput.style.outline = "3px solid #facc15";
        stakeInput.style.backgroundColor = "#fffbeb";
        stakeInput.focus();
        stakeInput.select();

        try {
          // Native prototype setter bypasses React/framework value interception
          const nativeSetter = Object.getOwnPropertyDescriptor(window.HTMLInputElement.prototype, "value")?.set;
          if (nativeSetter) {
            nativeSetter.call(stakeInput, String(safeAmount));
          } else {
            stakeInput.value = safeAmount;
          }
        } catch (e) {
          stakeInput.value = safeAmount;
        }

        stakeInput.dispatchEvent(new Event("input", { bubbles: true }));
        stakeInput.dispatchEvent(new Event("change", { bubbles: true }));

        // Hard verification of input amount
        const parsedFinalVal = parseFloat(stakeInput.value);
        if (isNaN(parsedFinalVal) || parsedFinalVal !== safeAmount || parsedFinalVal > MAX_HARD_STAKE) {
          addLog(`🚨 [ยกเลิกฉุกเฉิน] ยอดเดิมพันในช่องสลิปผิดปกติ (${stakeInput.value}) ไม่ตรงกับ ${safeAmount} บาท หรือเกินเพดาน 100 บาท!`, "error");
          consecutiveFailures++;
          return;
        }

        addLog(`💵 กรอกยอดเงินเดิมพัน ${safeAmount} บาท บนสลิปและตรวจสอบความถูกต้องแล้ว!`, "success");
      } else {
        addLog(`⚠️ ยังไม่พบบิลในสลิปเดิมพัน KUBET! ตรวจสอบว่าช่องราคาตอบสนองการคลิกหรือไม่`, "warning");
        return;
      }

      await sleep(600);

      // Step D: Confirm button handling
      if (state.dryRunMode) {
        addLog(`🧪 [DRY RUN ปลอดภัย] เปิดราคาและกรอกเงินให้ดูแล้ว! (ถ้าจะให้กดยืนยันจริง ให้สับสวิตช์ปิดโหมดทดสอบ)`, "warning");
        consecutiveFailures = 0; // Reset failure counter on success
        try {
          chrome.runtime.sendMessage({
            type: "BET_CONFIRMATION",
            status: "SUCCESS_DRY_RUN",
            match: job.match || res.matchedText,
            pick: job.pick || "ทดสอบ",
            amount: safeAmount,
            receipt_id: "DRY-" + Math.floor(Math.random() * 900000 + 100000)
          });
        } catch (e) {}
        return;
      }

      // SHIELD 10: REAL MONEY BETTING (Dry Run is OFF -> Click Confirm Bet with Post-Check)
      const submitBtn = findRealSubmitBtn();

      if (submitBtn) {
        triggerRealClick(submitBtn);
        addLog(`🚀 [ยิงเงินจริง!] กดยืนยันการวางเดิมพัน ${safeAmount} บาท เรียบร้อย!`, "success");

        // Record in persistent storage so this match + pick can NEVER be bet again in 4 hours
        await recordExecutedBet(betKey);
        consecutiveFailures = 0;

        try {
          chrome.runtime.sendMessage({
            type: "BET_CONFIRMATION",
            status: "SUCCESS_REAL",
            match: job.match || res.matchedText,
            pick: job.pick || "แทงจริง",
            amount: safeAmount,
            receipt_id: "REAL-" + Date.now()
          });
        } catch (e) {}
      } else {
        addLog(`⚠️ ไม่พบปุ่มกดยืนยันเดิมพันของ KUBET ในสลิป`, "error");
      }

    } catch (err) {
      consecutiveFailures++;
      addLog(`❌ เกิดข้อผิดพลาด: ${err.message}`, "error");
    } finally {
      isJobExecuting = false;
    }
  }

  function sleep(ms) {
    return new Promise((resolve) => setTimeout(resolve, ms));
  }

  // ── 4. Bootstrapping (Dynamic SPA & Frameset Polling) ─────────────────
  let initTries = 0;
  const pollInit = setInterval(() => {
    initTries++;
    createHUD();
    initExtensionBridge();
    if (document.getElementById("quant-hud-container") || initTries > 20) {
      clearInterval(pollInit);
    }
  }, 600);
})();
