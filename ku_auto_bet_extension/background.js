/**
 * KUBET Quant Auto-Bet Background Service Worker
 * Privileged connection to local Python WebSocket without any browser popups!
 * Author: cook45
 */

let ws = null;
let isConnected = false;

function connectWS() {
  try {
    ws = new WebSocket("ws://127.0.0.1:8765");

    ws.onopen = () => {
      console.log("[Background SW] 🟢 Connected to Python Bridge on ws://127.0.0.1:8765");
      isConnected = true;
      broadcastToTabs({ type: "BRIDGE_STATUS", connected: true });
      ws.send(JSON.stringify({ type: "CLIENT_READY", client: "BACKGROUND_SW" }));
    };

    ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        console.log("[Background SW] Message from Python:", data);

        // Direct Visual Vision: Capture screenshot of the active sportsbook tab
        if (data.type === "CAPTURE_SCREEN") {
          chrome.tabs.captureVisibleTab(null, { format: "png" }, (dataUrl) => {
            if (chrome.runtime.lastError || !dataUrl) {
              console.error("[Background SW] Capture failed:", chrome.runtime.lastError);
              return;
            }
            if (ws && ws.readyState === WebSocket.OPEN) {
              ws.send(JSON.stringify({
                type: "SCREENSHOT_DATA",
                data: dataUrl
              }));
            }
          });
          return;
        }

        broadcastToTabs(data);
      } catch (err) {
        console.error("[Background SW] Error parsing message:", err);
      }
    };

    ws.onclose = () => {
      console.log("[Background SW] 🔴 Disconnected from Python Bridge. Retrying in 3s...");
      isConnected = false;
      broadcastToTabs({ type: "BRIDGE_STATUS", connected: false });
      setTimeout(connectWS, 3000);
    };

    ws.onerror = (err) => {
      console.error("[Background SW] WS Error:", err);
      ws.close();
    };
  } catch (err) {
    setTimeout(connectWS, 3000);
  }
}

function broadcastToTabs(message) {
  chrome.tabs.query({}, (tabs) => {
    // If it's a general status or config update, send to all tabs
    if (message.type !== "BET_SIGNAL") {
      for (const tab of tabs) {
        if (tab.id) {
          chrome.tabs.sendMessage(tab.id, message).catch(() => {});
        }
      }
      return;
    }

    // CRITICAL SHIELD: For BET_SIGNAL, ONLY send to a SINGLE active Sportsbook tab!
    // Prevents multi-tab duplicate betting catastrophe
    const sportsTabs = tabs.filter((t) => {
      const url = (t.url || "").toLowerCase();
      return (
        url.includes("jjsskktt") ||
        url.includes("cuu") ||
        url.includes("hdfr") ||
        url.includes("heratven") ||
        url.includes("games.aspx") ||
        url.includes("bbview") ||
        url.includes("sport")
      );
    });

    if (sportsTabs.length === 0) {
      console.warn("[Background SW] ⚠️ No active KUBET sportsbook tabs found to execute signal.");
      return;
    }

    // Prioritize active/focused tab, otherwise fallback to the first sports tab
    const targetTab = sportsTabs.find((t) => t.active) || sportsTabs[0];
    console.log(`[Background SW] 🎯 Dispatching BET_SIGNAL to Tab ${targetTab.id} (${targetTab.url})`);
    
    // Broadcast to all frames in the tab so the frame with the odds table receives it
    chrome.tabs.sendMessage(targetTab.id, message).catch(() => {});
  });
}

// Listen for messages from content.js (e.g. bet confirmations) and forward to Python WS
chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
  if (request.type === "GET_STATUS") {
    sendResponse({ connected: isConnected });
    return true;
  }

  if (ws && ws.readyState === WebSocket.OPEN) {
    ws.send(JSON.stringify(request));
  }
  sendResponse({ status: "SENT_TO_WS" });
  return true;
});

// Port Keep-Alive
chrome.runtime.onConnect.addListener((port) => {
  if (port.name === "keepAlive") {
    port.onDisconnect.addListener(() => {});
  }
});

// Alarms Keep-Alive (every 20s prevents MV3 sleep)
chrome.alarms.create("keepAlive", { periodInMinutes: 0.3 });
chrome.alarms.onAlarm.addListener((alarm) => {
  if (alarm.name === "keepAlive") {
    if (!ws || ws.readyState !== WebSocket.OPEN) {
      connectWS();
    }
  }
});

// Start connection immediately
connectWS();
