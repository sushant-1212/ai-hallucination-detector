/**
 * popup.js
 * --------
 * Controller for the VeritasAI Hallucination Detector Chrome Extension.
 * Handles auto-detection of ChatGPT, Claude, and Gemini, DOM text extraction,
 * and calls the FastAPI backend for real-time claim verification.
 */

let currentPlatform = "manual";

const PLATFORM_CONFIG = {
  chatgpt: {
    name: "ChatGPT",
    icon: "🟢",
    badgeClass: "badge-chatgpt",
    domains: ["chatgpt.com", "chat.openai.com"]
  },
  claude: {
    name: "Claude",
    icon: "🟠",
    badgeClass: "badge-claude",
    domains: ["claude.ai"]
  },
  gemini: {
    name: "Gemini",
    icon: "🔵",
    badgeClass: "badge-gemini",
    domains: ["gemini.google.com"]
  },
  manual: {
    name: "Web / Custom",
    icon: "🌐",
    badgeClass: "badge-neutral",
    domains: []
  }
};

document.addEventListener("DOMContentLoaded", async () => {
  initUI();
  await detectActiveTabPlatform();
});

function initUI() {
  const inputText = document.getElementById("input-text");
  const charCount = document.getElementById("char-count");
  const btnVerify = document.getElementById("btn-verify");
  const btnGrabLatest = document.getElementById("btn-grab-latest");
  const btnGrabSelection = document.getElementById("btn-grab-selection");
  const btnClear = document.getElementById("btn-clear");

  // Character counter
  inputText.addEventListener("input", () => {
    charCount.textContent = `${inputText.value.length} chars`;
  });

  // Clear button
  btnClear.addEventListener("click", () => {
    inputText.value = "";
    charCount.textContent = "0 chars";
    hideResults();
    hideError();
  });

  // Grab Latest Response button
  btnGrabLatest.addEventListener("click", () => {
    grabFromActiveTab("GET_LATEST_RESPONSE");
  });

  // Grab Selection button
  btnGrabSelection.addEventListener("click", () => {
    grabFromActiveTab("GET_SELECTED_TEXT");
  });

  // Verify button
  btnVerify.addEventListener("click", () => {
    performVerification();
  });

  // Platform tab buttons
  document.querySelectorAll(".tab-btn").forEach(btn => {
    btn.addEventListener("click", () => {
      const platform = btn.getAttribute("data-platform");
      setPlatform(platform);
    });
  });
}

/**
 * Detect the active browser tab and auto-switch the UI platform.
 */
async function detectActiveTabPlatform() {
  try {
    const tabs = await chrome.tabs.query({ active: true, currentWindow: true });
    if (!tabs || tabs.length === 0) return;

    const currentTab = tabs[0];
    const url = currentTab.url || "";

    let detected = "manual";
    for (const [key, config] of Object.entries(PLATFORM_CONFIG)) {
      if (config.domains.some(domain => url.includes(domain))) {
        detected = key;
        break;
      }
    }

    setPlatform(detected);

    // If on a supported LLM page, attempt to auto-grab the latest answer
    if (detected !== "manual") {
      grabFromActiveTab("GET_LATEST_RESPONSE", /* silent */ true);
    }
  } catch (err) {
    console.error("Auto-detect failed:", err);
    setPlatform("manual");
  }
}

/**
 * Update UI for the selected platform
 */
function setPlatform(platform) {
  currentPlatform = platform;
  const config = PLATFORM_CONFIG[platform] || PLATFORM_CONFIG.manual;

  // Update top badge
  const badgeEl = document.getElementById("active-llm-badge");
  const nameEl = document.getElementById("active-llm-name");

  badgeEl.className = `active-badge ${config.badgeClass}`;
  nameEl.textContent = `${config.name} ${config.icon}`;

  // Update tabs
  document.querySelectorAll(".tab-btn").forEach(btn => {
    if (btn.getAttribute("data-platform") === platform) {
      btn.classList.add("active");
    } else {
      btn.classList.remove("active");
    }
  });
}

/**
 * Send extraction message to the content script in the active tab
 */
async function grabFromActiveTab(actionType, silent = false) {
  hideError();
  try {
    const tabs = await chrome.tabs.query({ active: true, currentWindow: true });
    if (!tabs || tabs.length === 0) return;

    chrome.tabs.sendMessage(tabs[0].id, { action: actionType }, (response) => {
      if (chrome.runtime.lastError) {
        if (!silent) {
          showError("Could not read text from this tab. Please refresh the page or paste text directly.");
        }
        return;
      }

      if (response && response.text) {
        const input = document.getElementById("input-text");
        input.value = response.text;
        document.getElementById("char-count").textContent = `${response.text.length} chars`;
        if (response.platform && response.platform !== "web") {
          setPlatform(response.platform);
        }
      } else if (!silent) {
        showError("No message found to grab. You can copy and paste the answer into the box.");
      }
    });
  } catch (err) {
    if (!silent) {
      showError("Unable to communicate with active tab.");
    }
  }
}

/**
 * Call the FastAPI backend to verify factual claims
 */
async function performVerification() {
  const text = document.getElementById("input-text").value.trim();
  if (!text) {
    showError("Please enter or grab some text to verify.");
    return;
  }

  const endpointBase = document.getElementById("select-endpoint").value;
  const useWiki = document.getElementById("check-wiki").checked;

  hideError();
  hideResults();
  setLoading(true);

  try {
    const response = await fetch(`${endpointBase}/api/check`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json"
      },
      body: JSON.stringify({
        text: text,
        llm_source: currentPlatform,
        use_wikipedia: useWiki
      })
    });

    if (!response.ok) {
      const errorData = await response.json().catch(() => ({}));
      throw new Error(errorData.detail || `Server responded with status ${response.status}`);
    }

    const data = await response.json();
    renderResults(data);
  } catch (err) {
    console.error("Verification error:", err);
    showError(
      `Connection failed: ${err.message}. If running locally, make sure 'python api.py' is running, or select 'Live Render Cloud' above.`
    );
  } finally {
    setLoading(false);
  }
}

/**
 * Render evaluation results into the DOM
 */
function renderResults(data) {
  const container = document.getElementById("results-container");
  container.classList.remove("hidden");

  // Scores
  document.getElementById("res-reliability-num").textContent = `${Math.round(data.reliability_score)}%`;
  document.getElementById("res-hallucination-num").textContent = `${Math.round(data.hallucination_rate)}%`;

  // Risk Badge
  const riskBadge = document.getElementById("res-risk-badge");
  riskBadge.style.backgroundColor = `${data.risk_color}25`;
  riskBadge.style.color = data.risk_color;
  riskBadge.style.border = `1px solid ${data.risk_color}50`;
  document.getElementById("res-risk-emoji").textContent = data.risk_emoji || "🛡️";
  document.getElementById("res-risk-label").textContent = data.risk_label || "Evaluated";

  // Stat Pills
  document.getElementById("stat-supported-num").textContent = data.supported_count;
  document.getElementById("stat-contradicted-num").textContent = data.contradicted_count;
  document.getElementById("stat-insufficient-num").textContent = data.insufficient_count;
  document.getElementById("claims-total-count").textContent = `${data.total_claims} claims`;

  // Claims List
  const listEl = document.getElementById("claims-list");
  listEl.innerHTML = "";

  if (!data.claims || data.claims.length === 0) {
    listEl.innerHTML = `<div style="text-align:center; color:#94a3b8; padding:10px;">No atomic factual claims found.</div>`;
    return;
  }

  data.claims.forEach(c => {
    const card = document.createElement("div");
    const vClass = c.verdict.toLowerCase().replace("_", "-");
    card.className = `claim-card verdict-${vClass}`;

    let badgeClass = "insufficient";
    if (c.verdict === "SUPPORTED") badgeClass = "supported";
    if (c.verdict === "CONTRADICTED") badgeClass = "contradicted";

    let evidenceHtml = "";
    if (c.evidence && c.evidence.length > 0) {
      const topEv = c.evidence[0];
      evidenceHtml = `
        <div class="evidence-box">
          <span class="evidence-source">📌 ${escapeHtml(topEv.source)}</span>:
          "${escapeHtml(topEv.text.slice(0, 160))}..."
        </div>
      `;
    }

    card.innerHTML = `
      <div class="claim-card-header">
        <span class="claim-badge ${badgeClass}">${c.verdict.replace("_", " ")}</span>
        <span class="claim-confidence">${c.confidence}% conf</span>
      </div>
      <div class="claim-text">"${escapeHtml(c.claim)}"</div>
      ${c.explanation ? `<div class="claim-explanation">${escapeHtml(c.explanation)}</div>` : ""}
      ${evidenceHtml}
    `;

    listEl.appendChild(card);
  });
}

function setLoading(isLoading) {
  const btn = document.getElementById("btn-verify");
  const text = document.getElementById("btn-verify-text");
  const spinner = document.getElementById("btn-spinner");

  btn.disabled = isLoading;
  if (isLoading) {
    text.textContent = "Verifying with Groq Engine...";
    spinner.classList.remove("hidden");
  } else {
    text.textContent = "🛡️ Check for Hallucinations";
    spinner.classList.add("hidden");
  }
}

function showError(msg) {
  const banner = document.getElementById("error-banner");
  banner.textContent = msg;
  banner.classList.remove("hidden");
}

function hideError() {
  document.getElementById("error-banner").classList.add("hidden");
}

function hideResults() {
  document.getElementById("results-container").classList.add("hidden");
}

function escapeHtml(str) {
  if (!str) return "";
  return str.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;");
}
