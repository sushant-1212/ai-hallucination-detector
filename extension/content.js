/**
 * content.js
 * -----------
 * Injected into ChatGPT, Claude, and Gemini tabs.
 * Extracts the latest assistant response bubble or highlighted selection.
 */

function detectPlatform() {
  const host = window.location.hostname;
  if (host.includes("chatgpt.com") || host.includes("openai.com")) {
    return "chatgpt";
  }
  if (host.includes("claude.ai")) {
    return "claude";
  }
  if (host.includes("gemini.google.com")) {
    return "gemini";
  }
  return "web";
}

function extractLatestResponse() {
  const platform = detectPlatform();
  let text = "";

  // 1. Check if user currently has text selected on the page
  const selection = window.getSelection().toString().trim();
  if (selection.length > 10) {
    return {
      text: selection,
      platform: platform,
      type: "selection"
    };
  }

  // 2. Platform-specific DOM scraping for latest assistant message
  try {
    if (platform === "chatgpt") {
      // ChatGPT selectors
      const assistantMsgs = document.querySelectorAll(
        '[data-message-author-role="assistant"], article[data-testid^="conversation-turn-"] .markdown, .agent-turn .markdown'
      );
      if (assistantMsgs.length > 0) {
        const latest = assistantMsgs[assistantMsgs.length - 1];
        text = latest.innerText.trim();
      }
    } else if (platform === "claude") {
      // Claude selectors
      const claudeMsgs = document.querySelectorAll(
        '.font-claude-message, [data-is-streaming="false"] .prose, div[class*="font-claude"], div[class*="ChatMessage"] .prose'
      );
      if (claudeMsgs.length > 0) {
        const latest = claudeMsgs[claudeMsgs.length - 1];
        text = latest.innerText.trim();
      }
    } else if (platform === "gemini") {
      // Gemini selectors
      const geminiMsgs = document.querySelectorAll(
        'message-content, .model-response-text, .response-container-content, div[class*="model-query"] ~ div, .markdown-content'
      );
      if (geminiMsgs.length > 0) {
        const latest = geminiMsgs[geminiMsgs.length - 1];
        text = latest.innerText.trim();
      }
    }
  } catch (err) {
    console.error("[VeritasAI] Error extracting response:", err);
  }

  // Fallback: check any general markdown or article container
  if (!text) {
    const generic = document.querySelectorAll('article, .prose, main div[class*="message"]');
    if (generic.length > 0) {
      text = generic[generic.length - 1].innerText.trim();
    }
  }

  return {
    text: text,
    platform: platform,
    type: "dom_extracted"
  };
}

// Listen for messages from popup
chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
  if (request.action === "GET_LATEST_RESPONSE") {
    const data = extractLatestResponse();
    sendResponse(data);
  } else if (request.action === "GET_SELECTED_TEXT") {
    const selection = window.getSelection().toString().trim();
    sendResponse({
      text: selection,
      platform: detectPlatform(),
      type: "selection"
    });
  }
  return true;
});
