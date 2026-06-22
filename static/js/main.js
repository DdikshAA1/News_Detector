/**
 * main.js
 * -------
 * Handles all frontend interactivity:
 *   1. Character counter on the textarea
 *   2. Quick-example buttons that pre-fill the textarea
 *   3. Sending the user's text to the Flask /predict endpoint via fetch()
 *   4. Rendering the prediction result (verdict, confidence bar, probabilities)
 *   5. Rendering global verification: reason, sources, analysis breakdown
 *   6. Error handling
 *
 * HOW FRONTEND <-> BACKEND WORKS:
 *   - User types text and clicks "Analyse"
 *   - fetch() sends a POST request to /predict with JSON body { "text": "..." }
 *   - Flask processes it (ML + web search) and returns JSON
 *   - We read the JSON and update the DOM (HTML elements) to show the result
 *
 * This file uses vanilla JavaScript — no external JS libraries needed.
 */

/* =========================================================================
   DOM Element References
   — We grab elements once and reuse them, rather than querying the DOM
     on every function call (which is slower).
   ========================================================================= */
const newsForm     = document.getElementById("newsForm");
const newsText     = document.getElementById("newsText");
const charCounter  = document.getElementById("charCounter");
const analyseBtn   = document.getElementById("analyseBtn");
const resultPanel  = document.getElementById("resultPanel");
const errorPanel   = document.getElementById("errorPanel");
const errorMsg     = document.getElementById("errorMsg");

// Verdict elements
const verdictBanner    = document.getElementById("verdictBanner");
const verdictIcon      = document.getElementById("verdictIcon");
const verdictIconClass = document.getElementById("verdictIconClass");
const verdictLabel     = document.getElementById("verdictLabel");
const verdictSub       = document.getElementById("verdictSub");

// Reason
const reasonText = document.getElementById("reasonText");

// Confidence bar
const confPct      = document.getElementById("confPct");
const progressFill = document.getElementById("progressFill");

// Probability cards
const fakeProb     = document.getElementById("fakeProb");
const realProb     = document.getElementById("realProb");

// Analysis breakdown
const mlScoreVal   = document.getElementById("mlScoreVal");
const webScoreVal  = document.getElementById("webScoreVal");
const langScoreVal = document.getElementById("langScoreVal");

// Sources
const sourceList   = document.getElementById("sourceList");
const sourcesCount = document.getElementById("sourcesCount");

// Metadata
const wordCountEl    = document.getElementById("wordCount");
const trustedCountEl = document.getElementById("trustedCount");
const responseTimeEl = document.getElementById("responseTime");


/* =========================================================================
   1. Character Counter
   ========================================================================= */
newsText.addEventListener("input", () => {
  const len = newsText.value.length;
  charCounter.textContent = `${len} / 5000`;

  // Warn when getting close to the limit
  charCounter.style.color = len > 4500
    ? "#fc8181"   // red — near limit
    : "#4a5568";  // default dim grey
});


/* =========================================================================
   2. Quick Example Buttons
   — Each button has a data-text attribute containing example text.
   ========================================================================= */
document.querySelectorAll(".btn-example").forEach(btn => {
  btn.addEventListener("click", () => {
    newsText.value = btn.dataset.text;

    // Trigger the 'input' event so the character counter updates
    newsText.dispatchEvent(new Event("input"));

    // Scroll the textarea into view smoothly
    newsText.scrollIntoView({ behavior: "smooth", block: "center" });
    newsText.focus();
  });
});


/* =========================================================================
   3. Form Submission → API Call
   ========================================================================= */
newsForm.addEventListener("submit", async (e) => {
  // Prevent the default browser form submission (which would reload the page)
  e.preventDefault();

  const text = newsText.value.trim();

  if (text.length < 10) {
    showError("Please enter at least a sentence of text.");
    return;
  }

  // --- Start loading state ---
  setLoading(true);
  hideResults();

  const startTime = performance.now(); // used to calculate response time

  try {
    /**
     * fetch() sends an HTTP request.
     *
     * POST /predict  — the route we defined in app.py
     * Content-Type   — tells Flask we're sending JSON
     * body           — JSON-encoded request body
     */
    const response = await fetch("/predict", {
      method:  "POST",
      headers: { "Content-Type": "application/json" },
      body:    JSON.stringify({ text }),
    });

    const data = await response.json(); // parse the JSON response from Flask

    const elapsed = Math.round(performance.now() - startTime);

    if (!response.ok || data.error) {
      showError(data.error || "Prediction failed. Please try again.");
      return;
    }

    // --- Render result ---
    showResult(data, elapsed);
    loadHistory();

  } catch (err) {
    // Network error (server down, no internet, etc.)
    showError("Could not reach the server. Make sure Flask is running.");
    console.error("Fetch error:", err);

  } finally {
    setLoading(false);
  }
});


/* =========================================================================
   4. Render Result
   ========================================================================= */
/**
 * showResult() — updates the DOM with the prediction from Flask.
 *
 * @param {Object} data    — the JSON object returned by /predict
 * @param {number} elapsed — request round-trip time in milliseconds
 */
function showResult(data, elapsed) {
  const isFake = data.prediction === "FAKE";

  // --- Verdict banner ---
  verdictBanner.className = `verdict-banner ${isFake ? "is-fake" : "is-real"}`;
  verdictLabel.textContent = data.label;    // "FAKE NEWS" or "REAL NEWS"

  // Icon inside the verdict circle
  verdictIconClass.className = `bi ${isFake ? "bi-x-octagon-fill" : "bi-check-circle-fill"}`;

  // Subtext below the verdict
  verdictSub.textContent = isFake
    ? "Cross-referencing with global sources suggests this content may be misleading."
    : "Cross-referencing with global sources suggests this content appears credible.";

  // --- Reason ---
  reasonText.textContent = data.reason || "Analysis complete.";

  // --- Confidence bar ---
  confPct.textContent = `${data.confidence}%`;
  // Slight delay so the CSS transition is visible (feels more dynamic)
  setTimeout(() => {
    progressFill.style.width = `${data.confidence}%`;
  }, 50);

  // Apply the colour class to the bar wrapper (see CSS: .result-panel.is-fake)
  resultPanel.className = `result-panel mt-4 ${isFake ? "is-fake" : "is-real"}`;

  // --- Analysis Breakdown ---
  mlScoreVal.textContent  = `${data.ml_score}%`;
  webScoreVal.textContent = `${data.web_score}%`;
  langScoreVal.textContent = `${data.lang_score}%`;

  // --- Probability cards ---
  fakeProb.textContent = `${data.fake_prob}%`;
  realProb.textContent = `${data.real_prob}%`;

  // --- Sources ---
  renderSources(data.sources || []);

  // --- Metadata ---
  wordCountEl.textContent    = data.word_count;
  trustedCountEl.textContent = data.trusted_found || 0;
  responseTimeEl.textContent = elapsed;

  // --- Show the panel ---
  resultPanel.classList.remove("d-none");

  // Smooth scroll to the result
  resultPanel.scrollIntoView({ behavior: "smooth", block: "start" });
}


/* =========================================================================
   5. Render Sources List
   ========================================================================= */
/**
 * renderSources() — builds the source list HTML from the sources array.
 *
 * @param {Array} sources — array of source objects from the API
 */
function renderSources(sources) {
  sourceList.innerHTML = "";

  const count = sources.length;
  const trusted = sources.filter(s => s.is_trusted).length;
  sourcesCount.textContent = `${trusted} trusted / ${count} found`;

  if (count === 0) {
    sourceList.innerHTML = `
      <li class="no-sources">
        <i class="bi bi-cloud-slash"></i>
        No matching news articles found online.
        This may indicate the claim is unverified.
      </li>
    `;
    return;
  }

  sources.forEach(src => {
    const li = document.createElement("li");
    li.className = "source-item";

    const tierLabels = {
      1: "Wire Service",
      2: "Broadcaster",
      3: "Newspaper",
      4: "Fact-Checker",
    };

    const tierLabel = src.tier > 0
      ? `<span class="tier-badge tier-${src.tier}">${tierLabels[src.tier] || ""}</span>`
      : "";

    const trustIcon = src.is_trusted
      ? '<div class="trust-badge trusted"><i class="bi bi-check-lg"></i></div>'
      : '<div class="trust-badge untrusted"><i class="bi bi-dash"></i></div>';

    li.innerHTML = `
      ${trustIcon}
      <div class="source-info">
        <div class="source-name">
          ${escapeHtml(src.source)}
          ${tierLabel}
        </div>
        <div class="source-title">${escapeHtml(src.title)}</div>
      </div>
      <a href="${escapeHtml(src.url)}" target="_blank" rel="noopener noreferrer"
         class="source-link" title="Open article">
        <i class="bi bi-box-arrow-up-right"></i>
      </a>
    `;

    sourceList.appendChild(li);
  });
}


/* =========================================================================
   6. Helper Utilities
   ========================================================================= */

/**
 * escapeHtml() — prevent XSS by escaping HTML entities in user/API content.
 */
function escapeHtml(str) {
  if (!str) return "";
  const div = document.createElement("div");
  div.textContent = str;
  return div.innerHTML;
}

/**
 * setLoading(bool) — toggle button between normal and loading state.
 */
function setLoading(isLoading) {
  analyseBtn.disabled = isLoading;

  const btnText    = analyseBtn.querySelector(".btn-text");
  const btnLoading = analyseBtn.querySelector(".btn-loading");

  if (isLoading) {
    btnText.classList.add("d-none");
    btnLoading.classList.remove("d-none");
  } else {
    btnText.classList.remove("d-none");
    btnLoading.classList.add("d-none");
  }
}

/**
 * hideResults() — hide both the result and error panels.
 */
function hideResults() {
  resultPanel.classList.add("d-none");
  errorPanel.classList.add("d-none");
  // Reset progress bar for next animation
  progressFill.style.width = "0%";
}

/**
 * showError(message) — display an error message below the form.
 */
function showError(message) {
  errorMsg.textContent = message;
  errorPanel.classList.remove("d-none");
  errorPanel.scrollIntoView({ behavior: "smooth", block: "nearest" });
}

/**
 * loadHistory() — fetch history from backend and render it
 */
async function loadHistory() {
  const historyList = document.getElementById("historyList");
  if (!historyList) return;

  try {
    const response = await fetch("/history");
    const data = await response.json();
    
    if (data.length === 0) {
      historyList.innerHTML = `
        <li class="no-sources">
          <i class="bi bi-clock"></i>
          No recent predictions found.
        </li>`;
      return;
    }

    historyList.innerHTML = "";
    data.forEach(item => {
      const isFake = item.prediction === "FAKE";
      const iconClass = isFake ? "bi-x-octagon-fill" : "bi-check-circle-fill";
      const trustClass = isFake ? "untrusted" : "trusted";
      const colorStyle = isFake ? "color: var(--fake-color);" : "color: var(--real-color);";

      const li = document.createElement("li");
      li.className = "source-item";
      li.innerHTML = `
        <div class="trust-badge ${trustClass}">
          <i class="bi ${iconClass}"></i>
        </div>
        <div class="source-info">
          <div class="source-name" style="${colorStyle}">
            ${item.prediction} <span class="tier-badge" style="color: var(--text-muted); background: rgba(255,255,255,0.1)">${item.confidence}%</span>
          </div>
          <div class="source-title" style="white-space: normal;">
            ${escapeHtml(item.text)}
          </div>
          <div style="font-size: 0.65rem; color: var(--text-dim); margin-top: 4px;">
            ${new Date(item.timestamp).toLocaleString()}
          </div>
        </div>
      `;
      historyList.appendChild(li);
    });

  } catch (err) {
    console.error("Failed to load history:", err);
  }
}

// Initial load of history when page loads
loadHistory();
