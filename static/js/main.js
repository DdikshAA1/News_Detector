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


/* =========================================================================
   7. News Feed Explorer (Global & Dataset)
   ========================================================================= */
const fetchGlobalBtn  = document.getElementById("fetchGlobalBtn");
const fetchDatasetBtn = document.getElementById("fetchDatasetBtn");
const exploreLoading  = document.getElementById("exploreLoading");
const exploreList     = document.getElementById("exploreList");
const manualTabBtn    = document.getElementById("manual-tab");

// Fetch and render global news
if (fetchGlobalBtn) {
  fetchGlobalBtn.addEventListener("click", async () => {
    try {
      exploreLoading.classList.remove("d-none");
      exploreList.innerHTML = "";
      
      const response = await fetch("/api/fetch_global");
      const data = await response.json();
      
      if (!data.success || !data.articles || data.articles.length === 0) {
        exploreList.innerHTML = `
          <li class="no-sources">
            <i class="bi bi-exclamation-triangle"></i>
            Failed to fetch global news. Make sure you have internet access.
          </li>`;
        return;
      }
      
      data.articles.forEach(art => {
        const li = document.createElement("li");
        li.className = "source-item d-flex align-items-center justify-content-between py-3";
        li.innerHTML = `
          <div class="source-info" style="flex: 1; min-width: 0; padding-right: 15px;">
            <div class="source-name" style="font-size: 0.8rem; color: var(--accent);">
              <i class="bi bi-globe me-1"></i> ${escapeHtml(art.source)}
            </div>
            <div class="source-title text-light" style="font-size: 0.88rem; white-space: normal; overflow: visible; text-overflow: clip;">
              ${escapeHtml(art.title)}
            </div>
          </div>
          <button class="verify-btn" data-text="${escapeHtml(art.title)}">
            <i class="bi bi-shield-fill-check"></i> Verify
          </button>
        `;
        exploreList.appendChild(li);
      });
      
      // Bind verify actions
      bindVerifyButtons();
      
    } catch (err) {
      console.error(err);
      exploreList.innerHTML = `
        <li class="no-sources">
          <i class="bi bi-x-circle"></i>
          Error loading global news feed.
        </li>`;
    } finally {
      exploreLoading.classList.add("d-none");
    }
  });
}

// Fetch and render dataset news samples
if (fetchDatasetBtn) {
  fetchDatasetBtn.addEventListener("click", async () => {
    try {
      exploreLoading.classList.remove("d-none");
      exploreList.innerHTML = "";
      
      const response = await fetch("/api/fetch_dataset");
      const data = await response.json();
      
      if (!data.success || !data.articles || data.articles.length === 0) {
        exploreList.innerHTML = `
          <li class="no-sources">
            <i class="bi bi-exclamation-triangle"></i>
            Failed to fetch dataset samples.
          </li>`;
        return;
      }
      
      data.articles.forEach(art => {
        const isFake = art.label === "FAKE";
        const labelStyle = isFake ? "background: rgba(252, 129, 129, 0.15); color: var(--fake-color);" : "background: rgba(104, 211, 145, 0.15); color: var(--real-color);";
        
        const li = document.createElement("li");
        li.className = "source-item d-flex align-items-center justify-content-between py-3";
        li.innerHTML = `
          <div class="source-info" style="flex: 1; min-width: 0; padding-right: 15px;">
            <div class="source-name">
              <span class="tier-badge" style="${labelStyle} font-size: 0.58rem; letter-spacing: 0.05em; padding: 0.1em 0.45em; border-radius: 4px; text-transform: uppercase;">
                Dataset: ${art.label}
              </span>
            </div>
            <div class="source-title text-light" style="font-size: 0.88rem; white-space: normal; overflow: visible; text-overflow: clip;">
              ${escapeHtml(art.text)}
            </div>
          </div>
          <button class="verify-btn" data-text="${escapeHtml(art.text)}">
            <i class="bi bi-shield-fill-check"></i> Verify
          </button>
        `;
        exploreList.appendChild(li);
      });
      
      // Bind verify actions
      bindVerifyButtons();
      
    } catch (err) {
      console.error(err);
      exploreList.innerHTML = `
        <li class="no-sources">
          <i class="bi bi-x-circle"></i>
          Error loading dataset samples.
        </li>`;
    } finally {
      exploreLoading.classList.add("d-none");
    }
  });
}

// Bind click event to dynamic Verify buttons
function bindVerifyButtons() {
  exploreList.querySelectorAll(".verify-btn").forEach(btn => {
    btn.addEventListener("click", () => {
      const textToVerify = btn.dataset.text;
      if (!textToVerify) return;
      
      // 1. Fill manual textarea
      newsText.value = textToVerify;
      
      // 2. Trigger input event so char counter updates
      newsText.dispatchEvent(new Event("input"));
      
      // 3. Switch tab back to manual
      if (manualTabBtn) {
        // Bootstrap 5 tab activation
        const tab = new bootstrap.Tab(manualTabBtn);
        tab.show();
      }
      
      // 4. Scroll textarea into view and focus
      newsText.scrollIntoView({ behavior: "smooth", block: "center" });
      newsText.focus();
      
      // 5. Submit form automatically
      newsForm.dispatchEvent(new Event("submit"));
    });
  });
}


/* =========================================================================
   8. Voice Input — Speech Recognition (Microphone)
   — Uses the Web Speech API to transcribe spoken news into the textarea.
   — The transcribed text is then analysed like any typed text.
   ========================================================================= */

const micBtn        = document.getElementById("micBtn");
const micIcon       = document.getElementById("micIcon");
const voiceStatus   = document.getElementById("voiceStatus");
const voiceStatusText = document.getElementById("voiceStatusText");

// Feature detection for Web Speech API
const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;

if (SpeechRecognition && micBtn) {
  const recognition = new SpeechRecognition();
  
  // Configuration
  recognition.continuous   = true;    // Keep listening until manually stopped
  recognition.interimResults = true;  // Show partial results while speaking
  recognition.lang         = "hi-IN"; // Default: Hindi (supports Hindi + English mixed)
  recognition.maxAlternatives = 1;

  let isListening   = false;
  let finalTranscript = "";

  /**
   * Start / Stop listening on mic button click
   */
  micBtn.addEventListener("click", () => {
    if (isListening) {
      stopListening();
    } else {
      startListening();
    }
  });

  /**
   * startListening() — activate microphone and start recognition
   */
  function startListening() {
    // Reset
    finalTranscript = newsText.value; // Preserve existing text in textarea
    
    try {
      recognition.start();
      isListening = true;

      // UI updates
      micBtn.classList.add("mic-active");
      micIcon.className = "bi bi-mic-fill";
      voiceStatus.classList.remove("d-none");
      voiceStatusText.textContent = "🎙️ Listening... Speak your news";

      // Hide any previous errors/results
      errorPanel.classList.add("d-none");
    } catch (err) {
      console.error("Speech recognition start error:", err);
      showError("Microphone access failed. Please check browser permissions.");
    }
  }

  /**
   * stopListening() — deactivate microphone and stop recognition
   */
  function stopListening() {
    recognition.stop();
    isListening = false;

    // UI updates
    micBtn.classList.remove("mic-active");
    micIcon.className = "bi bi-mic-fill";
    voiceStatus.classList.add("d-none");
  }

  /**
   * onresult — fired when the speech engine returns a result (partial or final)
   */
  recognition.onresult = (event) => {
    let interimTranscript = "";

    for (let i = event.resultIndex; i < event.results.length; i++) {
      const transcript = event.results[i][0].transcript;
      if (event.results[i].isFinal) {
        finalTranscript += (finalTranscript ? " " : "") + transcript;
      } else {
        interimTranscript += transcript;
      }
    }

    // Update textarea with final + interim text
    newsText.value = finalTranscript + (interimTranscript ? " " + interimTranscript : "");
    newsText.dispatchEvent(new Event("input")); // update char counter

    // Update status text
    if (interimTranscript) {
      voiceStatusText.textContent = "🎙️ Listening... (recognizing speech)";
    } else {
      voiceStatusText.textContent = "🎙️ Listening... Speak your news";
    }
  };

  /**
   * onerror — handle recognition errors gracefully
   */
  recognition.onerror = (event) => {
    console.error("Speech recognition error:", event.error);
    stopListening();

    const errorMessages = {
      "not-allowed":     "Microphone access denied. Please allow microphone permission in your browser.",
      "no-speech":       "No speech detected. Please try speaking again.",
      "audio-capture":   "No microphone found. Please connect a microphone.",
      "network":         "Network error. Speech recognition requires internet.",
      "aborted":         "Speech recognition was stopped.",
    };

    const msg = errorMessages[event.error] || `Speech recognition error: ${event.error}`;
    
    // Don't show error for "aborted" — user intentionally stopped
    if (event.error !== "aborted") {
      showError(msg);
    }
  };

  /**
   * onend — fired when recognition session ends (timeout or manual stop)
   */
  recognition.onend = () => {
    // If still supposed to be listening (timed out), restart
    if (isListening) {
      try {
        recognition.start();
      } catch (e) {
        stopListening();
      }
    }
  };

  // Language toggle: let user switch between Hindi and English
  // Double-click mic to toggle language
  micBtn.addEventListener("dblclick", (e) => {
    e.preventDefault();
    if (recognition.lang === "hi-IN") {
      recognition.lang = "en-US";
      micBtn.title = "Language: English — Click to speak, double-click to switch to Hindi";
      if (isListening) {
        voiceStatusText.textContent = "🎙️ Listening (English)... Speak your news";
      }
    } else {
      recognition.lang = "hi-IN";
      micBtn.title = "Language: Hindi — Click to speak, double-click to switch to English";
      if (isListening) {
        voiceStatusText.textContent = "🎙️ Listening (Hindi)... Speak your news";
      }
    }

    // If currently listening, restart with new language
    if (isListening) {
      recognition.stop();
      setTimeout(() => {
        try { recognition.start(); } catch(e) {}
      }, 200);
    }
  });

} else if (micBtn) {
  // Browser doesn't support Speech Recognition
  micBtn.disabled = true;
  micBtn.title = "Speech recognition is not supported in this browser. Use Chrome or Edge.";
  micBtn.classList.add("mic-unsupported");
}

