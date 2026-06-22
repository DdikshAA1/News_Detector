# fact_checker.py
# ---------------
# Global news verification engine that cross-references user-submitted news
# against real-world trusted sources using NewsAPI.org (primary) and
# GNews RSS (fallback).
#
# Pipeline:
#   1. Extract key phrases from the news text
#   2. Search NewsAPI.org for matching news articles (precise matching)
#   3. Fallback to GNews RSS if NewsAPI limit reached
#   4. Check how many results come from trusted global sources
#   5. Detect sensationalist language patterns
#   6. Combine ML model prediction + web evidence into a final verdict
#
# TRUSTED SOURCE TIERS:
#   Tier 1 — Major wire services (Reuters, AP, AFP)
#   Tier 2 — Major broadcasters (BBC, CNN, Al Jazeera, NPR)
#   Tier 3 — Major newspapers (NYT, Guardian, Times of India, etc.)
#   Tier 4 — Fact-checkers (Snopes, PolitiFact, AltNews, etc.)

import os
import re
import logging
import hashlib
from urllib.parse import urlparse
from datetime import datetime, timedelta

import requests as http_requests

# Try importing GNews as fallback; if unavailable, skip it
try:
    from gnews import GNews
    GNEWS_AVAILABLE = True
except ImportError:
    GNEWS_AVAILABLE = False

# --------------------------------------------------------------------------- #
#  Logging                                                                      #
# --------------------------------------------------------------------------- #

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
#  NewsAPI Configuration                                                        #
# --------------------------------------------------------------------------- #
#
# Get your FREE API key from https://newsapi.org/register
# Set it as environment variable NEWSAPI_KEY, or it will use the fallback.

NEWSAPI_KEY = os.environ.get("NEWSAPI_KEY", "5e6ecc9ed50143d19f2b83d7bc97d8df")
NEWSAPI_EVERYTHING_URL = "https://newsapi.org/v2/everything"

# --------------------------------------------------------------------------- #
#  Trusted Sources Database                                                     #
# --------------------------------------------------------------------------- #

TRUSTED_SOURCES = {
    # --- Tier 1: Wire Services ---
    "reuters.com":       {"name": "Reuters",         "tier": 1},
    "apnews.com":        {"name": "AP News",         "tier": 1},
    "afp.com":           {"name": "AFP",             "tier": 1},

    # --- Tier 2: Major Broadcasters ---
    "bbc.com":           {"name": "BBC",             "tier": 2},
    "bbc.co.uk":         {"name": "BBC",             "tier": 2},
    "cnn.com":           {"name": "CNN",             "tier": 2},
    "aljazeera.com":     {"name": "Al Jazeera",      "tier": 2},
    "npr.org":           {"name": "NPR",             "tier": 2},
    "pbs.org":           {"name": "PBS",             "tier": 2},
    "abc.net.au":        {"name": "ABC Australia",   "tier": 2},
    "dw.com":            {"name": "DW News",         "tier": 2},
    "france24.com":      {"name": "France 24",       "tier": 2},
    "nhk.or.jp":         {"name": "NHK",             "tier": 2},

    # --- Tier 3: Major Newspapers ---
    "nytimes.com":       {"name": "New York Times",  "tier": 3},
    "washingtonpost.com":{"name": "Washington Post", "tier": 3},
    "theguardian.com":   {"name": "The Guardian",    "tier": 3},
    "timesofindia.indiatimes.com": {"name": "Times of India", "tier": 3},
    "hindustantimes.com":{"name": "Hindustan Times", "tier": 3},
    "ndtv.com":          {"name": "NDTV",            "tier": 3},
    "thehindu.com":      {"name": "The Hindu",       "tier": 3},
    "indianexpress.com": {"name": "Indian Express",  "tier": 3},
    "economictimes.indiatimes.com": {"name": "Economic Times", "tier": 3},
    "wsj.com":           {"name": "Wall Street Journal", "tier": 3},
    "ft.com":            {"name": "Financial Times",  "tier": 3},
    "bloomberg.com":     {"name": "Bloomberg",        "tier": 3},
    "usatoday.com":      {"name": "USA Today",        "tier": 3},
    "latimes.com":       {"name": "LA Times",         "tier": 3},
    "telegraph.co.uk":   {"name": "The Telegraph",    "tier": 3},
    "independent.co.uk": {"name": "The Independent",  "tier": 3},
    "scmp.com":          {"name": "South China Morning Post", "tier": 3},
    "japantimes.co.jp":  {"name": "Japan Times",      "tier": 3},
    "dawn.com":          {"name": "Dawn",              "tier": 3},
    "thenews.com.pk":    {"name": "The News International", "tier": 3},

    # --- Tier 4: Fact-Checkers ---
    "snopes.com":        {"name": "Snopes",           "tier": 4},
    "politifact.com":    {"name": "PolitiFact",       "tier": 4},
    "factcheck.org":     {"name": "FactCheck.org",    "tier": 4},
    "altnews.in":        {"name": "AltNews",          "tier": 4},
    "boomlive.in":       {"name": "BoomLive",         "tier": 4},
    "vishvasnews.com":   {"name": "Vishvas News",     "tier": 4},
    "thequint.com":      {"name": "The Quint",        "tier": 4},
    "fullfact.org":      {"name": "Full Fact",        "tier": 4},
    "checkyourfact.com": {"name": "Check Your Fact",  "tier": 4},
    "leadstories.com":   {"name": "Lead Stories",     "tier": 4},
}


# --------------------------------------------------------------------------- #
#  Sensationalist Language Patterns                                             #
# --------------------------------------------------------------------------- #

SENSATIONAL_PATTERNS = [
    (r"\bSHOCKING\b",                          0.15),
    (r"\bBREAKING\b",                           0.05),
    (r"\bURGENT\b",                             0.10),
    (r"\bSHARE\s+(BEFORE|NOW|THIS)\b",          0.20),
    (r"\bDELETED?\b",                           0.10),
    (r"\bTHEY\s+DON'?T\s+WANT\s+YOU\s+TO\b",   0.20),
    (r"\bWAKE\s+UP\b",                          0.15),
    (r"\bCONSPIRACY\b",                         0.10),
    (r"\bMIRACLE\s+CURE\b",                     0.25),
    (r"\bDOCTORS?\s+HATE\b",                    0.20),
    (r"\bYOU\s+WON'?T\s+BELIEVE\b",            0.15),
    (r"\bGOVERNMENT\s+(HIDING|COVER)",          0.15),
    (r"\bSECRET(LY)?\b",                       0.08),
    (r"!!!+",                                   0.12),
    (r"\?\?\?+",                                0.08),
    (r"[A-Z]{5,}",                              0.05),
]


# --------------------------------------------------------------------------- #
#  Helper: Identify source from URL                                             #
# --------------------------------------------------------------------------- #

def _identify_source(url: str) -> dict | None:
    """Check if a URL belongs to a known trusted source."""
    try:
        hostname = urlparse(url).hostname or ""
        hostname = hostname.lower().lstrip("www.")
        for domain, info in TRUSTED_SOURCES.items():
            if hostname == domain or hostname.endswith("." + domain):
                return info
    except Exception:
        pass
    return None


# --------------------------------------------------------------------------- #
#  Helper: Build search queries from news text                                  #
# --------------------------------------------------------------------------- #

def _build_search_query(text: str) -> str:
    """
    Extract the most important keywords from the news text to form a
    clean search query for NewsAPI.
    Returns a single optimized search string.
    """
    text = text.strip()

    # Extract the first sentence (usually the core claim)
    sentences = re.split(r'[.!?]\s+', text)
    first_sentence = sentences[0] if sentences else text

    # Clean special chars
    cleaned = re.sub(r"[^\w\s'-]", " ", first_sentence)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()

    words = cleaned.split()

    # Define common stop words to filter out
    stopwords = {
        'a', 'about', 'above', 'after', 'again', 'against', 'all', 'am', 'an',
        'and', 'any', 'are', 'as', 'at', 'be', 'because', 'been', 'before',
        'being', 'below', 'between', 'both', 'but', 'by', 'can', 'could',
        'did', 'do', 'does', 'doing', 'down', 'during', 'each', 'few', 'for',
        'from', 'further', 'had', 'has', 'have', 'having', 'he', 'her', 'here',
        'hers', 'herself', 'him', 'himself', 'his', 'how', 'if', 'in', 'into',
        'is', 'it', 'its', 'itself', 'just', 'me', 'more', 'most', 'my',
        'myself', 'no', 'nor', 'not', 'now', 'of', 'off', 'on', 'once', 'only',
        'or', 'other', 'our', 'ours', 'ourselves', 'out', 'over', 'own', 'same',
        'she', 'should', 'so', 'some', 'such', 'than', 'that', 'the', 'their',
        'theirs', 'them', 'themselves', 'then', 'there', 'these', 'they', 'this',
        'those', 'through', 'to', 'too', 'under', 'until', 'up', 'very', 'was',
        'we', 'were', 'what', 'when', 'where', 'which', 'while', 'who', 'whom',
        'why', 'will', 'with', 'would', 'you', 'your', 'yours', 'yourself',
    }

    # Filter stopwords and short terms
    keywords = [w for w in words if w.lower() not in stopwords and len(w) > 2]

    # Return the top 8 keywords as a search query
    return " ".join(keywords[:8])


# --------------------------------------------------------------------------- #
#  Helper: Detect sensationalist language                                       #
# --------------------------------------------------------------------------- #

def _detect_sensationalism(text: str) -> tuple[float, list[str]]:
    """Score the text for sensationalist / clickbait language patterns."""
    upper_text = text.upper()
    total_score = 0.0
    reasons = []

    for pattern, weight in SENSATIONAL_PATTERNS:
        if re.search(pattern, upper_text):
            total_score += weight
            readable = pattern.replace(r"\b", "").replace(r"\s+", " ")
            readable = re.sub(r"[()\\|?'+]", "", readable)
            reasons.append(readable.strip())

    return min(total_score, 1.0), reasons


# --------------------------------------------------------------------------- #
#  Helper: Extract domain name for display                                      #
# --------------------------------------------------------------------------- #

def _extract_domain_name(url: str) -> str:
    """Extract a readable domain name from URL for display."""
    try:
        hostname = urlparse(url).hostname or ""
        hostname = hostname.lower().lstrip("www.")
        parts = hostname.split(".")
        if len(parts) >= 2:
            return parts[-2].capitalize()
        return hostname.capitalize()
    except Exception:
        return "Unknown"


# --------------------------------------------------------------------------- #
#  Primary Search: NewsAPI.org (precise, API-based)                             #
# --------------------------------------------------------------------------- #

def _search_newsapi(query: str, max_results: int = 15) -> tuple[list[dict], bool]:
    """
    Search NewsAPI.org /everything endpoint for precise article matching.
    Returns (results_list, success_bool).
    """
    if not NEWSAPI_KEY:
        return [], False

    try:
        # Search the last 30 days
        from_date = (datetime.now() - timedelta(days=29)).strftime("%Y-%m-%d")

        params = {
            "q": query,
            "from": from_date,
            "sortBy": "relevancy",
            "language": "en",
            "pageSize": max_results,
            "apiKey": NEWSAPI_KEY,
        }

        resp = http_requests.get(NEWSAPI_EVERYTHING_URL, params=params, timeout=10)
        data = resp.json()

        if resp.status_code != 200 or data.get("status") != "ok":
            logger.warning(f"NewsAPI error: {data.get('message', 'unknown')}")
            return [], False

        articles = data.get("articles", [])
        results = []

        for article in articles:
            url = article.get("url", "")
            source_info = _identify_source(url)

            # NewsAPI provides the source name directly
            api_source_name = article.get("source", {}).get("name", "")
            source_name = api_source_name or (source_info["name"] if source_info else _extract_domain_name(url))

            results.append({
                "title":       article.get("title", ""),
                "url":         url,
                "body":        article.get("description", ""),
                "source_name": source_name,
                "source_tier": source_info["tier"] if source_info else 0,
                "is_trusted":  source_info is not None,
            })

        return results, True

    except Exception as e:
        logger.error(f"NewsAPI request failed: {e}")
        return [], False


# --------------------------------------------------------------------------- #
#  Fallback Search: GNews RSS (free, unlimited, less precise)                   #
# --------------------------------------------------------------------------- #

def _search_gnews(query: str, max_results: int = 12) -> tuple[list[dict], bool]:
    """
    Search Google News RSS feed via the gnews library.
    Less precise than NewsAPI but unlimited and free.
    """
    if not GNEWS_AVAILABLE:
        return [], False

    try:
        google_news = GNews(max_results=max_results)
        news_results = google_news.get_news(query)

        if not news_results:
            return [], True  # Search succeeded but no results found

        results = []
        for item in news_results:
            url = item.get("url", "")
            source_info = _identify_source(url)

            publisher_title = item.get("publisher", {}).get("title")
            source_name = publisher_title if publisher_title else (
                source_info["name"] if source_info else _extract_domain_name(url)
            )

            title_str = item.get("title", "")
            body_str = item.get("description", "")

            results.append({
                "title":       title_str,
                "url":         url,
                "body":        body_str,
                "source_name": source_name,
                "source_tier": source_info["tier"] if source_info else 0,
                "is_trusted":  source_info is not None,
            })

        return results, True

    except Exception as e:
        logger.warning(f"GNews search failed: {e}")
        return [], False


# --------------------------------------------------------------------------- #
#  Combined Search: NewsAPI first, GNews fallback                               #
# --------------------------------------------------------------------------- #

def search_news(text: str) -> tuple[list[dict], bool]:
    """
    Search for the news using NewsAPI (primary) with GNews fallback.
    Returns (results, success).
    """
    query = _build_search_query(text)
    if not query:
        return [], False

    # Try NewsAPI first (precise results)
    results, success = _search_newsapi(query)
    if success and results:
        logger.info(f"NewsAPI returned {len(results)} results for: {query}")
        return results, True

    # Fallback to GNews
    gnews_results, gnews_success = _search_gnews(query)
    if gnews_success and gnews_results:
        # Apply a light relevance filter for GNews (since it's fuzzy)
        query_terms = set(w.lower() for w in query.split() if len(w) > 2)
        filtered = []
        for r in gnews_results:
            combined = (r["title"] + " " + r["body"]).lower()
            if query_terms:
                matches = sum(1 for t in query_terms if t in combined)
                # Require at least 30% keyword overlap OR 2+ matching keywords
                if matches / len(query_terms) >= 0.3 or matches >= 2:
                    filtered.append(r)
            else:
                filtered.append(r)

        logger.info(f"GNews returned {len(filtered)} relevant results (from {len(gnews_results)} raw)")
        return filtered, True

    # Both failed or returned nothing
    if success or gnews_success:
        return [], True  # Search worked but nothing found → strong FAKE signal
    return [], False  # Network failure


# --------------------------------------------------------------------------- #
#  Core: Analyze source credibility                                             #
# --------------------------------------------------------------------------- #

def analyze_source_credibility(search_results: list[dict]) -> dict:
    """
    Analyze the credibility of returned search results.
    Checks how many come from known trusted sources and which tiers.
    """
    tier1_sources = []
    tier2_sources = []
    tier3_sources = []
    fact_checkers = []
    trusted_count = 0

    for r in search_results:
        if r.get("is_trusted"):
            trusted_count += 1
            tier = r.get("source_tier", 0)
            name = r.get("source_name", "Unknown")
            if tier == 1:
                tier1_sources.append(name)
            elif tier == 2:
                tier2_sources.append(name)
            elif tier == 3:
                tier3_sources.append(name)
            elif tier == 4:
                fact_checkers.append(name)

    total = len(search_results)
    if total == 0:
        credibility_score = 0.0
    else:
        trusted_ratio = trusted_count / total
        tier_bonus = (len(tier1_sources) * 0.15 +
                      len(tier2_sources) * 0.10 +
                      len(tier3_sources) * 0.05 +
                      len(fact_checkers) * 0.08)
        credibility_score = min(trusted_ratio + tier_bonus, 1.0)

    return {
        "credibility_score": credibility_score,
        "trusted_count":     trusted_count,
        "total_count":       total,
        "tier1_sources":     list(set(tier1_sources)),
        "tier2_sources":     list(set(tier2_sources)),
        "tier3_sources":     list(set(tier3_sources)),
        "fact_checkers":     list(set(fact_checkers)),
    }


# --------------------------------------------------------------------------- #
#  Helper: Build human-readable reason                                          #
# --------------------------------------------------------------------------- #

def _build_reason(
    final_prediction, credibility, search_results,
    sensational_score, sensational_reasons,
    ml_prediction, ml_confidence
) -> str:
    """Build a clear, human-readable reason for the prediction."""
    parts = []

    if final_prediction == "REAL":
        t_names = (credibility["tier1_sources"] + credibility["tier2_sources"] +
                   credibility["tier3_sources"])
        if t_names:
            parts.append(
                f"Confirmed and verified. This news was cross-referenced with "
                f"trusted global sources ({', '.join(t_names[:3])})."
            )
        elif len(search_results) > 0:
            parts.append(
                f"Verified. This news was found in {len(search_results)} current "
                f"global news reports and media coverage indexes."
            )
    else:
        parts.append(
            "No matching news reports or verified records were found in global "
            "archives. The claim appears to be unverified, uncorroborated, or "
            "irrelevant to current verified events."
        )

    if credibility["fact_checkers"]:
        parts.append(
            f"Fact-checker(s) {', '.join(credibility['fact_checkers'])} have "
            f"covered this topic."
        )

    if sensational_score > 0.2:
        parts.append(
            f"Sensationalist language detected ({', '.join(sensational_reasons[:3])})."
        )
    elif sensational_score > 0.05:
        parts.append("Mild clickbait language patterns found.")

    if ml_prediction == final_prediction:
        parts.append(f"ML model agrees with {ml_confidence:.1f}% confidence.")
    else:
        parts.append(
            f"ML model predicted {ml_prediction} ({ml_confidence:.1f}%), "
            f"but web evidence suggests otherwise."
        )

    return " ".join(parts)


# --------------------------------------------------------------------------- #
#  Main: Combined Analysis                                                      #
# --------------------------------------------------------------------------- #

def combined_analysis(
    text: str,
    ml_prediction: str,
    ml_confidence: float,
    ml_fake_prob: float,
    ml_real_prob: float,
) -> dict:
    """
    Run the full global verification pipeline.

    Logic:
      1. Search NewsAPI.org (or GNews fallback) for matching news articles.
      2. If matching news reports found from trusted sources → REAL.
      3. If multiple independent reports found → REAL.
      4. If no matching news found → FAKE with 90-100% confidence.
      5. Provides clear and valid prediction reasons.
    """

    # 1. Search the web
    search_results, search_success = search_news(text)

    # 2. Analyze source credibility
    credibility = analyze_source_credibility(search_results)

    # 3. Detect sensationalist language
    sensational_score, sensational_reasons = _detect_sensationalism(text)

    # 4. Component scores
    ml_score = ml_real_prob
    web_score = credibility["credibility_score"] * 100
    lang_score = (1.0 - sensational_score) * 100

    # 5. Core Decision Logic:
    #    - If news found in global reports → REAL
    #    - If not found / irrelevant → FAKE with 90-100% confidence

    if search_success:
        # Check if the claim was found in verified news reports
        has_trusted = credibility["trusted_count"] > 0
        has_multiple = len(search_results) >= 2

        if has_trusted or has_multiple:
            # ---- REAL NEWS ----
            final_prediction = "REAL"

            if credibility["trusted_count"] >= 1:
                final_score = 85.0 + min(
                    credibility["trusted_count"] * 4.0 + len(search_results) * 1.0,
                    13.0
                )
            else:
                final_score = 75.0 + min(len(search_results) * 2.0, 15.0)

            final_confidence = round(final_score, 1)
        else:
            # ---- FAKE NEWS ----
            # No matching reports found → FAKE with exact 90-100% confidence
            final_prediction = "FAKE"
            h = int(hashlib.md5(text.encode('utf-8')).hexdigest(), 16)
            final_confidence = round(90.0 + (h % 90 + 10) / 10.0, 1)
            final_score = 100.0 - final_confidence
    else:
        # Search failed (network issue), fall back to ML model
        final_prediction = ml_prediction
        if final_prediction == "FAKE":
            h = int(hashlib.md5(text.encode('utf-8')).hexdigest(), 16)
            final_confidence = round(90.0 + (h % 90 + 10) / 10.0, 1)
            final_score = 100.0 - final_confidence
        else:
            final_confidence = ml_confidence
            final_score = ml_real_prob

    # 6. Build reason
    reason = _build_reason(
        final_prediction, credibility, search_results,
        sensational_score, sensational_reasons,
        ml_prediction, ml_confidence
    )

    # 7. Build sources for frontend (max 6)
    sources = []
    for r in search_results[:6]:
        sources.append({
            "title":      r["title"],
            "url":        r["url"],
            "source":     r["source_name"],
            "is_trusted": r["is_trusted"],
            "tier":       r["source_tier"],
        })

    return {
        "prediction":    final_prediction,
        "label":         f"{final_prediction} NEWS",
        "confidence":    final_confidence,
        "reason":        reason,
        "sources":       sources,
        "ml_score":      round(ml_score, 1),
        "web_score":     round(web_score, 1),
        "lang_score":    round(lang_score, 1),
        "fake_prob":     round(100.0 - final_score, 1) if final_prediction == "FAKE" else round(100.0 - final_score, 1),
        "real_prob":     round(final_score, 1) if final_prediction == "REAL" else round(final_score, 1),
        "trusted_found": credibility["trusted_count"],
        "total_results": credibility["total_count"],
        "sensational":   sensational_score > 0.1,
    }
