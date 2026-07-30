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
import time
import xml.etree.ElementTree as ET
from urllib.parse import urlparse, quote_plus
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed

import functools
import requests
from nltk.stem import PorterStemmer

GNEWS_AVAILABLE = True

# --------------------------------------------------------------------------- #
#  Logging                                                                      #
# --------------------------------------------------------------------------- #

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
#  NewsAPI Configuration & env loader                                           #
# --------------------------------------------------------------------------- #

def _load_env_file():
    """Manually parse .env file to load NEWSAPI_KEY locally."""
    env_path = os.path.join(os.path.dirname(__file__), '.env')
    if os.path.exists(env_path):
        try:
            with open(env_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith('#') or '=' not in line:
                        continue
                    key, val = line.split('=', 1)
                    key = key.strip()
                    val = val.strip().strip("'\"")
                    os.environ[key] = val
        except Exception as e:
            logger.warning(f"Failed to read .env file manually: {e}")

_load_env_file()

NEWSAPI_KEY = os.environ.get("NEWSAPI_KEY", "")
NEWSAPI_EVERYTHING_URL = "https://newsapi.org/v2/everything"
# Bing News API configuration
BING_API_KEY = os.environ.get("BING_API_KEY", "")
BING_NEWS_URL = "https://api.bing.microsoft.com/v7.0/news/search"


# Mask key for display
if NEWSAPI_KEY:
    masked_key = NEWSAPI_KEY[:6] + "..." + NEWSAPI_KEY[-4:]
    print(f"  [NEWSAPI] Loaded API Key: {masked_key}")
else:
    print("  [NEWSAPI] WARNING: No NEWSAPI_KEY found in .env — web verification will use GNews fallback only")

stemmer = PorterStemmer()


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
    "nbcnews.com":       {"name": "NBC News",        "tier": 2},
    "cbsnews.com":       {"name": "CBS News",        "tier": 2},
    "abcnews.go.com":    {"name": "ABC News",        "tier": 2},
    "foxnews.com":       {"name": "Fox News",        "tier": 2},
    "msnbc.com":         {"name": "MSNBC",           "tier": 2},
    "sky.com":           {"name": "Sky News",        "tier": 2},
    "skynews.com":       {"name": "Sky News",        "tier": 2},
    "euronews.com":      {"name": "Euronews",        "tier": 2},
    "rt.com":            {"name": "RT",              "tier": 2},
    "voanews.com":       {"name": "VOA News",        "tier": 2},

    # --- Tier 3: Major Newspapers & Digital Publications ---
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
    "forbes.com":        {"name": "Forbes",            "tier": 3},
    "cnbc.com":          {"name": "CNBC",              "tier": 3},
    "businessinsider.com":{"name": "Business Insider","tier": 3},
    "techcrunch.com":    {"name": "TechCrunch",        "tier": 3},
    "wired.com":         {"name": "Wired",             "tier": 3},
    "theverge.com":      {"name": "The Verge",         "tier": 3},
    "arstechnica.com":   {"name": "Ars Technica",      "tier": 3},
    "time.com":          {"name": "TIME",              "tier": 3},
    "newsweek.com":      {"name": "Newsweek",          "tier": 3},
    "theatlantic.com":   {"name": "The Atlantic",      "tier": 3},
    "politico.com":      {"name": "Politico",          "tier": 3},
    "axios.com":         {"name": "Axios",             "tier": 3},
    "thehill.com":       {"name": "The Hill",          "tier": 3},
    "marketwatch.com":   {"name": "MarketWatch",       "tier": 3},
    "nature.com":        {"name": "Nature",            "tier": 3},
    "sciencemag.org":    {"name": "Science",           "tier": 3},
    "space.com":         {"name": "Space.com",         "tier": 3},
    "nasa.gov":          {"name": "NASA",              "tier": 3},
    "who.int":           {"name": "WHO",               "tier": 3},
    "un.org":            {"name": "United Nations",    "tier": 3},
    "bworldonline.com":  {"name": "BusinessWorld",     "tier": 3},
    "thetimes.co.uk":    {"name": "The Times",         "tier": 3},
    "standardmedia.co.ke":{"name": "Standard Media",  "tier": 3},
    "dailymail.co.uk":   {"name": "Daily Mail",        "tier": 3},
    "mirror.co.uk":      {"name": "The Mirror",        "tier": 3},
    "huffpost.com":      {"name": "HuffPost",          "tier": 3},
    "vox.com":           {"name": "Vox",               "tier": 3},
    "vice.com":          {"name": "Vice",              "tier": 3},
    "buzzfeednews.com":  {"name": "BuzzFeed News",     "tier": 3},
    "propublica.org":    {"name": "ProPublica",        "tier": 3},

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
    (r"\bCONSPIRACY\b",                         0.15),
    (r"\bMIRACLE\s+CURE\b",                     0.25),
    (r"\bDOCTORS?\s+HATE\b",                    0.20),
    (r"\bYOU\s+WON'?T\s+BELIEVE\b",            0.15),
    (r"\bGOVERNMENT\s+(HIDING|COVER.UP|LIE)",  0.25),
    (r"\bHIDING\s+(THE\s+)?(TRUTH|FACTS?)\b",  0.25),
    (r"\bPOISON(ED|ING)?\s+(WATER|FOOD|SUPPLY)\b", 0.20),
    (r"\bSECRET(LY)?\b",                       0.08),
    (r"\bBRAINWASH(ING)?\b",                   0.30),
    (r"\bMICROCHIP\b",                          0.30),
    (r"\bCONTROL\s+(THE\s+)?INTERNET\b",        0.25),
    (r"!!!+",                                   0.12),
    (r"\?\?\?+",                                0.08),
    (r"[A-Z]{5,}",                              0.05),
]

# Minimum sensationalism score at which, absent trusted sources, we force FAKE
SENSATIONAL_FORCE_FAKE_THRESHOLD = 0.20


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
#  Helper: Determine relevance of article to query using PorterStemmer        #
# --------------------------------------------------------------------------- #

IGNORE_WORDS = {
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
    'new', 'today', 'day', 'month', 'year', 'announced', 'announces', 'report',
    'reports', 'says', 'official', 'officials', 'government', 'statement',
    'latest', 'update', 'updates', 'recent', 'recently', 'current', 'currently',
    'yesterday', 'tomorrow'
}

IGNORE_WORDS_STEMMED = {stemmer.stem(w) for w in IGNORE_WORDS}

DEBUNK_WORDS = {"hoax", "rumor", "rumours", "fake", "debunk", "false", "misleading", "factcheck", "fact-check", "untrue", "edited", "morphed"}
CONTEXT_SHIFT_WORDS = {"condoles", "condolence", "condolences", "tribute", "tributes", "grief", "mourns", "mourning", "expresses"}

def _extract_entities(text: str) -> list[str]:
    """
    Extract likely proper nouns / named entities from text.
    Uses capitalization patterns since we don't have spaCy.
    Returns lowercased entity tokens.
    """
    # Find sequences of capitalized words (proper nouns)
    entities = []
    words = text.split()
    i = 0
    while i < len(words):
        w = words[i]
        # Skip sentence-starting capitalization heuristic: skip first word
        clean = re.sub(r"[^a-zA-Z0-9'-]", "", w)
        if clean and clean[0].isupper() and clean.lower() not in IGNORE_WORDS and len(clean) > 1:
            entity_parts = [clean.lower()]
            # Extend with consecutive capitalized words
            j = i + 1
            while j < len(words):
                nw = re.sub(r"[^a-zA-Z0-9'-]", "", words[j])
                if nw and nw[0].isupper() and len(nw) > 1:
                    entity_parts.append(nw.lower())
                    j += 1
                else:
                    break
            if len(entity_parts) >= 1:
                entities.extend(entity_parts)
            i = j
        else:
            i += 1
    return entities


def _extract_specific_claims(text: str) -> dict:
    """
    Extract specific, verifiable claims from text:
    - Numbers / monetary amounts
    - Key action verbs (buys, arrested, killed, dies, launches, etc.)
    Returns a dict with claim components.
    """
    claims = {
        "numbers": [],
        "action_verbs": [],
        "key_phrases": [],
    }

    # Extract numbers and monetary amounts
    numbers = re.findall(r'\b\d[\d,]*(?:\.\d+)?\s*(?:billion|million|trillion|thousand|crore|lakh)?\b', text.lower())
    claims["numbers"] = [n.strip() for n in numbers if len(n.strip()) > 0]

    # Extract key action verbs that indicate specific events
    action_patterns = [
        r'\b(buys?|bought|purchase[ds]?|acquir(?:e[ds]?|ing))\b',
        r'\b(arrest(?:ed|s|ing)?|detained|jailed|imprisoned)\b',
        r'\b(die[ds]?|dead|death|killed|murder(?:ed)?|assassinat(?:ed|ion))\b',
        r'\b(launch(?:ed|es|ing)?|announc(?:ed|es|ing)?|reveal(?:ed|s|ing)?)\b',
        r'\b(resign(?:ed|s|ing)?|fire[ds]?|sack(?:ed)?|step(?:ped)?\s+down)\b',
        r'\b(ban(?:ned|s)?|block(?:ed|s)?|sanction(?:ed|s)?)\b',
        r'\b(win[s]?|won|defeat(?:ed|s)?|beat[s]?)\b',
        r'\b(invad(?:e[ds]?|ing)|attack(?:ed|s|ing)?|bomb(?:ed|s|ing)?)\b',
        r'\b(collaps(?:e[ds]?|ing)|crash(?:ed|es)?|bankrupt(?:cy)?)\b',
        r'\b(elect(?:ed|ion)?|vote[ds]?|inaugurat(?:ed|es|ion))\b',
    ]
    text_lower = text.lower()
    for pattern in action_patterns:
        matches = re.findall(pattern, text_lower)
        claims["action_verbs"].extend(matches)

    return claims


def _is_debunking_article(title: str, description: str) -> bool:
    """Check if an article is debunking/fact-checking a claim (not confirming it)."""
    combined = (title + " " + (description or "")).lower()
    return any(w in combined for w in DEBUNK_WORDS)


def is_relevant_article(query: str, title: str, description: str) -> tuple[bool, str]:
    """
    Checks if a returned news article is relevant to the search query.
    Returns a tuple: (is_relevant, relevance_type)
      - relevance_type: "confirming" if article confirms the claim,
                        "debunking" if article is fact-checking/debunking,
                        "none" if not relevant
    Uses multi-layered matching:
    1. Entity overlap (proper nouns must match)
    2. Adaptive stemmed keyword overlap
    3. Claim-specificity verification (numbers, action verbs)
    4. Context shift detection
    5. Action verb match requirement (stricter)
    """
    if not title:
        return False, "none"
        
    title_lower = title.lower()
    query_lower = query.lower()
    
    # 1. Debunk & Context shift word checks
    has_death_query = any(w in query_lower for w in ["dead", "death", "dies", "passed away", "killed", "dying"])
    has_shift_title = any(w in title_lower for w in CONTEXT_SHIFT_WORDS)
    if has_death_query and has_shift_title:
        return False, "none"
    
    # 2. Extract entities from query and check entity overlap
    query_entities = _extract_entities(query)
    if query_entities:
        combined_text = (title + " " + (description or "")).lower()
        entity_matches = sum(1 for e in query_entities if e in combined_text)
        entity_ratio = entity_matches / len(query_entities) if query_entities else 0
        
        # If query has named entities but NONE appear in the article, it's not relevant
        if entity_ratio == 0 and len(query_entities) >= 2:
            return False, "none"
    
    # 3. Extract specific claims and verify them
    query_claims = _extract_specific_claims(query)
    article_text = (title + " " + (description or "")).lower()
    
    # If query has specific numbers, check if any appear in the article
    if query_claims["numbers"]:
        has_number_match = any(n in article_text for n in query_claims["numbers"])
        specific_numbers = [n for n in query_claims["numbers"] if not re.match(r'^20\d\d$', n.strip())]
        if specific_numbers and not has_number_match:
            return False, "none"
    
    # If query has specific action verbs, check if the article has similar actions
    if query_claims["action_verbs"]:
        action_stems = {stemmer.stem(v) for v in query_claims["action_verbs"]}
        article_words = re.sub(r"[^\w\s'-]", " ", article_text).split()
        article_stems = {stemmer.stem(w) for w in article_words}
        action_match = bool(action_stems & article_stems)
    else:
        action_match = True  # No specific actions to check
        
    # 4. Clean and stem title and description
    title_clean = re.sub(r"[^\w\s'-]", " ", title).lower()
    desc_clean = re.sub(r"[^\w\s'-]", " ", description or "").lower()
    combined_words = (title_clean + " " + desc_clean).split()
    
    article_stemmed_words = {stemmer.stem(w) for w in combined_words}
    
    # 5. Clean and stem the query
    query_clean = re.sub(r"[^\w\s'-]", " ", query).lower()
    query_words = query_clean.split()
    
    # 6. Extract core query keywords
    query_keywords = [
        stemmer.stem(w) for w in query_words 
        if w not in IGNORE_WORDS and stemmer.stem(w) not in IGNORE_WORDS_STEMMED and len(w) > 2
    ]
    if not query_keywords:
        query_keywords = [
            stemmer.stem(w) for w in query_words 
            if len(w) > 2
        ]
        if not query_keywords:
            return False, "none"
            
    # 7. Count matches
    matches = sum(1 for kw in query_keywords if kw in article_stemmed_words)
    
    # 8. Apply STRICTER adaptive matching threshold (raised thresholds)
    n_keys = len(query_keywords)
    if n_keys == 1:
        keyword_pass = matches >= 1
    elif n_keys == 2:
        keyword_pass = matches >= 2
    elif n_keys <= 4:
        keyword_pass = matches >= 2 and (matches / n_keys) >= 0.55
    elif n_keys <= 6:
        keyword_pass = matches >= 3 and (matches / n_keys) >= 0.50
    else:
        keyword_pass = matches >= 4 and (matches / n_keys) >= 0.40
    
    # 9. Combined decision: keywords must pass
    if not keyword_pass:
        return False, "none"
    
    # 10. If we had specific action verbs but none matched, the article is
    #     about the same entity but a DIFFERENT event → NOT truly relevant
    if not action_match and n_keys >= 3:
        if (matches / n_keys) < 0.7:
            return False, "none"
    
    # 11. Determine if this is a debunking article or a confirming one
    is_debunk = _is_debunking_article(title, description or "")
    relevance_type = "debunking" if is_debunk else "confirming"
    
    return True, relevance_type


# --------------------------------------------------------------------------- #
#  Helper: Build search queries from news text                                  #
# --------------------------------------------------------------------------- #

def _build_search_queries(text: str) -> list[str]:
    """
    Extract key terms to form clean search queries.
    Returns a list of candidate search queries ordered from specific to broader.
    """
    text = text.strip()
    sentences = re.split(r'[.!?]\s+', text)
    first_sentence = sentences[0] if sentences else text

    cleaned = re.sub(r"[^\w\s'-]", " ", first_sentence)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    words = cleaned.split()

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

    keywords = [w for w in words if w.lower() not in stopwords and len(w) > 2]
    entities = _extract_entities(text)
    years_nums = [w for w in keywords if re.match(r'^\d{4}$', w) or (w.isdigit() and len(w) >= 2)]

    queries = []

    # 1. Full keywords query (up to 5 terms)
    if len(keywords) >= 3:
        q_full = " ".join(keywords[:5])
        queries.append(q_full)

    # 2. Entity-focused query including numbers/years
    if len(entities) >= 2:
        ent_parts = entities[:3] + [n for n in years_nums if n not in entities]
        q_ent = " ".join(ent_parts[:4])
        if q_ent not in queries:
            queries.append(q_ent)

    # 3. Top 3 key terms
    if len(keywords) >= 2:
        q_short = " ".join(keywords[:3])
        if q_short not in queries:
            queries.append(q_short)

    if not queries:
        queries.append(" ".join(keywords[:4]) if keywords else text[:50])

    return queries


def _build_search_query(text: str) -> str:
    """Legacy wrapper returning the primary query string."""
    q_list = _build_search_queries(text)
    return q_list[0] if q_list else text[:50]


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
        from_date = (datetime.now() - timedelta(days=29)).strftime("%Y-%m-%d")

        params = {
            "q": query,
            "from": from_date,
            "sortBy": "relevancy",
            "language": "en",
            "pageSize": max_results,
            "apiKey": NEWSAPI_KEY,
        }

        resp = requests.get(NEWSAPI_EVERYTHING_URL, params=params, timeout=10)
        data = resp.json()

        if resp.status_code != 200 or data.get("status") != "ok":
            logger.warning(f"NewsAPI error: {data.get('message', 'unknown')}")
            return [], False

        articles = data.get("articles", [])
        results = []

        for article in articles:
            url = article.get("url", "")
            source_info = _identify_source(url)

            api_source_name = article.get("source", {}).get("name", "")
            source_name = api_source_name or (source_info["name"] if source_info else _extract_domain_name(url))

            results.append({
                "title":       article.get("title", ""),
                "url":         url,
                "body":        article.get("description", ""),
                "source_name": source_name,
                "source_tier": source_info["tier"] if source_info else 0,
                "is_trusted":  source_info is not None,
                "api_source":  "newsapi",
            })

        return results, True

    except Exception as e:
        logger.error(f"NewsAPI request failed: {e}")
        return [], False


# --------------------------------------------------------------------------- #
#  Fallback Search: GNews RSS (free, unlimited, less precise)                   #
# --------------------------------------------------------------------------- #

def _search_bing(query: str, max_results: int = 10) -> tuple[list[dict], bool]:
    """
    Search Bing News API for matching articles.
    Returns (results_list, success_bool).
    Requires BING_API_KEY in .env.
    """
    if not BING_API_KEY:
        return [], False

    try:
        headers = {"Ocp-Apim-Subscription-Key": BING_API_KEY}
        params = {
            "q": query,
            "count": max_results,
            "mkt": "en-US",
            "freshness": "Month",
            "sortBy": "Relevance",
        }
        resp = requests.get(BING_NEWS_URL, headers=headers, params=params, timeout=10)
        if resp.status_code != 200:
            logger.warning(f"Bing News API error: {resp.status_code}")
            return [], False

        data = resp.json()
        articles = data.get("value", [])
        results = []

        for article in articles:
            url = article.get("url", "")
            source_info = _identify_source(url)
            provider = article.get("provider", [{}])
            source_name = provider[0].get("name", "") if provider else ""
            if not source_name:
                source_name = source_info["name"] if source_info else _extract_domain_name(url)

            results.append({
                "title":       article.get("name", ""),
                "url":         url,
                "body":        article.get("description", ""),
                "source_name": source_name,
                "source_tier": source_info["tier"] if source_info else 0,
                "is_trusted":  source_info is not None,
                "api_source":  "bing",
            })

        return results, True

    except Exception as e:
        logger.error(f"Bing News API request failed: {e}")
        return [], False


def _search_gnews(query: str, max_results: int = 8, _retry: int = 1) -> tuple[list[dict], bool]:
    """
    Search Google News RSS feed directly by parsing the XML.
    """
    try:
        url = f"https://news.google.com/rss/search?q={quote_plus(query)}&hl=en-US&gl=US&ceid=US:en"
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        }
        resp = requests.get(url, headers=headers, timeout=10)
        if resp.status_code != 200:
            logger.warning(f"Google News RSS request failed with status code: {resp.status_code}")
            return [], False
        
        root = ET.fromstring(resp.content)
        items = root.findall(".//item")
        
        results = []
        for item in items[:max_results]:
            title_el = item.find("title")
            link_el = item.find("link")
            source_el = item.find("source")
            
            title_str = title_el.text if title_el is not None else ""
            article_url = link_el.text if link_el is not None else ""
            
            source_name = ""
            source_url = ""
            if source_el is not None:
                source_name = source_el.text or ""
                source_url = source_el.attrib.get("url", "")
            
            if not source_name and " - " in title_str:
                parts = title_str.rsplit(" - ", 1)
                source_name = parts[1]
                title_str = parts[0]
            
            source_info = _identify_source(source_url) if source_url else _identify_source(article_url)
            if not source_name and source_info:
                source_name = source_info["name"]
            if not source_name:
                source_name = _extract_domain_name(source_url or article_url)
                
            results.append({
                "title":       title_str,
                "url":         article_url,
                "body":        "",
                "source_name": source_name,
                "source_tier": source_info["tier"] if source_info else 0,
                "is_trusted":  source_info is not None,
                "api_source":  "gnews",
            })
            
        return results, True
        
    except (ConnectionError, OSError) as e:
        if _retry > 0:
            logger.warning(f"GNews transient error, retrying... ({e})")
            time.sleep(1.5)
            return _search_gnews(query, max_results=max_results, _retry=_retry - 1)
        logger.warning(f"GNews search failed after retry: {e}")
        return [], False
    except Exception as e:
        logger.warning(f"GNews search failed: {e}")
        return [], False


# --------------------------------------------------------------------------- #
#  Combined Search: ALL sources in parallel, merge & deduplicate                #
# --------------------------------------------------------------------------- #

def _deduplicate_results(results: list[dict]) -> list[dict]:
    seen_urls = set()
    unique = []
    for r in results:
        url = r.get("url", "").rstrip("/").lower()
        if url and url not in seen_urls:
            seen_urls.add(url)
            unique.append(r)
        elif not url:
            unique.append(r)
    return unique


def search_news(text: str) -> tuple[list[dict], bool]:
    """
    Search for news across ALL available sources (NewsAPI, Bing, GNews)
    in parallel, trying candidate queries until RELEVANT results are found.
    Returns (results, success).
    """
    candidate_queries = _build_search_queries(text)
    if not candidate_queries:
        return [], False

    all_results = []
    any_success = False

    for idx, query in enumerate(candidate_queries):
        logger.info(f"Searching ALL sources (attempt {idx+1}/{len(candidate_queries)}): {query}")
        print(f"  [SEARCH] Query (attempt {idx+1}): {query}")

        query_results = []
        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = {}

            if NEWSAPI_KEY:
                futures[executor.submit(_search_newsapi, query, 15)] = "NewsAPI"

            if BING_API_KEY:
                futures[executor.submit(_search_bing, query, 10)] = "Bing"

            futures[executor.submit(_search_gnews, query, 10)] = "GNews"

            for future in as_completed(futures, timeout=12):
                source_name = futures[future]
                try:
                    results, success = future.result()
                    if success:
                        any_success = True
                        query_results.extend(results)
                        print(f"  [SEARCH] {source_name}: {len(results)} results")
                    else:
                        print(f"  [SEARCH] {source_name}: failed or 0 results")
                except Exception as e:
                    logger.warning(f"{source_name} search error: {e}")

        query_results = _deduplicate_results(query_results)
        if query_results:
            # Check if any article in query_results is relevant to the text
            relevant = [r for r in query_results if is_relevant_article(text, r["title"], r.get("body", ""))[0]]
            if relevant:
                all_results.extend(query_results)
                break  # Found relevant articles!
            else:
                if not all_results:
                    all_results.extend(query_results)

    all_results = _deduplicate_results(all_results)
    total = len(all_results)
    print(f"  [SEARCH] Total unique results from all sources: {total}")

    if any_success:
        return all_results, True
    else:
        return [], False


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

    Improved Logic (ML-first approach):
      1. Search the web for matching news articles.
      2. Filter for strictly relevant articles, distinguishing confirming vs debunking.
      3. ML model is the primary signal; web evidence can reinforce or override
         ONLY with strong, specific evidence.
      4. Web can override ML FAKE only if there are confirming trusted sources.
      5. Web can override ML REAL only if debunking evidence is found.
      6. Sensationalist language is a strong FAKE signal.
    """

    # 1. Search the web
    search_results, search_success = search_news(text)

    # Filter search results for actual relevance + classify as confirming/debunking
    relevant_results = []
    confirming_results = []    # Articles that confirm the claim
    debunking_results = []     # Articles that debunk/fact-check the claim
    has_mainstream_confirming = False
    has_fact_checker = False

    for r in search_results:
        is_relevant, rel_type = is_relevant_article(text, r["title"], r["body"])
        if is_relevant:
            r["_relevance_type"] = rel_type
            relevant_results.append(r)

            if rel_type == "debunking":
                debunking_results.append(r)
                # Fact-checker sources debunking → strong FAKE signal
                if r.get("is_trusted") and r.get("source_tier") == 4:
                    has_fact_checker = True
            else:
                confirming_results.append(r)
                if r.get("is_trusted"):
                    tier = r.get("source_tier", 0)
                    if tier in (1, 2, 3):
                        has_mainstream_confirming = True
                    elif tier == 4:
                        has_fact_checker = True

    # 2. Analyze source credibility of CONFIRMING articles only
    credibility = analyze_source_credibility(confirming_results)

    # 3. Detect sensationalist language
    sensational_score, sensational_reasons = _detect_sensationalism(text)

    # 4. Component scores — ML gets primary weight
    ml_score = ml_real_prob
    web_score = credibility["credibility_score"] * 100
    lang_score = (1.0 - sensational_score) * 100

    # 5. Core Decision Logic — ML-first approach
    #
    # Key principle: ML model is well-trained and accurate.
    # Web evidence should REINFORCE ML, not blindly override it.
    # Web can override ML only with STRONG, SPECIFIC evidence.

    highly_sensational = sensational_score >= SENSATIONAL_FORCE_FAKE_THRESHOLD
    ml_says_fake_strongly = ml_prediction == "FAKE" and ml_confidence >= 70.0
    ml_says_real_strongly = ml_prediction == "REAL" and ml_confidence >= 70.0

    if search_success:
        # ---- FAKE signals (checked first) ----

        if has_fact_checker or len(debunking_results) >= 1:
            # Fact-checker or debunking article matched this claim → FAKE
            final_prediction = "FAKE"
            final_confidence = 96.0 if has_fact_checker else 90.0
            final_score = 100.0 - final_confidence

        elif highly_sensational and not has_mainstream_confirming:
            # Sensational language with zero trusted confirmation → FAKE
            final_prediction = "FAKE"
            h = int(hashlib.md5(text.encode('utf-8')).hexdigest(), 16)
            final_confidence = round(91.0 + (h % 80 + 10) / 10.0, 1)
            final_score = 100.0 - final_confidence

        elif ml_says_fake_strongly and not has_mainstream_confirming:
            # ML says FAKE with high confidence AND no trusted source confirms it
            # → Trust ML regardless of how many non-trusted results exist
            final_prediction = "FAKE"
            # Boost confidence if sensational language also present
            sens_boost = min(sensational_score * 10.0, 5.0)
            h = int(hashlib.md5(text.encode('utf-8')).hexdigest(), 16)
            base_conf = ml_confidence + sens_boost
            final_confidence = round(min(base_conf, 98.0), 1)
            final_score = 100.0 - final_confidence

        # ---- REAL signals ----

        elif has_mainstream_confirming and len(confirming_results) >= 2:
            # Strong evidence: trusted mainstream sources + multiple confirms → REAL
            # But only override ML FAKE if web evidence is truly strong
            if ml_says_fake_strongly:
                # ML says FAKE but web has strong confirmation → web wins but lower confidence
                final_prediction = "REAL"
                final_score = 75.0 + min(
                    credibility["trusted_count"] * 3.0 + len(confirming_results) * 1.5,
                    15.0
                )
                final_confidence = round(final_score, 1)
            else:
                final_prediction = "REAL"
                final_score = 88.0 + min(
                    credibility["trusted_count"] * 3.0 + len(confirming_results) * 1.5,
                    11.0
                )
                final_confidence = round(final_score, 1)

        elif has_mainstream_confirming and len(confirming_results) >= 1:
            # Moderate evidence: trusted source with one confirming match
            if ml_says_fake_strongly:
                # ML says FAKE, only 1 trusted confirmation → ML wins
                final_prediction = "FAKE"
                h = int(hashlib.md5(text.encode('utf-8')).hexdigest(), 16)
                final_confidence = round(max(ml_confidence - 5.0, 70.0), 1)
                final_score = 100.0 - final_confidence
            else:
                final_prediction = "REAL"
                final_score = 82.0 + min(
                    credibility["trusted_count"] * 3.0 + len(confirming_results) * 2.0,
                    15.0
                )
                final_confidence = round(final_score, 1)

        elif len(confirming_results) >= 2 and not ml_says_fake_strongly:
            # Multiple confirms from non-trusted sources, ML does NOT say FAKE → lean REAL
            final_prediction = "REAL" if ml_prediction == "REAL" else ml_prediction
            if final_prediction == "REAL":
                final_score = 70.0 + min(len(confirming_results) * 3.0, 18.0)
                final_confidence = round(final_score, 1)
            else:
                final_confidence = ml_confidence
                final_score = 100.0 - final_confidence

        elif len(confirming_results) >= 2 and ml_says_fake_strongly:
            # Multiple confirms from non-trusted sources BUT ML says FAKE strongly
            # → Trust ML — non-trusted sources don't override strong ML FAKE
            final_prediction = "FAKE"
            h = int(hashlib.md5(text.encode('utf-8')).hexdigest(), 16)
            final_confidence = round(max(ml_confidence - 5.0, 72.0), 1)
            final_score = 100.0 - final_confidence

        elif len(confirming_results) == 1:
            # Single confirm → weak web signal, trust ML
            final_prediction = ml_prediction
            final_confidence = ml_confidence
            if final_prediction == "FAKE":
                h = int(hashlib.md5(text.encode('utf-8')).hexdigest(), 16)
                final_confidence = round(max(ml_confidence, 70.0), 1)
                final_score = 100.0 - final_confidence
            else:
                final_score = min(65.0 + ml_confidence * 0.15, 78.0)
                final_confidence = round(final_score, 1)

        else:
            # No relevant web results found
            if ml_prediction == "REAL":
                if highly_sensational:
                    # Sensationalist language + zero web proof → FAKE
                    final_prediction = "FAKE"
                    h = int(hashlib.md5(text.encode('utf-8')).hexdigest(), 16)
                    final_confidence = round(85.0 + (h % 80 + 10) / 10.0, 1)
                    final_score = 100.0 - final_confidence
                else:
                    # Clean text predicted REAL by ML → respect ML REAL prediction
                    final_prediction = "REAL"
                    final_confidence = round(max(ml_confidence, 65.0), 1)
                    final_score = final_confidence
            else:
                # ML says FAKE + no web corroboration → FAKE
                final_prediction = "FAKE"
                h = int(hashlib.md5(text.encode('utf-8')).hexdigest(), 16)
                final_confidence = round(max(ml_confidence + 3.0, 85.0), 1)
                final_confidence = min(final_confidence, 98.0)
                final_score = 100.0 - final_confidence
    else:
        # Search failed (network issue), fall back to ML model entirely
        final_prediction = ml_prediction
        final_confidence = ml_confidence
        if final_prediction == "FAKE":
            h = int(hashlib.md5(text.encode('utf-8')).hexdigest(), 16)
            final_confidence = round(max(ml_confidence, 85.0), 1)
            final_score = 100.0 - final_confidence
        else:
            final_score = ml_real_prob

    # 6. Build reason
    reason = _build_reason(
        final_prediction, credibility, relevant_results,
        sensational_score, sensational_reasons,
        ml_prediction, ml_confidence
    )

    # 7. Build sources for frontend (max 6, relevant results only)
    sources = []
    for r in relevant_results[:6]:
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
