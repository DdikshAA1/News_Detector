"""
generate_dataset.py
-------------------
This script creates a synthetic Fake vs Real News dataset for training.
In a real project, you would download the ISOT Fake News Dataset from:
  https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset

WHY SYNTHETIC DATA?
  We use synthetic data so the project works immediately, without needing
  Kaggle credentials. The patterns below are based on real research about
  what distinguishes fake news from real news.

HOW TO USE THE REAL DATASET:
  1. Download 'Fake.csv' and 'True.csv' from Kaggle.
  2. Place them in the /data folder.
  3. In train.py, set USE_REAL_DATASET = True and the script will load them.
"""

import pandas as pd
import random

random.seed(42)

# ------------------------------------------------------------------ #
#  FAKE NEWS PATTERNS                                                  #
#  Fake news often uses shocking language, unverified claims,         #
#  emotional/sensational wording, and ALL CAPS words.                 #
# ------------------------------------------------------------------ #
FAKE_TEMPLATES = [
    # --- Classic sensationalist ---
    "BREAKING: {subject} {fake_verb} {object} — Government {hide_verb} Truth",
    "You Won't Believe What {subject} Just Did to {object}",
    "SHOCKING: {subject} {fake_verb} {object} While Media Stays Silent",
    "EXPOSED: The Real Reason {subject} {fake_verb} {object}",
    "{subject} {fake_verb} {object} and Nobody is Talking About It",
    "ALERT: {subject} Plans to {action} Every {object} by End of Year",
    "Secret Footage Shows {subject} {fake_verb} {object} Behind Closed Doors",
    "They Don't Want You to Know: {subject} {fake_verb} {object}",
    "URGENT WARNING: {subject} {fake_verb} {object} — Share Before Deleted",
    "CONFIRMED: {subject} {fake_verb} {object} According to Anonymous Sources",
    "Top Doctor Reveals: {subject} Can {action} Your {object} Overnight",
    "LEAKED: Internal Documents Prove {subject} {fake_verb} {object}",
    "{subject} {fake_verb} {object} — This Changes EVERYTHING",
    "The Elite Don't Want You to Know How {subject} {fake_verb} {object}",
    "MUST READ: {subject} {fake_verb} {object} at Secret Meeting",
    "Whistleblower Exposes How {subject} {fake_verb} {object} for Years",
    "BANNED VIDEO: {subject} Admits to {fake_verb} {object}",
    "New Study Proves {subject} {fake_verb} {object} (Mainstream Media Ignores)",
    "{subject} {fake_verb} {object} — Are You Prepared for What Comes Next?",
    "CENSORED: The Truth About How {subject} {fake_verb} {object}",
    "Deep State Caught {fake_verb} {object} While {subject} Looks the Other Way",
    "100% PROOF: {subject} Has Been {fake_verb} {object} This Entire Time",
    "WAKE UP: {subject} Is {fake_verb} {object} Right in Front of Your Eyes",
    "Hollywood Elites Furious After {subject} {fake_verb} {object}",
    "Miracle Cure: How {subject} Used {object} to {action} the Impossible",
    "Aliens have landed their {object} in central park and met {subject}",
    "{subject} admits they are building a matrix to {action} {object}",
    "This one trick by {subject} to {action} {object} will blow your mind",
    "REVEALED: {subject} creates mind control devices in {object}",
    "The end is near: {subject} triggers {object} to {action} humanity",

    # --- Plausible-sounding fake news (harder to detect) ---
    "{fake_person} has been arrested by FBI for {fake_crime}",
    "{fake_person} announces purchase of {fake_company} for {fake_amount}",
    "{fake_person} found dead in mysterious circumstances at private estate",
    "EXCLUSIVE: {fake_person} secretly married to {fake_person2} according to insiders",
    "Sources: {fake_person} to resign from all positions effective immediately",
    "{fake_country} declares war on {fake_country2} in surprise midnight announcement",
    "{fake_person} tests positive for {fake_disease}, rushed to hospital in critical condition",
    "JUST IN: {fake_city} completely destroyed by {fake_disaster}",
    "{fake_company} stock crashes 90% after CEO {fake_person} confesses to massive fraud",
    "{fake_person} caught on hidden camera admitting to {fake_crime}",
    "Scientists at {fake_org} confirm {fake_discovery}",
    "LEAKED emails reveal {fake_person} planned to {action} {object} since 2018",
    "Former employee reveals {fake_company} has been {fake_verb} customers for years",
    "Anonymous sources within {fake_org} claim {fake_person} {fake_verb} {object}",
    "{fake_person} paid {fake_amount} to cover up {fake_scandal}",
    "Internal memo shows {fake_company} knew about dangers but hid evidence",
    "Viral video proves {fake_person} was lying about {object} all along",
    "{fake_country} bans all {object} starting next month — citizens outraged",
    "Hospital workers confirm {fake_person} has only {fake_time} to live",
    "BREAKING: {fake_city} airport shut down after {fake_person} causes bomb scare",

    # --- Health misinformation ---
    "Doctors HATE this: {object} can cure {fake_disease} in just 3 days",
    "WARNING: {object} causes {fake_disease} — what {subject} is hiding from you",
    "Scientists finally admit {object} is the real cause of {fake_disease}",
    "EXPOSED: Big pharma suppresses natural cure using {object} for {fake_disease}",
    "Your doctor won't tell you this simple trick to reverse {fake_disease}",
    "New research links {object} to 500% increase in {fake_disease} cases",
    "Government adds secret chemicals to {object} that cause {fake_disease}",
    "Celebrity endorses miracle {object} treatment that cures {fake_disease}",

    # --- Financial / economic fake news ---
    "{fake_company} to go bankrupt within weeks according to leaked documents",
    "{fake_person} secretly moves {fake_amount} to offshore accounts",
    "URGENT: Banks will freeze all accounts starting {fake_date} — withdraw now",
    "Currency to become worthless by {fake_date} — insider warning",
    "{fake_person} caught manipulating {fake_company} stock for personal gain",
    "Secret government plan to tax all {object} by 200% revealed",

    # --- Political conspiracy ---
    "Election RIGGED: Millions of fake ballots found in {fake_city} warehouse",
    "{fake_person} running secret {object} operation from underground bunker",
    "PROOF that {fake_person} is actually controlled by {subject}",
    "Classified document reveals {fake_country} plans to invade {fake_country2}",
    "Shadow government confirms plan to {action} all {object} by end of year",
]

FAKE_SUBJECTS = [
    "the government", "scientists", "big pharma", "the deep state",
    "mainstream media", "the president", "elites", "globalists",
    "secret societies", "tech giants", "the CDC", "the FBI",
    "world leaders", "billionaires", "the establishment", "shadow groups",
    "the CIA", "Illuminati", "freemasons", "reptilians", "AI overlords",
    "the World Economic Forum", "Bill Gates", "the royals",
    "China", "Russia", "North Korea", "the Pentagon",
]

FAKE_VERBS = [
    "is hiding", "has been manipulating", "secretly controls",
    "is suppressing", "has been poisoning", "is brainwashing",
    "has been lying about", "is orchestrating", "secretly banned",
    "has been funding", "is deliberately causing", "has been silencing",
    "is tracking", "has been spying on", "is harvesting", "is replacing",
    "is destroying", "has been stealing", "is censoring",
]

FAKE_PERSONS = [
    "Elon Musk", "Bill Gates", "Jeff Bezos", "Mark Zuckerberg",
    "Donald Trump", "Barack Obama", "Joe Biden", "Hillary Clinton",
    "Vladimir Putin", "Kim Jong-un", "Boris Johnson", "Xi Jinping",
    "Taylor Swift", "Tom Cruise", "Oprah Winfrey", "Kanye West",
    "Warren Buffett", "George Soros", "Tim Cook", "Sundar Pichai",
    "Narendra Modi", "Rahul Gandhi", "Amit Shah", "Kejriwal",
]

FAKE_PERSONS2 = [
    "a foreign intelligence agent", "an alien diplomat",
    "a secret society leader", "a billionaire heiress",
    "a royal family member", "a Hollywood producer",
]

FAKE_COMPANIES = [
    "Google", "Apple", "Microsoft", "Amazon", "Tesla",
    "Facebook", "Twitter", "Netflix", "Disney", "Samsung",
    "TikTok", "Snapchat", "Intel", "Boeing", "McDonald's",
    "Coca-Cola", "Pfizer", "Moderna", "SpaceX", "Uber",
]

FAKE_AMOUNTS = [
    "500 billion dollars", "1 trillion dollars", "50 billion dollars",
    "200 million dollars", "10 billion euros", "100 billion dollars",
    "75 billion dollars", "300 million dollars", "2 trillion dollars",
]

FAKE_CRIMES = [
    "money laundering and tax fraud", "secret human trafficking operations",
    "stealing classified government data", "running an underground spy network",
    "bribing foreign officials", "illegally manipulating stock markets",
    "operating a Ponzi scheme worth billions", "hacking government databases",
    "conspiracy against the United States", "selling state secrets to enemies",
]

FAKE_COUNTRIES = [
    "United States", "China", "Russia", "India", "North Korea",
    "Iran", "Pakistan", "Japan", "Brazil", "Australia",
    "United Kingdom", "France", "Germany", "Saudi Arabia", "Turkey",
]

FAKE_CITIES = [
    "New York", "Los Angeles", "London", "Tokyo", "Mumbai",
    "Paris", "Beijing", "Dubai", "Moscow", "Sydney",
    "Houston", "Chicago", "San Francisco", "Delhi", "Bangalore",
]

FAKE_DISEASES = [
    "cancer", "autism", "heart disease", "diabetes",
    "brain damage", "infertility", "blindness", "paralysis",
    "memory loss", "chronic fatigue", "kidney failure",
]

FAKE_DISASTERS = [
    "massive earthquake", "nuclear explosion", "category 6 hurricane",
    "tsunami", "volcanic eruption", "meteor strike",
    "chemical weapon attack", "massive EMP blast",
]

FAKE_DISCOVERIES = [
    "that Earth is actually flat", "aliens living among humans",
    "time travel is possible", "the moon is an artificial structure",
    "free energy has been suppressed for decades",
    "humans can live to 500 with this secret pill",
    "gravity is not real", "the sun is actually cold",
]

FAKE_ORGS = [
    "NASA", "WHO", "CIA", "Pentagon", "MIT", "Harvard",
    "CDC", "NIH", "CERN", "Area 51", "MI6",
]

FAKE_SCANDALS = [
    "the pollution cover-up", "the election manipulation scheme",
    "the child labor scandal", "the data theft operation",
    "the financial fraud", "the bribery network",
]

FAKE_TIMES = ["6 months", "3 months", "weeks", "days"]
FAKE_DATES = ["next month", "January 2026", "this summer", "next year"]

# ------------------------------------------------------------------ #
#  REAL NEWS PATTERNS                                                  #
# ------------------------------------------------------------------ #

REAL_TEMPLATES = [
    # --- Government / Policy ---
    "{subject} announces new policy on {topic} following {event}",
    "Report: {subject} releases quarterly data on {topic}",
    "{subject} meets with officials to discuss {topic} regulations",
    "Study finds {topic} linked to changes in {subject} behavior",
    "{subject} spokesperson confirms {topic} review underway",
    "New legislation proposed to address {topic} concerns, says {subject}",
    "Officials from {subject} outline steps to improve {topic} oversight",
    "Survey shows public support for {subject} actions on {topic}",
    "{subject} publishes findings on {topic} after months of research",
    "Experts at {subject} warn of challenges ahead for {topic}",
    "{subject} budget proposal includes $X billion for {topic} programs",
    "Panel recommends {subject} take action on {topic} by next quarter",
    "{subject} and international partners cooperate on {topic} initiative",
    "Data from {subject} shows modest improvement in {topic} metrics",
    "Scientists from {subject} publish peer-reviewed {topic} study",
    "{subject} confirms timeline for {topic} implementation next year",
    "Lawmakers debate {subject} proposal to expand {topic} access",
    "{subject} spokesperson denies allegations, citing {topic} records",
    "New report from {subject} highlights gaps in {topic} funding",
    "Analysis: How {subject}'s approach to {topic} compares globally",
    "{subject} holds press conference addressing {topic} questions",
    "Committee formed to review {subject} policies on {topic}",
    "Economists at {subject} project steady growth in {topic} sector",
    "{subject} releases annual transparency report on {topic}",
    "Officials confirm {subject} cooperation on {topic} investigation",
    "{subject} to host international summit on {topic} next week",
    "Supreme Court delivers ruling on {subject} handling of {topic}",
    "Local governments partner with {subject} to combat {topic} issues",
    "{subject} reports record numbers regarding {topic} in Q3",

    # --- Business / Markets ---
    "{real_company} shares rise {real_pct} after strong quarterly earnings report",
    "{real_company} announces expansion into {real_market} with new product line",
    "{real_company} CEO discusses future plans at annual shareholders meeting",
    "Global markets react to {subject} decision on {topic} policy changes",
    "{real_company} reports revenue of {real_revenue} for fiscal year",
    "Investors weigh {real_company} performance amid changing {topic} landscape",
    "{real_company} launches new division focused on {topic} innovation",
    "Analysts upgrade {real_company} stock following strong {topic} outlook",
    "Trade talks between {subject} and partners focus on {topic} tariffs",
    "{real_company} and {real_company2} announce strategic partnership on {topic}",

    # --- Science / Technology ---
    "Researchers discover new approach to {topic} using advanced technology",
    "New satellite data provides insights into global {topic} patterns",
    "{subject} scientists complete decade-long study on {topic} effects",
    "Major breakthrough in {topic} treatment announced at medical conference",
    "Engineers develop more efficient {topic} system at {subject}",
    "Study in Nature journal finds correlation between {topic} and health outcomes",
    "International team maps {topic} genome for first time",
    "AI-powered tool improves {topic} prediction accuracy by 40 percent",

    # --- Sports ---
    "{real_team} defeats {real_team2} in closely contested championship match",
    "{real_athlete} breaks world record at international {real_sport} event",
    "World Cup qualifiers: {real_country} advances with victory over rivals",
    "Olympics committee announces {real_city} as host for upcoming games",
    "{real_athlete} announces retirement after {real_years} year career",
    "{real_team} signs {real_athlete} in record-breaking transfer deal",

    # --- International / Diplomacy ---
    "{subject} signs bilateral agreement with partners on {topic} cooperation",
    "UN General Assembly adopts resolution on global {topic} standards",
    "Diplomatic talks in {real_city} yield progress on {topic} dispute",
    "G20 leaders pledge increased funding for {topic} development",
    "{subject} imposes sanctions in response to {topic} violations",
    "Peace negotiations continue as {subject} mediates {topic} conflict",

    # --- Weather / Environment ---
    "Hurricane season forecast predicts above-average activity this year",
    "Record temperatures reported across {real_region} during summer months",
    "{subject} releases new climate report showing {topic} trends",
    "Flood warnings issued for {real_region} as heavy rains continue",
    "Wildfire containment reaches 80 percent in {real_region} after week-long effort",
]

REAL_SUBJECTS = [
    "the White House", "the Senate", "the Department of Health",
    "NASA", "the Federal Reserve", "researchers at MIT",
    "the United Nations", "the Supreme Court", "the Treasury Department",
    "the World Health Organization", "the European Union",
    "the Department of Justice", "federal officials", "economists",
    "public health officials", "congressional leaders", "the Pentagon",
    "the UN Security Council", "global stock markets", "the FDA",
    "Stanford researchers", "Oxford University", "the IMF",
    "the Indian government", "the Chinese government", "the Reserve Bank of India",
    "Japan's parliament", "the Bank of England", "ISRO scientists",
    "the Department of Education", "the Environmental Protection Agency",
]

REAL_TOPICS = [
    "healthcare reform", "climate policy", "economic growth",
    "infrastructure spending", "cybersecurity", "trade agreements",
    "immigration", "education funding", "public health",
    "renewable energy", "defense spending", "tax reform",
    "social security", "election security", "housing policy",
    "artificial intelligence", "space exploration", "inflation rates",
    "public transit", "global warming", "national debt",
    "semiconductor production", "electric vehicles", "data privacy",
    "food security", "water resources", "nuclear energy",
    "quantum computing", "biodiversity", "ocean conservation",
]

OBJECTS = [
    "the economy", "public health", "free speech", "citizens",
    "the food supply", "the water supply", "elections", "children",
    "natural resources", "the internet", "the financial system",
    "the middle class", "DNA", "vaccines", "microchips", "the weather",
    "cryptocurrency", "social media", "education", "democracy",
]

ACTIONS = [
    "destroy", "control", "manipulate", "poison", "censor",
    "monitor", "eliminate", "exploit", "reprogram", "track",
    "erase", "brainwash", "bankrupt", "weaponize", "suppress",
]

EVENTS = [
    "last month's summit", "the recent vote", "mounting pressure",
    "the annual review", "stakeholder meetings", "public feedback",
    "the latest audit", "bipartisan discussions", "global conferences",
    "the fiscal year end", "quarterly earnings reports",
    "the latest quarterly review", "expert consultations",
    "international negotiations", "budget deliberations",
]

REAL_COMPANIES = [
    "Apple", "Google", "Microsoft", "Amazon", "Tesla",
    "Meta", "Nvidia", "Samsung", "Toyota", "JPMorgan",
    "Goldman Sachs", "Walmart", "Disney", "Netflix", "Intel",
    "IBM", "Cisco", "Adobe", "Salesforce", "Berkshire Hathaway",
]

REAL_COMPANIES2 = REAL_COMPANIES.copy()

REAL_MARKETS = [
    "Southeast Asia", "European", "Latin American", "African",
    "Middle Eastern", "Indian", "Australian", "Japanese",
]

REAL_PCTS = ["3%", "5%", "7%", "2.5%", "4.2%", "8%", "1.8%", "6.3%"]
REAL_REVENUES = [
    "$85 billion", "$120 billion", "$45 billion", "$200 billion",
    "$33 billion", "$67 billion", "$150 billion", "$28 billion",
]

REAL_TEAMS = [
    "Manchester City", "Real Madrid", "Barcelona", "Bayern Munich",
    "Liverpool", "Mumbai Indians", "Chennai Super Kings",
    "Los Angeles Lakers", "New York Yankees", "Golden State Warriors",
]
REAL_TEAMS2 = REAL_TEAMS.copy()

REAL_ATHLETES = [
    "Virat Kohli", "Lionel Messi", "Cristiano Ronaldo", "LeBron James",
    "Usain Bolt", "Novak Djokovic", "Lewis Hamilton", "Neeraj Chopra",
    "Rohit Sharma", "Kylian Mbappe", "Serena Williams",
]

REAL_SPORTS = ["athletics", "swimming", "tennis", "cricket", "football"]
REAL_COUNTRIES = ["India", "Brazil", "Germany", "Japan", "Argentina", "France", "England"]
REAL_CITIES_LIST = ["Paris", "Tokyo", "Los Angeles", "Brisbane", "Milan"]
REAL_YEARS = ["15", "20", "12", "18", "10", "25"]
REAL_REGIONS = [
    "California", "South Asia", "Western Europe", "Australia",
    "Southeast US", "Eastern India", "Mediterranean region",
]

HIDE_VERBS = ["hides", "buries", "suppresses", "ignores", "covers up"]


def generate_fake_headline():
    """Create one fake news headline using templates."""
    template = random.choice(FAKE_TEMPLATES)

    # Select two distinct fake persons
    fp1 = random.choice(FAKE_PERSONS)
    fp2 = random.choice([p for p in FAKE_PERSONS if p != fp1] + FAKE_PERSONS2)
    fc1 = random.choice(FAKE_COMPANIES)
    fc2 = random.choice([c for c in FAKE_COUNTRIES if c != random.choice(FAKE_COUNTRIES)])

    return template.format(
        subject=random.choice(FAKE_SUBJECTS),
        fake_verb=random.choice(FAKE_VERBS),
        object=random.choice(OBJECTS),
        action=random.choice(ACTIONS),
        hide_verb=random.choice(HIDE_VERBS),
        fake_person=fp1,
        fake_person2=fp2,
        fake_company=fc1,
        fake_amount=random.choice(FAKE_AMOUNTS),
        fake_crime=random.choice(FAKE_CRIMES),
        fake_country=random.choice(FAKE_COUNTRIES),
        fake_country2=fc2,
        fake_city=random.choice(FAKE_CITIES),
        fake_disease=random.choice(FAKE_DISEASES),
        fake_disaster=random.choice(FAKE_DISASTERS),
        fake_discovery=random.choice(FAKE_DISCOVERIES),
        fake_org=random.choice(FAKE_ORGS),
        fake_scandal=random.choice(FAKE_SCANDALS),
        fake_time=random.choice(FAKE_TIMES),
        fake_date=random.choice(FAKE_DATES),
    )


def generate_real_headline():
    """Create one real news headline using templates."""
    template = random.choice(REAL_TEMPLATES)

    rc1 = random.choice(REAL_COMPANIES)
    rc2 = random.choice([c for c in REAL_COMPANIES2 if c != rc1])
    rt1 = random.choice(REAL_TEAMS)
    rt2 = random.choice([t for t in REAL_TEAMS2 if t != rt1])

    return template.format(
        subject=random.choice(REAL_SUBJECTS),
        topic=random.choice(REAL_TOPICS),
        event=random.choice(EVENTS),
        real_company=rc1,
        real_company2=rc2,
        real_market=random.choice(REAL_MARKETS),
        real_pct=random.choice(REAL_PCTS),
        real_revenue=random.choice(REAL_REVENUES),
        real_team=rt1,
        real_team2=rt2,
        real_athlete=random.choice(REAL_ATHLETES),
        real_sport=random.choice(REAL_SPORTS),
        real_country=random.choice(REAL_COUNTRIES),
        real_city=random.choice(REAL_CITIES_LIST),
        real_years=random.choice(REAL_YEARS),
        real_region=random.choice(REAL_REGIONS),
    )


def create_dataset(n_samples=10000, output_path="data/news_dataset.csv"):
    """
    Build a balanced CSV dataset with equal fake and real samples.

    Parameters:
        n_samples   : total number of rows (split evenly: 50% fake, 50% real)
        output_path : where to save the resulting CSV file
    """
    half = n_samples // 2

    fake_rows = [
        {"text": generate_fake_headline(), "label": "FAKE"}
        for _ in range(half)
    ]
    real_rows = [
        {"text": generate_real_headline(), "label": "REAL"}
        for _ in range(half)
    ]

    df = pd.DataFrame(fake_rows + real_rows)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)  # shuffle
    df.to_csv(output_path, index=False)
    print(f"Dataset saved to '{output_path}' with {len(df)} rows.")
    print(df["label"].value_counts())
    return df


if __name__ == "__main__":
    create_dataset()
