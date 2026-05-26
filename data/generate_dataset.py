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
]

FAKE_SUBJECTS = [
    "the government", "scientists", "big pharma", "the deep state",
    "mainstream media", "the president", "elites", "globalists",
    "secret societies", "tech giants", "the CDC", "the FBI",
    "world leaders", "billionaires", "the establishment", "shadow groups",
]

FAKE_VERBS = [
    "is hiding", "has been manipulating", "secretly controls",
    "is suppressing", "has been poisoning", "is brainwashing",
    "has been lying about", "is orchestrating", "secretly banned",
    "has been funding", "is deliberately causing", "has been silencing",
]

REAL_TEMPLATES = [
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
]

REAL_SUBJECTS = [
    "the White House", "the Senate", "the Department of Health",
    "NASA", "the Federal Reserve", "researchers at MIT",
    "the United Nations", "the Supreme Court", "the Treasury Department",
    "the World Health Organization", "the European Union",
    "the Department of Justice", "federal officials", "economists",
    "public health officials", "congressional leaders",
]

REAL_TOPICS = [
    "healthcare reform", "climate policy", "economic growth",
    "infrastructure spending", "cybersecurity", "trade agreements",
    "immigration", "education funding", "public health",
    "renewable energy", "defense spending", "tax reform",
    "social security", "election security", "housing policy",
]

OBJECTS = [
    "the economy", "public health", "free speech", "citizens",
    "the food supply", "the water supply", "elections", "children",
    "natural resources", "the internet", "the financial system",
]

ACTIONS = [
    "destroy", "control", "manipulate", "poison", "censor",
    "monitor", "eliminate", "exploit", "reprogram",
]

EVENTS = [
    "last month's summit", "the recent vote", "mounting pressure",
    "the annual review", "stakeholder meetings", "public feedback",
    "the latest audit", "bipartisan discussions",
]


def generate_fake_headline():
    """Create one fake news headline using templates."""
    template = random.choice(FAKE_TEMPLATES)
    return template.format(
        subject=random.choice(FAKE_SUBJECTS),
        fake_verb=random.choice(FAKE_VERBS),
        object=random.choice(OBJECTS),
        action=random.choice(ACTIONS),
        hide_verb=random.choice(["hides", "buries", "suppresses", "ignores"]),
    )


def generate_real_headline():
    """Create one real news headline using templates."""
    template = random.choice(REAL_TEMPLATES)
    return template.format(
        subject=random.choice(REAL_SUBJECTS),
        topic=random.choice(REAL_TOPICS),
        event=random.choice(EVENTS),
    )


def create_dataset(n_samples=2000, output_path="data/news_dataset.csv"):
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
