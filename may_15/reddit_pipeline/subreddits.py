"""Curated subreddit allowlist + bot blocklist.

The allowlist is intentionally narrow. We want subs where:
  - Posts are predominantly questions or help requests (high question density).
  - Top comments are answers, not jokes or memes.
  - Moderation suppresses the worst noise so the thanks-reply signal isn't
    drowned out by sarcastic "thanks Obama" replies.

Categories are advisory metadata for downstream slicing; the pipeline treats
the union as a flat allowlist. Subreddit names are stored lowercased; comparison
is case-insensitive because Reddit usernames/subs are case-insensitive in
practice though the API preserves casing.
"""

from __future__ import annotations

ALLOWLIST: dict[str, list[str]] = {
    "general_qa": [
        "AskReddit",            # noisy but huge; rely on thanks-reply precision
        "NoStupidQuestions",
        "explainlikeimfive",
        "answers",
        "TooAfraidToAsk",
    ],
    "expert_curated": [
        "AskHistorians",
        "askscience",
        "AskAcademia",
        "AskDocs",
        "AskEngineers",
        "AskCulinary",
        "askphilosophy",
        "AskStatistics",
    ],
    "hobby_practical": [
        "Cooking",
        "AskBaking",
        "homeimprovement",
        "DIY",
        "gardening",
        "houseplants",
        "Sewing",
        "woodworking",
        "bicycling",
        "running",
        "Fitness",
        "bodyweightfitness",
        "Coffee",
        "tea",
    ],
    "travel_local": [
        "travel",
        "solotravel",
        "JapanTravel",
        "Shoestring",
    ],
    "tech_help": [
        "techsupport",
        "buildapc",
        "homelab",
        "selfhosted",
        "learnpython",
        "learnprogramming",
        "learnjavascript",
        "rust",
        "golang",
        "MachineLearning",
        "LanguageTechnology",
        "datascience",
    ],
    "life_advice": [
        "personalfinance",
        "legaladvice",
        "relationships",
        "relationship_advice",
        "careerguidance",
        "jobs",
    ],
    "language_learning": [
        "languagelearning",
        "Spanish",
        "French",
        "German",
        "LearnJapanese",
    ],
}

ALLOWLIST_FLAT: set[str] = {s.lower() for subs in ALLOWLIST.values() for s in subs}


# Known auto-responder / bot accounts. Comments authored by these are dropped
# before any answer-quality scoring. List is not exhaustive — the pipeline also
# applies heuristic bot detection (account name ends in 'Bot', body contains
# 'I am a bot' / 'beep boop', etc.) in signals.py.
BOT_AUTHORS: set[str] = {
    "AutoModerator",
    "RemindMeBot",
    "WikiTextBot",
    "sneakpeekbot",
    "imguralbumbot",
    "TotesMessenger",
    "TweetPoster",
    "TweetsInCommentsBot",
    "GoodBot_BadBot",
    "B0tRank",
    "savevideo",
    "SaveVideo",
    "VredditDownloader",
    "video_descriptbot",
    "transcribot",
    "stabbot",
    "gifv-bot",
    "HelperBot_",
    "LinkifyBot",
    "Mentioned_Videos",
    "youtubefactsbot",
    "FatFingerHelperBot",
    "CommonMisspellingBot",
    "of_patrol_bot",
    "RepostSleuthBot",
    "TitleLinkHelperBot",
    "SmallSubBot",
    "FriendlyKurdishBot",
}


# Sentinels that appear in `author` when the account or content was removed.
DELETED_SENTINELS: set[str] = {"[deleted]", "[removed]", "", None}  # type: ignore[arg-type]


def is_allowed_subreddit(sub: str | None) -> bool:
    return bool(sub) and sub.lower() in ALLOWLIST_FLAT


def is_bot_author(author: str | None) -> bool:
    if not author:
        return False
    if author in BOT_AUTHORS:
        return True
    a = author.lower()
    return a.endswith("bot") or a.endswith("-bot") or a.endswith("_bot")


def is_deleted_author(author: str | None) -> bool:
    return author in DELETED_SENTINELS
