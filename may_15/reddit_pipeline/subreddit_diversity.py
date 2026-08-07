"""Subreddit diversity dataset + cosine analysis.

Goal: find candidate subreddits that are *very different* from the communities we
already extracted, so a new pull diversifies the dataset instead of piling onto
the same advice/Q&A/hobby/tech cluster.

Method: embed a one-line topic description of each subreddit (sentence-transformers,
shared space for extracted + candidates), then for every candidate compute its
cosine similarity to the NEAREST extracted subreddit. Low nearest-sim = far from
everything we have = a good diversifying pick.

Output:
  data/subreddit_diversity.json  (records: subreddit, extracted, description,
                                   nearest_extracted, nearest_cos, mean_cos)
  printed ranking of candidates, most-different first.
"""
import json
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

HERE = Path(__file__).resolve().parent
OUT = (HERE / ".." / "data" / "subreddit_diversity.json").resolve()
EMBED_MODEL = "sentence-transformers/all-mpnet-base-v2"

# --- already extracted (extracted=True) : sub -> topic description -------------
EXTRACTED = {
    "AskDocs": "asking doctors about medical symptoms and health concerns",
    "AskEngineers": "engineering questions and professional engineering advice",
    "AskHistorians": "in-depth historical questions answered by historians",
    "askscience": "rigorous science questions answered by experts",
    "AskAcademia": "academic careers, grad school, and university life",
    "AskCulinary": "professional cooking technique questions",
    "askphilosophy": "philosophy questions and concepts",
    "AskStatistics": "statistics methodology and data questions",
    "statistics": "statistics theory and applied statistics discussion",
    "datascience": "data science careers, tools, and methods",
    "MachineLearning": "machine learning research and engineering",
    "learnmachinelearning": "learning machine learning as a beginner",
    "LanguageTechnology": "natural language processing and computational linguistics",
    "programming": "general software programming discussion",
    "learnprogramming": "learning to code as a beginner",
    "learnpython": "learning the Python programming language",
    "learnjavascript": "learning the JavaScript programming language",
    "rust": "the Rust systems programming language",
    "golang": "the Go programming language",
    "Cooking": "home cooking recipes and technique",
    "AskBaking": "home baking questions and troubleshooting",
    "Coffee": "brewing coffee, beans, and gear",
    "tea": "tea varieties, brewing, and culture",
    "homeimprovement": "home repair and renovation projects",
    "DIY": "do-it-yourself home and craft projects",
    "gardening": "growing plants and outdoor gardens",
    "houseplants": "caring for indoor houseplants",
    "Sewing": "sewing garments and fabric projects",
    "woodworking": "woodworking craft and furniture building",
    "bicycling": "road and recreational cycling",
    "personalfinance": "personal budgeting, saving, and money management",
    "povertyfinance": "managing money on a low income",
    "legaladvice": "everyday legal questions and advice",
    "relationships": "interpersonal relationship problems and advice",
    "relationship_advice": "advice on romantic and personal relationships",
    "GradSchool": "graduate school experience and advice",
    "languagelearning": "learning foreign languages",
    "LearnJapanese": "learning the Japanese language",
    "German": "learning the German language",
    "JapanTravel": "travel planning for Japan",
    "solotravel": "traveling alone and backpacking",
    "Shoestring": "budget travel on very little money",
}

# --- candidate pool (extracted=False) : deliberately spread across far domains --
CANDIDATES = {
    # sports
    "nba": "professional basketball news and discussion",
    "soccer": "international football/soccer news and matches",
    "formula1": "Formula 1 motorsport racing",
    "nfl": "American football league discussion",
    # gaming
    "gaming": "video game culture and news",
    "leagueoflegends": "the League of Legends video game",
    "Minecraft": "the Minecraft video game",
    # entertainment / media
    "movies": "film news, reviews, and discussion",
    "television": "TV shows and series discussion",
    "anime": "Japanese animation shows and discussion",
    "StarWars": "the Star Wars franchise",
    # music
    "Music": "general music discussion and news",
    "hiphopheads": "hip-hop music news and releases",
    # news / politics
    "worldnews": "international news and current events",
    "politics": "US political news and debate",
    # finance speculation / crypto
    "CryptoCurrency": "cryptocurrency markets and blockchain",
    "wallstreetbets": "high-risk stock and options trading memes",
    "stocks": "stock market investing discussion",
    # vehicles
    "cars": "automobiles, car culture and news",
    "motorcycles": "motorcycles, riding and gear",
    # animals
    "aww": "cute animal pictures and videos",
    "dogs": "dog ownership and care",
    # fashion / beauty
    "malefashionadvice": "men's fashion and style advice",
    "MakeupAddiction": "makeup looks and product discussion",
    # nature / outdoors
    "hiking": "hiking trails and backpacking outdoors",
    "space": "astronomy, spaceflight and the cosmos",
    # humor / stories
    "funny": "humorous images and jokes",
    "nosleep": "original horror fiction short stories",
    "WritingPrompts": "creative writing prompts and short fiction",
    # lifestyle / misc far
    "Parenting": "raising children and parenting advice",
    "conspiracy": "conspiracy theories and alternative narratives",
    "todayilearned": "interesting trivia facts people just learned",
}

# Q&A / advice communities in domains we DON'T have -- these keep the
# ask-a-question, thank-a-reply signal so they extract well, while still
# diversifying the topic mix.
CANDIDATES_QA = {
    "MechanicAdvice": "car repair and automotive troubleshooting advice",
    "AskVet": "veterinary and pet health questions",
    "DogTraining": "dog behavior and training advice",
    "SkincareAddiction": "skincare routines and skin problem advice",
    "AskElectronics": "electronics circuits and components help",
    "photography": "camera gear and photography technique advice",
    "Fitness": "exercise, training programs and fitness advice",
    "nutrition": "diet, food science and nutrition questions",
    "investing": "long-term investing strategy questions",
    "Bonsai": "growing and caring for bonsai trees",
    "Guitar": "learning guitar and gear advice",
    "AskCarSales": "buying and selling cars advice",
    "Plumbing": "home plumbing repair advice",
    "knitting": "knitting patterns and technique help",
    "BeginnerWoodWorking": "beginner woodworking project help",
}


def main():
    model = SentenceTransformer(EMBED_MODEL)
    ALL_CAND = {**{n: (d, "discussion") for n, d in CANDIDATES.items()},
                **{n: (d, "qa_advice") for n, d in CANDIDATES_QA.items()}}
    ex_names = list(EXTRACTED)
    ca_names = list(ALL_CAND)
    ex_emb = model.encode([f"{n}: {EXTRACTED[n]}" for n in ex_names],
                          normalize_embeddings=True)
    ca_emb = model.encode([f"{n}: {ALL_CAND[n][0]}" for n in ca_names],
                          normalize_embeddings=True)

    records = []
    # extracted rows (nearest OTHER extracted, for reference)
    sim_ex_ex = ex_emb @ ex_emb.T
    np.fill_diagonal(sim_ex_ex, -1)
    for i, n in enumerate(ex_names):
        j = int(np.argmax(sim_ex_ex[i]))
        records.append({"subreddit": n, "extracted": True,
                        "description": EXTRACTED[n],
                        "nearest_extracted": ex_names[j],
                        "nearest_cos": round(float(sim_ex_ex[i][j]), 4)})

    # candidate rows: similarity to nearest extracted sub
    sim_ca_ex = ca_emb @ ex_emb.T
    cand_rows = []
    for i, n in enumerate(ca_names):
        j = int(np.argmax(sim_ca_ex[i]))
        rec = {"subreddit": n, "extracted": False, "type": ALL_CAND[n][1],
               "description": ALL_CAND[n][0],
               "nearest_extracted": ex_names[j],
               "nearest_cos": round(float(sim_ca_ex[i][j]), 4),
               "mean_cos": round(float(sim_ca_ex[i].mean()), 4)}
        records.append(rec); cand_rows.append(rec)

    OUT.write_text(json.dumps(records, ensure_ascii=False, indent=2))
    print(f"wrote {len(records)} subs -> {OUT}\n")

    for typ, title in [("discussion", "DISCUSSION/NEWS subs (max diversity, low extraction yield)"),
                       ("qa_advice", "Q&A/ADVICE subs in new domains (diversify AND extract well)")]:
        rows = sorted([r for r in cand_rows if r["type"] == typ],
                      key=lambda r: r["nearest_cos"])
        print(f"\n=== {title} -- most-different first ===")
        print(f"{'subreddit':20s} {'nearest_cos':>11} {'mean_cos':>9}  nearest extracted")
        for r in rows:
            print(f"{r['subreddit']:20s} {r['nearest_cos']:11.3f} {r['mean_cos']:9.3f}  "
                  f"~{r['nearest_extracted']}")


if __name__ == "__main__":
    main()
