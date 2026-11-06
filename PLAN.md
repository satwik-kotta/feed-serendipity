# Build Plan · Feed Serendipity Engine

Repo name `feed-serendipity` under satwik-kotta

## Core idea
Recommenders that only exploit known tastes make a user's feed narrower over time. This project measures that narrowing on real short video data, tests whether it lowers engagement, and evaluates a policy that injects adjacent content only when a user's feed becomes too narrow.

## Why KuaiRec makes this credible
- Real logs from Kuaishou, a short video app
- `big_matrix` has 7,176 users and 10,728 videos at 16.3% density. Used for training and for the fatigue analysis
- `small_matrix` has 1,411 users and 3,327 videos at 99.6% density. Almost every user has seen almost every video, so the true watch ratio of any recommended video is known. This allows an honest offline A/B test with no invented reactions
- Interactions in the small matrix are excluded from the big matrix, so training and evaluation never overlap
- Each video has 1 to 4 tags out of 31. Daily item features include the author, which gives the creator side
- Label is `watch_ratio` = play duration / video duration. Above 1 means watched fully, above 2 means replayed

## Repo structure
```
feed-serendipity/
├── README.md
├── requirements.txt
├── data/                     # gitignored, download script only
│   └── download.py
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_narrowness_and_fatigue.ipynb
│   ├── 03_baseline_recommender.ipynb
│   ├── 04_policies_offline_ab.ipynb
│   └── 05_threshold_sweep.ipynb
├── src/
│   ├── load.py               # read CSVs, parse tag lists, join author_id
│   ├── sessions.py           # split big_matrix into sessions by time gaps
│   ├── narrowness.py         # Narrowness Index per user and window
│   ├── adjacency.py          # tag adjacency from co-occurrence
│   ├── models.py             # ALS candidates + LightGBM reranker
│   ├── policies.py           # baseline, random, triggered injection, MMR
│   ├── metrics.py            # relevance, diversity, coverage, creator Gini
│   └── simulate.py           # multi-session simulation with fatigue
├── app/
│   └── streamlit_app.py      # interactive demo
├── reports/
│   ├── figures/
│   └── results.md            # experiment readout
└── tests/
    └── test_metrics.py
```

## Phases

### Phase 1 · Data and EDA (2 days)
- Download script, load all files, parse tags
- Watch ratio distribution, tag frequency, author concentration
- Confirm small matrix density and the big/small split

### Phase 2 · Narrowness and fatigue analysis (3 days)

**Done. Findings** (numbers in `reports/phase2_results.json`)
- Within a session, a narrower recent feed goes with slightly higher watch ratio (+2.9%, top vs bottom decile) and fewer exits (-1.1 pp on a 6.9% base). Replicated on a second user group and on 10, 20 and 50 video windows
- Long same-tag runs hurt. Watch ratio is 2.6% lower after 4–5 in a row and 10.9% lower after 6–9 (1.3% of views)
- Next day, the narrowest fifth of days predicts 6.1% fewer views (CI -8.9% to -3.3%) and a 1.2 pp lower chance of returning, against equally busy and engaging broad days
- So narrowness pays today and costs tomorrow. That is the case for targeted exploration

Original plan for this phase
- Sessions in big_matrix from timestamp gaps (start with 30 minutes)
- Narrowness Index = 1 minus normalized Shannon entropy of tag shares over the last N videos (N = 20, test 10 and 50)
- Fatigue test. Does watch ratio fall as the Narrowness Index rises, or as the run of same tag videos grows? Use a fixed effects regression with user fixed effects so heavy watchers do not drive the result
- Key output. A fatigue curve and an estimated threshold where engagement starts dropping
- This is the analysis bullet

### Phase 3 · Baseline recommender (3 days)

**Done. Findings** (numbers in `reports/phase3_baselines.json`)
- Raw watch ratio is the wrong target. A video's average watch ratio correlates -0.91 with its length. A LightGBM trained on it recommends 3-second clips, the same 17 for everyone, and earns fewer watch seconds than random (7.8 vs 8.4 per video)
- Main baseline is LightGBM on a duration-debiased target: percentile of how far a play's watch ratio sits above a smooth watch-ratio-by-length curve. Like D2Q (Kuaishou, KDD 2022) but without bins, which left a saw-tooth length bias
- Evaluation measure is fitted on the small matrix, so it is length-neutral there (r = 0.02)
- Baseline relevance 0.561 vs 0.539 for the best global list, 0.503 ALS, 0.500 random, 0.988 oracle. Paired bootstrap intervals exclude zero
- It is the narrowest feed of all the personalised options. 3.9 distinct tags per 10, 8% of the catalog, creator Gini 0.97. That is the before picture for Phase 4

Original plan for this phase
- ALS (implicit library) on big_matrix for candidates
- LightGBM reranker predicting watch ratio from user, item and tag features
- Evaluate on small_matrix with nDCG@10, debiased relevance, watch seconds and raw watch ratio
- Baseline needs to be strong, otherwise beating it means nothing

### Phase 4 · Policies and offline A/B (4 days)

**Done. Findings** (numbers in `reports/phase4_results.json`, all paired over 1,411 users with 95% bootstrap intervals)
- 35% of users have a narrow recent feed and get the discovery slots
- At the same list variety (4.5 tags), MMR keeps relevance flat (+0.4%, n.s.) and triggered injection costs 0.3% (n.s.). On list variety alone, MMR wins
- But MMR mostly varies within tags the user already watches (+0.3 new-territory slots per page for narrow users). Triggered injection adds +1.6
- Recommended policy F = MMR + triggered discovery + streak breaker: +30% tags per list, new territory for narrow users 1.96 -> 3.85 slots, relevance -0.2% (n.s.), watch time unchanged
- Streak breaker alone is free: relevance +0.4% (significant), +5% tags
- Tag variety barely moves creator concentration (Gini 0.97). Only random exploration does, at a 1.6% relevance cost

Original plan for this phase
Four arms, all scored on small_matrix ground truth
- A. Baseline. Top 10 from the reranker
- B. Random exploration. Replace 2 of 10 slots with random items
- C. Triggered adjacent injection (the proposal), with two triggers from Phase 2
  - Streak breaker. If the last 4 videos share a main tag, the next slot comes from an adjacent tag
  - Narrowness trigger. If rolling narrowness of the last 20 videos is above 0.378 (60th percentile), replace 1 to 2 of 10 slots with the highest predicted items from adjacent tags the user rarely watches
- D. MMR diversity rerank. Standard academic benchmark, tuned to the same diversity as C
- F. MMR + C's triggered discovery slots + streak breaker (added after the first results)

Adjacent tags come from tag co-occurrence in the big matrix (Jaccard or PMI), so "adjacent" is learned from data, not hand picked

Metrics per arm
- Relevance. nDCG@10, mean watch ratio, replay rate (watch ratio above 2)
- Diversity. Intra list diversity, tag coverage, novelty
- Discovery success rate. Share of injected videos watched fully
- Creator side. Gini coefficient of exposure across authors, share of exposure going to mid tier authors
- Bootstrap confidence intervals over users for every difference
- This is the modelling bullet

### Phase 5 · Long term simulation and threshold sweep (3 days)

**Done. Findings** (numbers in `reports/phase5_results.json`, sweep in `reports/phase5_sweep.csv`)
- 68 settings swept: share of users triggered (0–100%), discovery slots (1–4), MMR strength (0–4x)
- Stronger MMR is nearly free. Discovery slots are where the cost is. Targeting the narrowest 20–35% keeps most of the benefit for a fraction of the cost
- Chosen setting (max new territory with relevance within 0.5% of baseline): MMR 4x, 35% triggered, 3 slots. Tags per page 3.9 -> 6.4 (+65%), new-territory slots +78%, relevance -0.4% (n.s.), watch time -0.5% (n.s.)
- Rough next-day value via the Phase 2 slope: +0.9% next-day views overall [+0.5, +1.2], +2.2% for triggered users [+1.4, +3.1]. Observational, so this is a case for a live test, not a forecast
- Replaced the planned multi-session simulation with this direct valuation, because the simulation would have needed an invented fatigue model, and Phase 2 gave a measured slope instead

Original plan for this phase
- Simulate 10 consecutive sessions per user. Responses come from small_matrix truth, scaled by the fatigue curve fitted in Phase 2
- Compare cumulative watch time and narrowness over sessions for each arm
- Sweep the trigger threshold and plot the frontier of relevance against diversity
- Clearly labeled as a simulation in the README

### Phase 6 · Communicate (3 days)
- Streamlit app. Pick a user, see their narrowness over time, toggle threshold and injection slots, see the feed and metrics change, see creator Gini move
- Optional Tableau Public dashboard of arm results
- `reports/results.md` written as an experiment readout with a launch recommendation
- Deploy Streamlit on Streamlit Community Cloud using a precomputed results file, since the raw data is large
- This is the communicating bullet

## Honest limits to state in the README
- Small matrix exposure was forced, so it carries some popularity bias
- Long term effects come from a simulation built on a fitted fatigue curve, not from live users
- No like or follow signal exists. Watch ratio is the only label

## Resume bullet frames (fill numbers after results)
- Analysis. Quantified feed narrowing across 7K users and 12.5M interactions, finding watch ratio falls X% once the Narrowness Index passes Y
- Modelling. Built an ALS + LightGBM recommender and a triggered exploration policy, evaluated in a fully observed offline A/B test that lifted tag coverage X% with a Y% change in nDCG and Z% lower creator exposure Gini
- Communicating. Shipped a Streamlit app and experiment readout letting stakeholders tune the exploration threshold and see the engagement and creator tradeoff live

## Interview story
Optimizing only for the next click makes feeds predictable. I wanted to test whether that predictability actually costs engagement on real data, and whether a small, targeted dose of exploration can recover it without hurting relevance or starving smaller creators.
