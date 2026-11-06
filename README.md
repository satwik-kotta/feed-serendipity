# Feed Serendipity Engine

Detecting feed narrowing and testing targeted exploration on real short video data

[Live demo](#) · [Experiment readout](reports/results.md) · [Dashboard](#)

## TL;DR
- **Problem.** Engagement optimized feeds narrow over time and users get bored
- **Question.** Does narrowing actually lower engagement, and can targeted exploration fix it without hurting relevance?
- **Data.** KuaiRec, real logs from the Kuaishou short video app, including a 1,411 user by 3,327 video matrix that is 99.6% fully observed
- **Result.** _[fill in. Headline lift in coverage and creator Gini with relevance change and confidence interval]_
- **Recommendation.** _[fill in. Launch, iterate or drop, at which threshold]_

## 1. Why this matters
Short video, music and pin feeds are tuned for the next click. Over weeks that produces a narrow content diet, which is linked to fatigue and churn. Most fixes add random exploration, which wastes slots on irrelevant content. This project tests a targeted alternative. Inject adjacent content only for users whose feed has become measurably narrow.

## 2. Data
| Matrix | Users | Videos | Interactions | Density | Use |
|---|---|---|---|---|---|
| big_matrix | 7,176 | 10,728 | 12.5M | 16.3% | Training, fatigue analysis |
| small_matrix | 1,411 | 3,327 | 4.7M | 99.6% | Ground truth offline A/B |

- Label is `watch_ratio`, play duration divided by video duration
- Each video has 1 to 4 of 31 tags. Author IDs come from daily item features
- Small matrix interactions are excluded from the big matrix, so there is no leakage
- Source. Gao et al., KuaiRec, CIKM 2022

## 3. Approach
_[architecture diagram]_

1. **Narrowness Index.** 1 minus normalized tag entropy over a user's last N videos
2. **Fatigue analysis.** User fixed effects regression of watch ratio on narrowness
3. **Baseline recommender.** ALS candidates reranked by LightGBM
4. **Policies.** Baseline, random exploration, triggered adjacent injection, MMR
5. **Offline A/B.** Every arm scored against fully observed small matrix responses
6. **Simulation.** 10 sessions per user with a fitted fatigue curve
7. **Threshold sweep.** Relevance against diversity frontier

## 4. Key findings
### Narrowing and fatigue
_[fatigue curve figure + 2 sentences]_

### Offline A/B results
| Arm | nDCG@10 | Mean watch ratio | Tag coverage | Discovery success | Creator Gini |
|---|---|---|---|---|---|
| A. Baseline | | | | n/a | |
| B. Random | | | | | |
| C. Triggered injection | | | | | |
| D. MMR | | | | | |

95% bootstrap intervals over users in `reports/results.md`

### Threshold tradeoff
_[frontier figure + recommended operating point]_

### Creator side impact
_[exposure share by author tier, before and after]_

## 5. Recommendation
_[3 to 4 sentences written for a product lead. What to launch, at what threshold, what to monitor]_

**Guardrails to monitor in a live test.** Session length, next day return rate, skip rate on injected slots

## 6. Limitations
- Small matrix exposure was forced, so it carries popularity bias
- Long term results come from a simulation, not live users
- Watch ratio is the only signal. There is no like or follow label

## 7. Run it
```bash
git clone https://github.com/satwik-kotta/feed-serendipity
cd feed-serendipity
pip install -r requirements.txt
python data/download.py
streamlit run app/streamlit_app.py
```

## 8. Repo map
```
notebooks/   EDA, fatigue, baseline, policies, sweep
src/         loading, narrowness, models, policies, metrics, simulation
app/         Streamlit demo
reports/     figures and experiment readout
```

## Tech
Python, pandas, implicit (ALS), LightGBM, statsmodels, Streamlit, Plotly
