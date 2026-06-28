# student-query-log-analysis

> Computational analysis of 3.38 million Google Search queries from Singapore secondary school students, characterising information-seeking behaviour along Specific-Broad (S-B) and Surface-Deep (S-D) dimensions using NLP, semantic embeddings, and clustering.

Published in the **Proceedings of URECA@NTU 2025–26** under the NTU Undergraduate Research Experience on Campus (URECA) programme.

**Tech stack:** Python · pandas · spaCy · Sentence-BERT (all-MiniLM-L6-v2) · NLTK · scikit-learn (K-means, LDA) · Brysbaert Concreteness Norms · Levenshtein distance · Jupyter Notebooks

---

## Key outcomes

- 3.38 million queries retained from 4,961 active students across 47 Singapore secondary schools after preprocessing
- Replicable pipeline for extracting, normalising, and segmenting Google Search URI logs into behavioural episodes
- 9 S-B features (lexical, semantic, entity-based, psycholinguistic, topical) and 11 S-D features (intent signals, reformulation, engagement) engineered per episode
- Four distinct user-level specificity profiles identified via K-means clustering
- Pre-generative-AI search behaviour baseline established for future post-GenAI comparison

---

## Table of Contents

1. [Research overview](#research-overview)
2. [Dimensions](#dimensions)
3. [Methodology](#methodology)
4. [Key findings](#key-findings)
5. [Repository structure](#repository-structure)
6. [Prerequisites](#prerequisites)
7. [Setup](#setup)

---

## Research overview

With the nationwide rollout of Personal Learning Devices (PLDs) in Singapore secondary schools, web search has become central to everyday learning. Yet, little is known about how students actually formulate information needs or how those needs evolve across a search session.

This study addresses that gap by analysing large-scale Google Search query logs from PLDs across 47 schools. Rather than treating search volume as a marker of sophistication, the analysis characterises search behaviour through *how* students search — not just how much.

The research is structured around two complementary dimensions grounded in the search-as-learning literature:

| Dimension | What it captures |
|---|---|
| **Specific-Broad (S-B)** | Scope and granularity of information needs — from narrowly targeted queries to broad, exploratory searches |
| **Surface-Deep (S-D)** | Cognitive depth of the search process — from simple fact retrieval to iterative refinement and comparative reasoning |

Together these dimensions form an interpretable, scalable framework for profiling student information-seeking behaviour that can support educational interventions, learning analytics dashboards, and adaptive system design.

---

## Dimensions

### Specific-Broad (S-B) — 9 episode-level features

| Feature | Method | Interpretation |
|---|---|---|
| Query Length | Mean token count per episode | Surface complexity of information needs |
| Lexical Diversity | Shannon entropy over word types | Vocabulary variability within an episode |
| Semantic Diversity | Mean/variance of pairwise Sentence-BERT distances | Conceptual spread across queries |
| Entity-Type Diversity | Shannon entropy over spaCy NER labels | Diversity of entity categories (e.g., PERSON, ORG, GPE) |
| Topic Diversity | Entropy over topic distribution (99-topic LDA model) | Breadth of latent semantic topics |
| Topic Count | Number of unique topics per episode | Discrete topical coverage |
| Hypernym Specificity | WordNet hypernym depth (NLTK) | Degree of lexical specificity |
| Concreteness | Brysbaert et al. (2013) concreteness norms | Abstract vs. concrete vocabulary use |
| Category Noun Signals | Lexicon + embedding-based detection (e.g., "type", "kind") | Degree of abstraction framing |

### Surface-Deep (S-D) — 11 episode-level features

| Feature | Method | Interpretation |
|---|---|---|
| Open, Non-Possibility (O_NP) | Regex: WH-starter + non-modal | Explicit question-asking intent |
| Possibility Questions (O_P) | Regex: WH-starter + modal verbs | Hypothetical or conditional inquiry |
| Closed Questions (C_NP) | Regex: must/should/do/is/are/does/did | Closed, factual-answer-seeking intent |
| Closure-Seeking | Lexicon + embedding expansion (e.g., definition, meaning) | Intent to obtain a definitive answer |
| Comparative | Lexicon + embedding expansion (e.g., vs, pros, compare) | Intent to compare and evaluate |
| Category Nouns | Lexicon + embedding expansion (e.g., type, kind, group) | Taxonomic or categorical framing |
| Reformulation Strength | Mean normalised Levenshtein distance between consecutive queries | Degree of query rewriting |
| Reformulation Share | Proportion of query-pairs with non-zero Levenshtein distance | Prevalence of reformulation |
| Queries per Episode | Number of queries per session | Episode depth / search intensity |
| Time Interval | Mean time gap between consecutive queries (seconds) | Within-episode pacing |
| Total Episode Time | Sum of inter-query intervals | Overall session duration |

---

## Methodology

### 1. Data preprocessing

- Raw PLD browsing logs from 47 schools consolidated into a single master log
- Search queries extracted from Google Search URLs via the `q=` parameter using a regex-based parser
- Queries normalised to lowercase; duplicate entries from SafeSearch URL variants removed
- Navigational queries (platform logins, entertainment sites, etc.) filtered out using a two-layer fuzzy + substring matching framework
- Remaining queries segmented into *search episodes* using a **5-minute inactivity threshold**

### 2. Dataset scaling

The full corpus contained 40,586 users. To support reliable feature estimation while maintaining computational feasibility for NER and semantic embedding:
- Only users with **≥ 400 query episodes** were retained (above the 75th percentile of episode counts)
- Final dataset: **4,961 active users · 3,382,011 queries**

### 3. Feature engineering

S-B features used spaCy (`en_core_web_md`) for NER, Sentence-BERT (`all-MiniLM-L6-v2`) for semantic embeddings (cached per unique query), a pretrained 99-topic LDA model for topical breadth, NLTK WordNet for hypernym depth, and Brysbaert concreteness norms for psycholinguistic scoring.

S-D intent features used regex rules for question-type detection and a lexicon-embedding hybrid approach for closure-seeking, comparative, and category-noun detection — with seed lexicons expanded via Sentence-BERT at cosine similarity > 0.58, followed by manual validation.

### 4. User profiling

K-means clustering on user-level S-B features (mean hypernym distance, within-user variability, episode specificity ratio, mean query length) yielded four interpretable behavioural profiles ranging from consistently narrow searchers to highly variable, broad explorers.

---

## Key findings

- **Most queries are short keyword fragments** — user mean query lengths centred around 3.5–4 tokens; natural-language questions are rare
- **Search episodes are predominantly single-topic** — episode topic diversity is strongly bimodal with most sessions near zero entropy
- **Closure-seeking (6.4%) and comparative queries (3%) are uncommon** — higher-order cognitive behaviours are not expressed spontaneously without instructional scaffolding
- **Reformulation is pervasive and increases with session depth** — consecutive query pairs show normalised Levenshtein distances of 0.7–0.9, with deeper episodes converging toward near-complete rewriting
- **Possibility questions appear later in episodes** — users occasionally shift from broad keyword exploration toward hypothetical framing as sessions progress
- **Four distinct specificity profiles** identified: consistently narrow, variably specific, moderately specific, and broadly variable searchers

---

## Repository structure

```
ureca-google-search-analysis/
├── Data Cleaning/              # Preprocessing pipeline: query extraction, normalisation,
│                               # navigational filtering, episode segmentation
├── Feature Engineering/        # S-B and S-D feature computation notebooks
├── Generative AI Chatlog/      # Logged AI-assisted development sessions
├── Submissions/                # Final paper submission files
└── README.md
```

---

## Prerequisites

- **Python 3.9+**
- **Jupyter Notebook** or JupyterLab
- spaCy with `en_core_web_md` model
- Sentence-Transformers (`all-MiniLM-L6-v2`)
- NLTK with WordNet corpus
- scikit-learn, pandas, numpy

---

## Setup

```bash
git clone https://github.com/tdfffffffff/student-query-log-analysis.git
cd student-query-log-analysis

pip install pandas numpy scikit-learn nltk spacy sentence-transformers python-Levenshtein
python -m spacy download en_core_web_md
```

Then open the notebooks in order — start with `Data Cleaning/` before running `Feature Engineering/`.
