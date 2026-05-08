#  IPL Batting Phase Win Prediction — Machine Learning

A machine learning project that predicts **IPL match outcomes** based on team batting performance across three phases — **Powerplay, Middle, and Death overs** — using a **Random Forest Classifier**.

---

##  Project Overview

In IPL cricket, a match is split into 3 key batting phases:
- 🟢 **Powerplay** → Overs 1–6
- 🟡 **Middle** → Overs 7–15
- 🔴 **Death** → Overs 16–20

This project analyzes how runs scored in each phase influence the final match result (Win/Loss), and builds a classification model to predict match outcomes.

---

##  Tech Stack

| Tool | Purpose |
|---|---|
| **Python 3.x** | Core language |
| **Pandas** | Data loading and manipulation |
| **Scikit-learn** | Machine learning model |
| **Jupyter Notebook** | Development environment |

---

## Project Structure

```
ipl-batting-phase/
├── batting_phase.ipynb     # Main analysis and model notebook
├── matches.csv             # IPL match-level data
└── deliveries.csv          # Ball-by-ball delivery data
```

---

##  Datasets Used

| File | Description |
|---|---|
| `matches.csv` | Match-level info — teams, winner, venue, season |
| `deliveries.csv` | Ball-by-ball data — runs, wickets, over, batting team |

---

##  Project Pipeline

```
Load Data
    ↓
Merge matches + deliveries
    ↓
Assign Phase to each delivery (Powerplay / Middle / Death)
    ↓
Aggregate runs per team per phase per match
    ↓
Feature Engineering (Total Runs, Phase %, Inning, Wickets Lost)
    ↓
Train Random Forest Classifier
    ↓
Evaluate Model (Accuracy + Classification Report)
```

---

##  Phase Classification Logic

```python
def get_phase(over):
    if over < 6:   return 'Powerplay'
    if over < 15:  return 'Middle'
    else:          return 'Death'
```

---

##  Feature Engineering

### Version 1 — Basic Features
| Feature | Description |
|---|---|
| `Powerplay Runs` | Total runs scored in overs 1–6 |
| `Middle Runs` | Total runs scored in overs 7–15 |
| `Death Runs` | Total runs scored in overs 16–20 |

### Version 2 — Extended Features (Improved Accuracy)
| Feature | Description |
|---|---|
| `Powerplay Runs` | Runs in powerplay overs |
| `Middle Runs` | Runs in middle overs |
| `Death Runs` | Runs in death overs |
| `Total Runs` | Sum of all phase runs |
| `Powerplay %` | % of total runs in powerplay |
| `Middle %` | % of total runs in middle overs |
| `Death %` | % of total runs in death overs |
| `Inning` | Whether batting first or second |
| `Wickets Lost` | Total wickets lost in innings |

---

##  Model

**Algorithm:** Random Forest Classifier (`sklearn.ensemble.RandomForestClassifier`)

| Parameter | Version 1 | Version 2 |
|---|---|---|
| `n_estimators` | 100 | 200 |
| `random_state` | 42 | 42 |
| `test_size` | 20% | 20% |

---

##  Results

### Version 1 — Basic Features
```
Model Accuracy: 60.27%

              precision    recall  f1-score
Loss               0.60      0.59      0.60
Win                0.61      0.61      0.61
accuracy                               0.60
```

### Version 2 — Extended Features
```
Model Accuracy: 70.09%

              precision    recall  f1-score
Loss               0.69      0.72      0.70
Win                0.71      0.68      0.70
accuracy                               0.70
```

 Adding phase percentages, inning info, and wickets lost improved accuracy by **~10%**

---

## Key Insights

-  **Death overs** scoring is the strongest predictor of match outcome
-  **Batting second (Inning 2)** teams have a strategic advantage with chase context
-  **Wickets lost** significantly impacts win probability
-  Balanced win/loss distribution (216 losses vs 222 wins in test set) ensures unbiased evaluation

---

##  Setup & Installation

### Step 1: Install dependencies
```bash
pip install pandas scikit-learn jupyter
```

### Step 2: Download IPL Dataset
Get the dataset from Kaggle:
 https://www.kaggle.com/datasets/nowke9/ipldata

Place `matches.csv` and `deliveries.csv` in the project folder.

### Step 3: Run the notebook
```bash
jupyter notebook batting_phase.ipynb
```

---

##  Future Improvements

- [ ] Add venue and toss decision as features
- [ ] Include player-level batting stats
- [ ] Try XGBoost / LightGBM for better accuracy
- [ ] Build an interactive prediction tool (enter runs → get win probability)
- [ ] Add season-wise trend analysis
- [ ] Visualize feature importances from Random Forest

---

##  Author

**Your Name**
- GitHub: [@sushmasree2004](https://github.com/sushmasree2004)

---

##  License

This project is for educational and analytical purposes using publicly available IPL data.
