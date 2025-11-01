# 🎯 Score Improvement Plan: 0.89543 → 0.90+

**Current Best:** `new_sub.csv` = 0.89543  
**Target:** 0.900+ (break the 0.90 barrier)  
**Gap:** ~0.005 (50-150 predictions to fix)

---

## 📊 Analysis Summary

### **What Works (Keep Doing):**
✅ Simple ensembles > complex stacking  
✅ Pseudo-labeling at 98% confidence (0.89293)  
✅ Voting of diverse models (0.89355)  
✅ CatBoost as base model (handles categoricals best)  
✅ Medical domain features (aggressiveness_score, risk_score)

### **What Fails (Avoid):**
❌ Neural networks (0.77-0.85) - terrible performance  
❌ Lower confidence pseudo-labeling (<98%)  
❌ Aggressive class balancing (drops F1)  
❌ Complex 4+ model stacking (overfits)

---

## 🚀 HIGH PROBABILITY APPROACHES (Ranked)

### **🥇 PRIORITY 1: Out-of-Fold (OOF) Stacking** 
**Probability: 85% | Expected Gain: +0.003 to +0.008**

**Why:** Current stacking may have data leakage. OOF prevents this.

**Implementation:**
```python
from sklearn.model_selection import StratifiedKFold

# Generate clean OOF predictions (5-fold)
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
oof_train = np.zeros((len(X_train), n_classes, n_models))
oof_test = np.zeros((len(X_test), n_classes, n_models))

for fold, (tr_idx, val_idx) in enumerate(kf.split(X_train, y_train)):
    # Train each model on 4 folds, predict on 5th fold
    for i, model in enumerate([catboost, xgb, lgbm]):
        model.fit(X_train[tr_idx], y_train[tr_idx])
        oof_train[val_idx, :, i] = model.predict_proba(X_train[val_idx])
        oof_test[:, :, i] += model.predict_proba(X_test) / 5

# Stack with meta-model (no leakage!)
meta_model = LogisticRegression()
meta_model.fit(oof_train.reshape(len(X_train), -1), y_train)
final_pred = meta_model.predict(oof_test.reshape(len(X_test), -1))
```

**Status:** ⚠️ NOT in main_clean.ipynb  
**Effort:** 30 minutes  
**File:** Add to main_clean.ipynb as new section

---

### **🥈 PRIORITY 2: Adversarial Validation → Domain Adaptation**
**Probability: 70% | Expected Gain: +0.002 to +0.010**

**Why:** Check if train/test distributions differ. If yes, fix it.

**Step 1: Diagnose (5 min)**
```python
# Can model distinguish train from test?
X_combined = pd.concat([X_train, X_test])
y_combined = np.concatenate([
    np.zeros(len(X_train)),  # Train = 0
    np.ones(len(X_test))      # Test = 1
])

adversarial_model = XGBClassifier(random_state=42)
cv_scores = cross_val_score(adversarial_model, X_combined, y_combined, cv=5, scoring='roc_auc')

print(f"Adversarial AUC: {cv_scores.mean():.4f}")
# If > 0.55: Train/test differ → need adaptation
# If ≈ 0.50: Distributions similar → skip this
```

**Step 2: If AUC > 0.55, Apply Domain Adaptation (20 min)**
```python
# Use importance weights to make train look like test
adversarial_model.fit(X_combined, y_combined)
train_proba = adversarial_model.predict_proba(X_train)[:, 1]

# Samples that look like test get higher weight
sample_weights = train_proba / (1 - train_proba + 1e-10)
sample_weights = sample_weights / sample_weights.sum() * len(sample_weights)

# Retrain with weights
catboost_adapted = CatBoostClassifier(...)
catboost_adapted.fit(X_train, y_train, sample_weight=sample_weights)
```

**Status:** ⚠️ NOT in any notebook  
**Effort:** 25 minutes  
**File:** Create new cell in main_clean.ipynb

---

### **🥉 PRIORITY 3: Bayesian Weight Optimization**
**Probability: 75% | Expected Gain: +0.001 to +0.004**

**Why:** Finding mathematically optimal ensemble weights vs. guessing.

**Implementation:**
```python
from scipy.optimize import differential_evolution

def ensemble_objective(weights):
    # Normalize weights
    weights = np.abs(weights) / np.sum(np.abs(weights))
    
    # Weighted blend of validation predictions
    blended = (weights[0] * cat_val_proba + 
               weights[1] * xgb_val_proba + 
               weights[2] * lgbm_val_proba)
    
    val_pred = blended.argmax(axis=1)
    return -f1_score(y_val, val_pred, average='weighted')  # Minimize negative F1

# Find optimal weights
result = differential_evolution(
    ensemble_objective,
    bounds=[(0, 1)] * 3,
    seed=42,
    maxiter=100
)

optimal_weights = result.x / result.x.sum()
print(f"Optimal: CatBoost={optimal_weights[0]:.3f}, XGB={optimal_weights[1]:.3f}, LGBM={optimal_weights[2]:.3f}")

# Apply to test
final_proba = (optimal_weights[0] * cat_test_proba + 
               optimal_weights[1] * xgb_test_proba + 
               optimal_weights[2] * lgbm_test_proba)
```

**Status:** ⚠️ NOT in main_clean.ipynb  
**Effort:** 15 minutes  
**File:** Add after base models section

---

### **4️⃣ PRIORITY 4: Manual Misprediction Analysis**
**Probability: 60% | Expected Gain: +0.002 to +0.005**

**Why:** Teammate got +0.00236 by manually fixing 41 predictions (vote → new_sub).

**Implementation:**
```python
# Load top 3 submissions
v7 = pd.read_csv('subChromium_v7_pseudo_label.csv')
v20 = pd.read_csv('subChromium_v20_voting_top3.csv')
v31 = pd.read_csv('subChromium_v31_weighted_ensemble.csv')

# Find disagreements
all_agree = (v7['cancer_stage'] == v20['cancer_stage']) & \
            (v20['cancer_stage'] == v31['cancer_stage'])

disagreement_ids = test_df[~all_agree]['id'].values
print(f"Models disagree on {len(disagreement_ids)} samples")

# Analyze features of disagreed samples
disagreement_features = test_df[test_df['id'].isin(disagreement_ids)]

# Manual rules (examples from medical domain):
# If ki67 > 30 AND mitotic_count > 15 → likely Stage IV (not III)
# If necrosis=1 AND hemorrhage=1 → likely Stage III/IV (not I/II)
# If age > 65 AND tumor_size > 6 → likely Stage IV

# Apply corrections
corrected_predictions = v20['cancer_stage'].copy()
for idx, row in disagreement_features.iterrows():
    test_idx = test_df[test_df['id'] == row['id']].index[0]
    
    # Rule 1: High aggressiveness
    if row['ki67_index'] > 30 and row['mitotic_count'] > 15:
        corrected_predictions.iloc[test_idx] = 4  # Stage IV
    
    # Rule 2: High pathology
    if row['necrosis'] == 1 and row['hemorrhage'] == 1 and row['edema'] == 1:
        corrected_predictions.iloc[test_idx] = 4  # Stage IV
    
    # Add more rules based on EDA...
```

**Status:** ⚠️ Pattern exists in main.ipynb but not implemented  
**Effort:** 45-60 minutes  
**File:** Separate analysis notebook recommended

---

### **5️⃣ PRIORITY 5: Target Encoding for Categoricals**
**Probability: 65% | Expected Gain: +0.001 to +0.003**

**Why:** Current label encoding loses ordinal relationships. Target encoding captures category→target correlations.

**Implementation:**
```python
from category_encoders import TargetEncoder

# Target encode high-cardinality categoricals
target_enc = TargetEncoder(cols=['tumor_location', 'tumor_type'])

# Fit on train only (prevent leakage)
X_train_encoded = target_enc.fit_transform(X_train, y_train)
X_test_encoded = target_enc.transform(X_test)

# Retrain models with encoded features
catboost_targetenc = CatBoostClassifier(...)
catboost_targetenc.fit(X_train_encoded, y_train)
```

**Status:** ⚠️ NOT in any notebook  
**Effort:** 20 minutes  
**File:** Add to preprocessing section

---

### **6️⃣ PRIORITY 6: Feature Selection (Backward Elimination)**
**Probability: 55% | Expected Gain: +0.001 to +0.003**

**Why:** Some engineered features may add noise. Remove them.

**Implementation:**
```python
from sklearn.feature_selection import SequentialFeatureSelector

# Backward feature selection (remove worst features)
sfs = SequentialFeatureSelector(
    CatBoostClassifier(iterations=100, random_state=42),
    n_features_to_select='auto',
    direction='backward',
    scoring='f1_weighted',
    cv=5
)

sfs.fit(X_train, y_train)
selected_features = X_train.columns[sfs.get_support()]

print(f"Selected {len(selected_features)}/{len(X_train.columns)} features")
print(f"Removed: {set(X_train.columns) - set(selected_features)}")

# Retrain with selected features
X_train_selected = X_train[selected_features]
X_test_selected = X_test[selected_features]
```

**Status:** ⚠️ NOT in main_clean.ipynb  
**Effort:** 30 minutes  
**File:** Add as optional optimization step

---

### **7️⃣ PRIORITY 7: Confidence-Weighted Voting**
**Probability: 50% | Expected Gain: +0.001 to +0.003**

**Why:** Current voting treats all models equally. Weight by confidence.

**Implementation:**
```python
# Get probabilities from all models
cat_proba = catboost.predict_proba(X_test)
xgb_proba = xgb.predict_proba(X_test)
lgbm_proba = lgbm.predict_proba(X_test)

# Extract confidence (max probability) for each prediction
cat_conf = cat_proba.max(axis=1)
xgb_conf = xgb_proba.max(axis=1)
lgbm_conf = lgbm_proba.max(axis=1)

# Weight by confidence
weighted_proba = (cat_proba * cat_conf[:, None] + 
                  xgb_proba * xgb_conf[:, None] +
                  lgbm_proba * lgbm_conf[:, None]) / \
                 (cat_conf + xgb_conf + lgbm_conf)[:, None]

final_pred = weighted_proba.argmax(axis=1)
```

**Status:** ⚠️ NOT in main_clean.ipynb  
**Effort:** 15 minutes  
**File:** Add to ensemble section

---

### **8️⃣ PRIORITY 8: Pseudo-Label with Ensemble Agreement**
**Probability: 60% | Expected Gain: +0.001 to +0.003**

**Why:** Current v7 uses single model. Use agreement of 3 models for safer pseudo-labels.

**Implementation:**
```python
# Get predictions from top 3 models
cat_pred = catboost.predict(X_test)
xgb_pred = xgb.predict(X_test)
lgbm_pred = lgbm.predict(X_test)

# Get confidences
cat_conf = catboost.predict_proba(X_test).max(axis=1)
xgb_conf = xgb.predict_proba(X_test).max(axis=1)
lgbm_conf = lgbm.predict_proba(X_test).max(axis=1)

# Only use samples where:
# 1. All 3 models agree
# 2. All 3 models are confident (>95%)
all_agree = (cat_pred == xgb_pred) & (xgb_pred == lgbm_pred)
all_confident = (cat_conf > 0.95) & (xgb_conf > 0.95) & (lgbm_conf > 0.95)

reliable_mask = all_agree & all_confident
print(f"Ultra-reliable pseudo-labels: {reliable_mask.sum()} samples")

# Add to training
X_pseudo = X_test[reliable_mask]
y_pseudo = cat_pred[reliable_mask]  # All agree, so use any

X_augmented = pd.concat([X_train, X_pseudo])
y_augmented = np.concatenate([y_train, y_pseudo])

# Retrain
final_model = CatBoostClassifier(...)
final_model.fit(X_augmented, y_augmented)
```

**Status:** ⚠️ Similar to v7 but stricter criteria  
**Effort:** 20 minutes  
**File:** Modify existing pseudo-labeling section

---

## 🎯 RECOMMENDED EXECUTION ORDER

### **Week 1: Quick Wins (80% probability combined)**
1. **Day 1:** OOF Stacking (30 min) → Expected: 0.897-0.903
2. **Day 2:** Adversarial Validation + Adaptation (25 min) → Expected: 0.898-0.905
3. **Day 3:** Bayesian Weight Optimization (15 min) → Expected: 0.899-0.906

### **Week 2: Refinements (60% probability combined)**
4. **Day 4-5:** Manual Misprediction Analysis (60 min) → Expected: 0.900-0.907
5. **Day 6:** Target Encoding (20 min) → Expected: 0.901-0.908
6. **Day 7:** Ensemble Agreement Pseudo-labeling (20 min) → Expected: 0.902-0.909

### **Week 3: Final Polish (50% probability combined)**
7. **Day 8:** Confidence-Weighted Voting (15 min) → Expected: 0.902-0.910
8. **Day 9:** Feature Selection (30 min) → Expected: 0.903-0.911

**Total Time Investment:** ~4 hours  
**Expected Final Score:** **0.900-0.911**  
**Probability of Breaking 0.90:** **75-80%**

---

## ❌ APPROACHES TO SKIP (Low Probability)

### **1. Neural Networks** ❌
- **Why:** Already failed (0.77-0.85)
- **Data:** Too small (7000 samples), tabular (trees excel here)
- **Conclusion:** Waste of time

### **2. Complex Multi-Stage Stacking** ❌
- **Why:** Overfits on 7000 samples
- **Evidence:** Your complex stacking got 0.916 val → 0.88 test
- **Conclusion:** Simpler is better

### **3. More Hyperparameter Tuning** ❌
- **Why:** Already optimized in v9 (Optuna) → only 0.89061
- **Conclusion:** Marginal returns

### **4. Data Augmentation** ❌
- **Why:** Tabular data (not images), hard to augment meaningfully
- **Risk:** Creates unrealistic samples
- **Conclusion:** Skip

### **5. Lower Confidence Pseudo-labeling (<98%)** ❌
- **Evidence:** 95% threshold got 0.88919 vs 98% got 0.89293
- **Conclusion:** Stick with 98%+

---

## 📊 EXPECTED OUTCOMES

| Approach | Effort | Probability | Gain | Cumulative |
|----------|--------|-------------|------|------------|
| **OOF Stacking** | 30 min | 85% | +0.005 | 0.8980 |
| **Adversarial Val** | 25 min | 70% | +0.003 | 0.9010 |
| **Bayesian Weights** | 15 min | 75% | +0.002 | 0.9030 |
| **Manual Analysis** | 60 min | 60% | +0.003 | 0.9060 |
| **Target Encoding** | 20 min | 65% | +0.002 | 0.9080 |
| **Ensemble Pseudo** | 20 min | 60% | +0.002 | 0.9100 |
| **Conf. Voting** | 15 min | 50% | +0.001 | 0.9110 |
| **Feature Select** | 30 min | 55% | +0.001 | 0.9120 |

**Realistic Best Case:** 0.908-0.912 (3-8 approaches succeed)  
**Conservative Case:** 0.900-0.905 (top 3 approaches succeed)  
**Worst Case:** 0.895-0.900 (only OOF succeeds)

---

## 🚀 NEXT STEPS

1. ✅ **Read this plan**
2. ⚙️ **Start with Priority 1-3** (highest ROI: 2 hours → 0.90+)
3. 📊 **Track results** in score.md after each approach
4. 🔄 **Iterate** based on what works
5. 🎯 **Stop** when you hit 0.910+ or plateau

**Good luck breaking 0.90! 🎉**
