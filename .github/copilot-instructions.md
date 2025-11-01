# AI Agent Instructions for ADA Project

## Project Overview
**ADA** is a Kaggle competition repository for medical ML classification tasks. Currently contains:
- **Brain Tumor Classification**: Tabular data (7000 train samples) predicting cancer stages I-IV using clinical/imaging features (PRIMARY FOCUS)
- **Breast Cancer Detection**: Image classification (secondary, future work)

**Critical**: This is a Kaggle competition project optimized for leaderboard scores, not production ML systems. F1 score is the evaluation metric.

### Kaggle Competition Links
- **Brain Tumor Classification (Data)**: https://www.kaggle.com/competitions/micro-club-pinktober-brain-tumor-classification/data
- **Brain Tumor Classification (Leaderboard)**: https://www.kaggle.com/competitions/micro-club-pinktober-brain-tumor-classification/leaderboard
- **Breast Cancer Detection**: https://www.kaggle.com/competitions/micro-club-pinktober-breast-cancer-detection

## Architecture & Data Flow

### Brain Tumor Classification Pipeline
1. **Data**: `train.csv` (7000 rows) + `test.csv` (3000 rows) - tabular features only, no images
2. **Feature Engineering**: Domain-specific medical features (aggressiveness_score, risk_score, age_ki67_interaction, etc.) - see `main.ipynb` cells 11-12
3. **Model Stack**: CatBoost → XGBoost → LightGBM → Stacking Ensemble → Extra Trees (optional)
4. **Output**: `sub*.csv` files matching `sample_submission.csv` format

**Key Insight**: Features are mix of categorical (tumor_type, location, size) and numerical (ki67_index, mitotic_count). Gender field has non-standard values ('amira', 'wa7ch', 'male', 'female') requiring label encoding.

## Critical Workflows

### Notebook Execution (Brain Tumor Classification)
```python
# ALWAYS run cells sequentially - models depend on trained predecessors
# Total execution time: ~90-120 minutes for full pipeline
# Cell order matters: Feature engineering → Encoding → Train/Val split → Model training

# Key variables that MUST exist before final predictions:
# - best_model, best_final_f1, target_encoder, X_test_final
```

### Git Branching Strategy
- **main**: Protected, PR-only merges
- **chrome/brain**: Best performing model branch (0.89277 Kaggle score)
- **chrome/9att** (or chrome/0.9+): Experimental branches for 0.90+ attempts
- **Pattern**: Create `chrome/[descriptor]` branches for experimentation

**Workflow**:
```bash
# Revert to best version if experiment fails:
git checkout chrome/brain
git restore "Brain Tumor Classification/main.ipynb"

# Save working versions before major changes
git checkout -b chrome/backup-[date]
```

## Project-Specific Conventions

### 1. **Class Imbalance is NOT a Bug**
Training data distribution: Stage I (3.6%), Stage II (6.9%), Stage III (21.9%), Stage IV (67.6%)
- **DO NOT** apply aggressive class balancing - it consistently degrades F1 score
- Natural distribution reflects medical reality and test set distribution
- Class balancing attempts dropped validation F1 from 0.888 → 0.800

### 2. **Ensemble Strategy Over Hyperparameter Tuning**
Winning approach: Diverse model families > deep single-model optimization
```python
# GOOD: Different algorithms with different random seeds
ensemble = [Stacking(CatBoost+XGBoost+LightGBM), CatBoost(seed=43), ExtraTrees]

# BAD: Same model with different hyperparameters
ensemble = [XGBoost(params1), XGBoost(params2), XGBoost(params3)]
```

### 3. **Submission File Naming Convention**
- `subChromium.csv`, `subChromium1.csv`: Production submissions
- `subC.csv`, `subC1.csv`, `subC2.csv`: Experimental submissions
- Pattern: Include branch/strategy identifier in filename for tracking

### 4. **Feature Engineering is Domain-Specific**
Medical ML patterns already implemented (don't reinvent):
- `aggressiveness_score = ki67_index * 0.5 + mitotic_count * 2.5` (medical domain knowledge)
- Age/Ki67 interaction terms (cancer staging correlates with age)
- Risk scoring based on pathological markers (necrosis, hemorrhage, edema)

## Integration Points & Dependencies

### External Dependencies
- **Kaggle Dataset**: Must download manually, not in repo (see README.md)
- **CatBoost**: Handles categorical features natively, often best performer
- **No GPU required**: All models use CPU (`task_type='CPU'`, `tree_method='hist'`)

### Model Persistence
**NEW PRACTICE**: Save trained models to avoid retraining
```python
import joblib

# Save the best model after training
joblib.dump(best_model, 'best_brain_tumor_model.pkl')
joblib.dump(target_encoder, 'target_encoder.pkl')

# Load for predictions
best_model = joblib.load('best_brain_tumor_model.pkl')
target_encoder = joblib.load('target_encoder.pkl')
```

**File naming convention**:
- `best_brain_tumor_model.pkl`: Current best performer
- `catboost_solo.pkl`, `extra_trees.pkl`: Individual models for ensemble
- Add `.pkl` files to `.gitignore` (large file sizes)

### Critical Files
- `main.ipynb`: 43 cells (base) or 48+ cells (with ensemble enhancement)
- `train.csv` / `test.csv`: Original Kaggle data (DO NOT MODIFY)
- `sample_submission.csv`: Submission format template
- `catboost_info/`: Auto-generated training logs (safe to ignore)
- `*.pkl`: Model checkpoints (not in repo - add to `.gitignore`)

## Common Pitfalls & Solutions

### ❌ **"My F1 score dropped after adding [technique]"**
**Likely causes**:
1. Applied class balancing (remove `auto_class_weights`, `sample_weight`, `class_weight`)
2. Used pseudo-labeling with low confidence threshold (<98%)
3. Added polynomial features without validation (they often hurt)

**Solution**: Revert to `chrome/brain` branch baseline (0.89277)

### ❌ **"Notebook variables are undefined"**
**Cause**: Ran cells out of order or skipped critical setup cells
**Solution**: Restart kernel + "Run All" (takes 90-120 min)

### ❌ **"Predictions have wrong distribution"**
**Expected**: ~70% Stage IV, ~22% Stage III, ~7% Stage II, ~1% Stage I
**If different**: Model isn't learning correctly - check:
1. Feature engineering applied to both train AND test
2. Label encoding used same encoder for train/test
3. No data leakage between train/val split

## Debugging Quick Reference

### Validation F1 Score Benchmarks
- 0.82-0.85: Poor (baseline Random Forest)
- 0.86-0.88: Good (single optimized model)
- 0.88-0.89: Excellent (stacking ensemble) ← Current best
- 0.90+: Competition winning (requires ensemble of 3+ diverse models)

### When to Use Which Model
- **CatBoost**: Best for categorical-heavy data, use as baseline
- **XGBoost**: Best for numerical features with regularization
- **LightGBM**: Fastest training, good for experimentation
- **Stacking**: When single models plateau (0.88+)
- **Extra Trees**: Final ensemble diversity boost

## Testing & Validation

**No unit tests** - this is Kaggle competition code, validation is:
1. **Validation F1 score** in notebook (split from train)
2. **Public leaderboard score** from Kaggle submission
3. **Private leaderboard** reveals final ranking (end of competition)

**Always check**:
- Validation distribution matches training distribution
- No Stage I over-prediction (should be <5%)
- Stage IV around 65-75% (not 40% or 90%)

---

**Last Updated**: Based on chrome/brain branch achieving 0.89277 (3rd place) with stacking ensemble of CatBoost+XGBoost+LightGBM.