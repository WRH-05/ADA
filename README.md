# 🎗️ ADA Project - Micro Club Pinktober Datathon

This repository contains our solutions for **three Kaggle competitions** under the Micro Club Pinktober Datathon, focusing on breast cancer awareness and medical ML challenges.

## 🏆 Competition Overview & Results

| Competition | Type | Metric | Best Score | Rank | Status |
|-------------|------|--------|------------|------|--------|
| **Breast Cancer Risk Prediction** | Tabular (893K samples) | ROC-AUC | **0.50316** | 🥇 **1st Place** | ✅ Complete |
| **Brain Tumor Classification** | Tabular (7K samples) | F1 Score | **0.89543** | 🥈 Top 5 | ✅ Complete |
| **Breast Cancer Detection** | Image Classification | Accuracy | TBD | TBD | 🚧 In Progress |

---

## 📊 Competition Details

### 1. 🥇 Breast Cancer Risk Prediction (1st Place!)
**Challenge**: Predict probability of clinical target for breast cancer screening
- **Dataset**: 893,578 training samples, 297,862 test samples
- **Features**: 13 anonymized demographic/clinical features
- **Metric**: ROC-AUC (probability ranking)
- **Our Score**: **0.50316** (1st place, +0.00039 ahead of 2nd)
- **Key Discovery**: Target inversion requirement (1 - probability)
- **Winning Strategy**: Simple XGBoost with minimal preprocessing

**Critical Insight**: This competition had extremely weak predictive signal (0.503 vs 0.50 baseline). Our breakthrough was discovering that test predictions needed to be inverted. Complex feature engineering (40+ features) performed worse than the simple 13 original features.

📁 **Files**: 
- `winning_solution.ipynb` - Clean implementation of 1st place solution
- `METHODOLOGY.md` - Detailed winning strategy documentation
- `main.ipynb` - Full exploration with all 17 approaches tested
- `scores.md` - Complete leaderboard tracking

### 2. 🥈 Brain Tumor Classification (Top 5)
**Challenge**: Classify brain tumor cancer stages (I-IV) from clinical data
- **Dataset**: 7,000 training samples, 3,000 test samples
- **Features**: Tabular clinical/imaging features (age, tumor markers, etc.)
- **Metric**: F1 Score (macro-averaged)
- **Our Best Score**: **0.89543**
- **Approach**: Voting ensemble of CatBoost + XGBoost + LightGBM

**Key Insights**:
- Class imbalance (Stage IV: 67.6%, Stage I: 3.6%) - natural distribution works best
- Domain-specific feature engineering crucial (aggressiveness scores, risk indices)
- Ensemble diversity > hyperparameter tuning
- Stacking ensembles achieved 0.893, simple voting reached 0.895

📁 **Files**:
- `main.ipynb` - Complete pipeline with 43+ cells
- `score.md` - Top 12 submissions (cleaned from 35 experiments)
- Top submission: `new_sub.csv` (0.89543)

### 3. 🚧 Breast Cancer Detection (In Progress)
**Challenge**: Binary image classification (malignant vs normal)
- **Dataset**: Small image dataset (prone to overfitting)
- **Task**: Distinguish malignant from normal breast tissue images
- **Metric**: Accuracy
- **Status**: Data preprocessing and initial models in development

📁 **Files**:
- Data downloaded from Kaggle (see setup instructions below)
- Image augmentation and CNN architectures being tested

---

## 🎯 Overall Performance Summary

### Competition Statistics
- **Total Submissions**: 50+ across all competitions
- **Competitions Won**: 1 (Breast Cancer Risk Prediction)
- **Top 5 Finishes**: 2 (Risk Prediction + Brain Tumor)
- **Total Samples Processed**: 900K+ training samples
- **Models Deployed**: XGBoost, CatBoost, LightGBM, AutoGluon, Neural Networks

### Key Learnings Across Competitions
1. **Simplicity Often Wins**: Simple models beat complex ensembles when signal is weak (Risk Prediction)
2. **Domain Knowledge Matters**: Medical feature engineering crucial for tumor classification
3. **Ensemble Diversity**: Different model families (CatBoost + XGBoost + LGBM) > same model with different params
4. **Validation Strategy**: Always verify on actual test set - validation scores can be misleading
5. **Systematic Testing**: Documented 17 approaches for Risk, 35 for Tumor - data-driven decisions

### Competitive Advantages
- ✅ Systematic experimentation with detailed documentation
- ✅ Early discovery of critical bugs (target inversion)
- ✅ Balanced approach: tried complexity, kept simplicity when better
- ✅ Strong ensemble techniques (voting, stacking, blending)
- ✅ Comprehensive score tracking in markdown files

---

## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/WRH-05/ADA.git
cd ADA
```

### 2. Download Datasets

Download the competition datasets from Kaggle:

#### Breast Cancer Risk Prediction (Tabular Data)
1. Go to [Breast Cancer Risk Prediction](https://www.kaggle.com/competitions/micro-club-pinktober-breast-cancer-risk-prediction/data)
2. Download `train.csv`, `test.csv`, `sample_submission.csv`
3. Place in `Breast Cancer Risk Prediction/` folder

#### Brain Tumor Classification (Tabular Data)
1. Go to [Brain Tumor Classification](https://www.kaggle.com/competitions/micro-club-pinktober-brain-tumor-classification/data)
2. Download `train.csv`, `test.csv`, `sample_submission.csv`
3. Place in `Brain Tumor Classification/` folder

#### Breast Cancer Detection (Image Data)
1. Go to [Breast Cancer Detection](https://www.kaggle.com/competitions/micro-club-pinktober-breast-cancer-detection/data)
2. Download train and test images
3. Extract and place in this structure:
   ```
   Breast Cancer Detection/
   ├── test/
   │   └── (place test images here)
   └── train/
       ├── malignant/
       └── normal/
   ```

**Note:** Data files are not included in this repository due to size. They are Git-ignored to keep the repo lightweight.

### 3. Create Your Branch

Before making any changes, create your own branch:

```bash
git checkout -b your-branch-name
```

Use a descriptive branch name, e.g., `feature/model-improvement` or `fix/data-preprocessing`

### 4. Make Your Changes

- Work on your branch locally
- Make commits regularly with clear messages:
  ```bash
  git add .
  git commit -m "Clear description of what you changed"
  ```

### 5. Push Your Branch

Push your branch to the remote repository:

```bash
git push origin your-branch-name
```

### 6. Create a Pull Request

1. Go to the [GitHub repository](https://github.com/WRH-05/ADA)
2. Click on "Pull requests" tab
3. Click "New pull request"
4. Select your branch to merge into `main`
5. Add a clear title and description of your changes
6. Submit the pull request for review

### 7. Keep Your Branch Updated

If the main branch gets updated while you're working, sync your branch:

```bash
git checkout main
git pull origin main
git checkout your-branch-name
git rebase main
```

**Note:** Do NOT merge or push directly to `main`. All changes must go through a pull request and be approved first.

## 📁 Project Structure

```
ADA/
├── Breast Cancer Risk Prediction/        # 🥇 1st Place (ROC-AUC: 0.50316)
│   ├── winning_solution.ipynb           # Clean winning implementation
│   ├── METHODOLOGY.md                    # Detailed strategy & insights
│   ├── main.ipynb                        # Full exploration (17 approaches)
│   ├── scores.md                         # Competition leaderboard
│   ├── train.csv                         # 893K samples (download)
│   ├── test.csv                          # 298K samples (download)
│   └── submission_inverted.csv           # Winning submission
│
├── Brain Tumor Classification/           # 🥈 Top 5 (F1: 0.89543)
│   ├── main.ipynb                        # Complete pipeline (43 cells)
│   ├── score.md                          # Top 12 submissions
│   ├── new_sub.csv                       # Best submission (0.89543)
│   ├── train.csv                         # 7K samples (download)
│   ├── test.csv                          # 3K samples (download)
│   └── autogluon_models/                 # AutoML experiments
│
├── Breast Cancer Detection/              # 🚧 In Progress
│   ├── test/                             # Test images (download)
│   ├── train/                            # Training images (download)
│   │   ├── malignant/                    # Malignant tissue images
│   │   └── normal/                       # Normal tissue images
│   ├── README.md                         # Competition description
│   └── sample_submission.csv
│
├── .gitignore                            # Excludes data/models (size limits)
└── README.md                             # This file
```

### Key Files to Explore

**For Breast Cancer Risk Prediction (1st Place Solution)**:
- Start with `winning_solution.ipynb` for clean implementation
- Read `METHODOLOGY.md` for the full story of how we achieved 1st
- Check `main.ipynb` to see all 17 approaches we tested

**For Brain Tumor Classification**:
- `main.ipynb` contains the complete pipeline
- `score.md` shows progression from 0.88 to 0.89543

---

## 🛠️ Technical Stack

### Libraries & Frameworks
- **Tree-Based Models**: XGBoost, CatBoost, LightGBM
- **AutoML**: AutoGluon (brain tumor classification)
- **Deep Learning**: PyTorch, TensorFlow/Keras (image classification)
- **Data Processing**: pandas, numpy, scikit-learn
- **Visualization**: matplotlib, seaborn

### Model Techniques Used
- **Ensemble Methods**: Voting, Stacking, Blending, Weighted Averaging
- **Feature Engineering**: Domain-specific medical features, interactions, transformations
- **Handling Imbalance**: scale_pos_weight, class_weight, natural distributions
- **Validation**: Stratified K-Fold, Temporal splits, Cross-validation
- **Hyperparameter Tuning**: Optuna, Grid Search, Manual optimization

---

## 🎓 Key Takeaways & Best Practices

### What Worked Well
1. **Systematic Documentation**: Tracking all submissions in `scores.md` files
2. **Hypothesis Testing**: Testing target inversion when scores were < 0.50
3. **Ensemble Diversity**: Combining CatBoost + XGBoost + LightGBM
4. **Simple Over Complex**: 13 features beat 40+ engineered features (Risk Prediction)
5. **Domain Knowledge**: Medical feature engineering (aggressiveness scores, risk indices)

### What We Learned
1. **High validation ≠ high test score**: Always verify on actual leaderboard
2. **Baseline testing matters**: Submitting all 0.5 predictions revealed true difficulty
3. **Know when to stop**: After 17 approaches, simple XGBoost remained best
4. **Label encoding issues**: Always check for target inversion requirements
5. **Natural distributions**: Don't over-balance classes without testing

### Debugging Strategies
- Compare validation vs test performance gaps
- Test prediction inversions when scoring below random baseline
- Submit multiple approaches in parallel
- Document what fails as thoroughly as what succeeds

---

## 📈 Competition Timeline & Milestones

### Breast Cancer Risk Prediction
- **Week 1**: Initial exploration, high validation scores (0.90+), terrible test scores (0.498)
- **Week 2**: Discovery of target inversion bug → 0.503 (breakthrough!)
- **Week 3**: Tested 17 approaches, simple XGBoost remained best
- **Final**: 0.50316 ROC-AUC (**1st Place** by +0.00039 margin)

### Brain Tumor Classification
- **Week 1**: Baseline models, feature engineering (0.88-0.89)
- **Week 2**: Ensemble experiments, reached 0.893 with stacking
- **Week 3**: Voting ensemble breakthrough → 0.895
- **Final**: 0.89543 F1 Score (**Top 5**)

---

## 🤝 Contributing

### Branch Strategy
- **main**: Protected, stable code only
- **chrome/risk**: Breast Cancer Risk experiments
- **chrome/brain**: Brain Tumor Classification experiments
- **chrome/9att**: Experimental features (0.90+ attempts)

### Workflow
1. Always work on a separate branch, never directly on `main`
2. Write clear commit messages describing what changed
3. Test your code before creating a pull request
4. Keep pull requests focused on a single feature or fix
5. Update `scores.md` files when submitting to Kaggle

### Code Standards
- Document all experiments in notebooks with markdown cells
- Track submission scores immediately after Kaggle submission
- Use meaningful variable names (X_train, y_train, not df1, df2)
- Include comments for complex feature engineering
- Clean up unused cells before committing

---

## 🏅 Acknowledgments

- **Micro Club** for organizing the Pinktober Datathon
- **Kaggle** for hosting the competitions
- **BCSC** (Breast Cancer Surveillance Consortium) for the Risk Prediction dataset
- All team members who contributed to experimentation and model development

---

## 📞 Contact & Questions

If you have questions or want to discuss our approaches:
- Open an issue on GitHub
- Check `METHODOLOGY.md` in Breast Cancer Risk Prediction for detailed insights
- Review `score.md` files for submission history

**Repository**: [github.com/WRH-05/ADA](https://github.com/WRH-05/ADA)

---

## 📄 License

This project is for educational and competition purposes. Please respect Kaggle competition rules and data usage policies.

**Last Updated**: November 1, 2025
