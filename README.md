# ADA Project - Machine Learning Models

This repository contains machine learning projects for medical image classification.

## Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/WRH-05/ADA.git
cd ADA
```

### 2. Download Dataset

After cloning, you'll need to download the **Breast Cancer Detection** dataset from Kaggle:

1. Go to the [Breast Cancer Detection competition on Kaggle](https://www.kaggle.com/competitions/micro-club-pinktober-breast-cancer-detection/data)
2. Download the train and test data
3. Extract the files and place them in the following structure:
   ```
   Breast Cancer Detection/
   ├── test/
   │   └── (place test images here)
   └── train/
       ├── malignant/
       └── normal/
   ```

**Note:** The data files are not included in this repository due to their size. They are ignored by Git to keep the repository lightweight.

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

## Project Structure

```
ADA/
├── Brain Tumor Classification/
│   ├── train.csv
│   ├── test.csv
│   └── sample_submission.csv
├── Breast Cancer Detection/
│   ├── test/ (download from Kaggle)
│   ├── train/ (download from Kaggle)
│   └── sample_submission.csv
└── README.md
```

## Contributing

- Always work on a separate branch, never directly on `main`
- Write clear commit messages
- Test your code before creating a pull request
- Keep pull requests focused on a single feature or fix

## Questions?

If you have any questions or run into issues, please open an issue on GitHub or contact the repository maintainer.
