
# Overview

Welcome to the Pink October Challenge – a club-internal data science competition focused on predictive modeling for breast cancer screening outcomes.

Your task is to build models that accurately predict the probability of a clinical target (TARGET) for unseen data. Success will require careful exploration, feature analysis, and clever engineering rather than off-the-shelf solutions.

Tip: Patterns in the dataset may be subtle — understanding relationships, distributions, and missing value implications will be key.

# Description
The dataset contains multiple anonymized features representing patient screening and demographic information. Some features may contain missing values, unusual distributions, or hidden dependencies.

Challenge hints:

Investigate feature distributions and correlations carefully.

Some predictive signals are non-obvious and may require feature engineering.

Preprocessing decisions can significantly affect your evaluation metric.

Think like a detective: not all patterns are straightforward, and noise may hide subtle signals.

You are encouraged to experiment with different approaches, test hypotheses, and validate assumptions. Share your insights with the club to foster collaborative learning.

Dataset citation:

"Data collection and sharing was supported by the National Cancer Institute-funded Breast Cancer Surveillance Consortium (HHSN261201100031C). Learn more at BCSC ."

# Evaluation
Submissions are evaluated using ROC-AUC, measuring the ability of your model to rank positive cases higher than negative ones.

Hints to push learning:

Direct accuracy is not enough — focus on probability ranking.

Explore feature interactions, missing value patterns, and non-linear relationships.

Outlier handling and preprocessing choices can impact ROC-AUC significantly.

# Data Description
This dataset contains anonymized breast cancer screening information, designed to simulate real-world predictive modeling challenges. Participants will use this data to predict the probability of a critical clinical outcome for each individual in the test set.

Explore carefully: some features are straightforward, others require thoughtful transformation, interaction analysis, or domain reasoning. Patterns may be subtle, missing values may carry meaning, and some features may be highly correlated.

# Files
train.csv – The training set, includes features and the target variable.
test.csv – The test set, includes features only. Your task is to predict the target for these examples.
* sample_submission.csv – A correctly formatted example submission file.
# Columns / Features
example_id – Unique identifier for each record. Use this to merge with the sample submission.

feature_0 – Demographic or clinical information (numeric). May require normalization or binning. Patterns may be subtle.

feature_1 – Categorical feature representing a screening attribute. Values may encode multiple underlying conditions; explore distribution carefully.

feature_2 – Age-related feature. Likely correlated with outcome, but beware of non-linear effects.

feature_3 – Numeric feature, possibly representing a test result or measurement. Outliers may exist—consider their significance.

feature_4 – Binary indicator, may flag presence/absence of a prior condition. Consider interactions with other binary features.

feature_5 – Ordinal feature. Relationships may not be linear; encoding strategy can impact model performance.

feature_6 – Categorical feature with multiple levels. Some levels may be rare but highly predictive.

feature_7 – Continuous feature. Explore distribution, missing values, and potential transformation for non-linear effects.

feature_8 – Derived or composite metric, potentially correlated with other measurements. Look for hidden signals.

feature_9 – Time or sequence-related feature. Patterns may appear when combined with feature_2 or other demographic features.

feature_10 – Likely a screening history or prior outcome indicator. Useful for trend analysis.

TARGET (train.csv only) – Binary outcome variable. Represents the presence or absence of the condition you are predicting. Must be predicted as a binary value 0 or 1.

# Notes / Tips for Participants
Expect missing values, unusual distributions, and correlated features. Handling these cleverly may improve your model.
Some features may interact in non-obvious ways. Think creatively about feature engineering.
Explore the metaData.csv file thoroughly—some cryptic abbreviations or codes have meaningful hints.
Consider non-linear models or ensemble strategies to capture subtle relationships.
This is not just a “fit and predict” problem; insights, exploration, and reasoning are critical.