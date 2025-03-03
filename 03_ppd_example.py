# pip install interpret # install the package in a cmd prompt

# Python code to get similar plots to those in Fig. 4 and Fig.5 in the main text

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from interpret import set_visualize_provider
from interpret.provider import InlineProvider
set_visualize_provider(InlineProvider())

df = pd.read_csv("data_epds01.csv", header=None)

# Replace "_noisy" at the end of column names with an empty string
df.columns = df.columns.str.replace("_noisy$", "", regex=True)

# Remove the columns "country" and "date" from df
df = df.drop(columns=["country", "date"])

# Display the updated DataFrame
print(df)

# Renaming columns
df.columns = [
    "Age mother","Born in foreign country","Living with partner",
    "Low education","Unemployment","History of mental health problems",
    "Primiparity","Obstetrics complications","Singleton pregnancy",
    "Support by maternity care provider","Social support","Healt problems child",
    "Health problems partner","Cancer case in the household",
    "Mental disorder in household","EPDS01"
]
X = df.iloc[:, :-1]
y = (df.iloc[:, -1] == "Depression").astype(int)


seed = 42
np.random.seed(seed)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=seed)

from interpret.glassbox import ExplainableBoostingClassifier
ebm = ExplainableBoostingClassifier()
ebm.fit(X_train, y_train)

ExplainableBoostingClassifier()

from interpret import show
show(ebm.explain_global())

# Prediction with the test set for 20 randomly selected cases
show(ebm.explain_local(X_test[:20], y_test[:20]), 0)

# Using ROC curve to explain the prediction quality of EBM
from interpret.perf import ROC
ebm_perf = ROC(ebm.predict_proba).explain_perf(X_test, y_test, name='EBM')
show(ebm_perf)