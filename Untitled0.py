# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

import qimbs 
import numpy as np
import pandas as pd
from ggplot import *
from sklearn.pipeline import Pipeline

# <codecell>

#!Importing data
df = qimbs.import_month(7)

# <codecell>

from sklearn.ensemble import RandomForestClassifier as RF

# <codecell>

pipeline = Pipeline([
  ('add_timestamp', Add_timestamp()),
  ('classifier', RF())
])

# <codecell>

class Add_timestamp():

    def transform(self, X, **transform_params):
        return qimbs.create_timestamp(X)       

    def fit(self, X, y=None, **fit_params):
        return self

# <codecell>

pipeline.fit(df)

# <codecell>


