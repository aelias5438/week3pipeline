# College Completion Dataset

# 1. Can we predict the graduation in 100% of time rate based on state, awards compared to national average, and percentage of full-time students
# This is interesting as predicting what leads to high graduation rates is key in discovering where to target future funding for this purpose.
# 2. A key business metric here is delta in graduation rates after funding is allocated based on predictions from the model.
# 3. My instincts tell me that prediction will be possible due to the large number of features with large perceived individual impacts on an individual's success. I’m partially worried about missing data, but it doesn’t seem to be too big of an issue

import pandas as pd
df =  pd.read_csv("Data/cc_institution_details.csv")
df.head()
df.info()
df.shape

# The data isn't ready for prediction as strings and other incorrectly formatted data are present in numeric columns.


# Job Placement Dataset

# 1. How is status influenced by academic and demographic factors.
# 2. A good metric would be the actual placement rate.
# 3. Since we have less features here, my instincts tell me to worry a bit as we don’t have many features to test for predictive ability.

import pandas as pd
pd.set_option("display.max_rows", None)
df2 = pd.read_csv("Data/job_placement.csv")
df2.head()
df2.info()
# A lot of the data is strings, definitely needs cleaning before prediction.