# College Completion Dataset

# 1. Can we predict whether a college is an HBCU based on its demographics, funding amounts, and academic outcomes?
# This interesting because we could see whether HBCU's are ultimately achieving what they set out to do, which is provide a college opportunity for Black Americans.
# 2. A key business metric here could be testing institutional classification accuracy.
# 3. My instincts tell me that there are a large number of features that we can use to predict, there is a good mix of categorical and numerical data that cn help diversify modeling approaches.I'm worried about the missing data in several columns.

import pandas as pd
df =  pd.read_csv("Data/cc_institution_details.csv")
df.head()
df.info()
df.shape

# I see that there is a ton of non-null and strings that need to be corrected. I'll attempt to solve the problem in q4



# Job Placement Dataset

# 1. Can we predict a student's expected salary based on how they perform academically, and their work experience
# 2. A good IBM would be the accuracy of the salary prediction.
# 3. My insticts tell me that there is a smaller sample size which could hurt us. Also the salary only exists for students that were places which could be a form of selection bias.

import pandas as pd
pd.set_option("display.max_rows", None)
df2 = pd.read_csv("Data/job_placement.csv")
df2.head()
df2.info()
# I notice that alnmost half of them are strings. This could hurt us later on potentially.
