# Databricks notebook source
# MAGIC %md
# MAGIC 
# MAGIC ## Big Data Technical Project 1 Step 2 - Further Exploration, assess and visualize to find instights on the data.
# MAGIC The intent of this workbook is to injest the cleaned data file from step1 to carry out further analysis on relationships within the data to understand insights that can be identified to assist in creating a model that allows us to predict if someone will default on there loan, it is expected that further fields and data will have to be cleansed for use in any ML prediction 
# MAGIC 
# MAGIC Source Control file recorded here : https://github.com/sharpyATU/ATUBigData1/blob/main/BigDataTR%20-%20Step%202.py

# COMMAND ----------

import pandas as pd
import numpy as np
import pyspark.pandas as ps

import matplotlib
import matplotlib.pyplot as plt

import seaborn as sns
sns.set(color_codes=True)
sns.set_palette("Accent")

from scipy import stats

import plotly.plotly as py
import plotly.graph_objs as go

from plotly.offline import init_notebook_mode,iplot
init_notebook_mode(connected=True)


!pip install -U pandas-profiling[notebook]


# COMMAND ----------

#Load tjhe data file
file_location = "/FileStore/tables/step2_0402_2007_to_2018Q4-2201.csv"
#file_location = "/FileStore/tables/step2_0202_2007_to_2018Q4-2201.csv"
#file_location = "/FileStore/tables/step2_2007_to_2018Q4-2201.csv"
#file_location = "/FileStore/tables/tuncated_2007_to_2018Q4-2201.csv"
file_type = "csv"

# CSV options
infer_schema = "true"
first_row_is_header = "true"
delimiter = ","

# The applied options are for CSV files. For other file types, these will be ignored.
df_master = spark.read.format(file_type) \
  .option("inferSchema", infer_schema) \
  .option("header", first_row_is_header) \
  .option("sep", delimiter) \
  .load(file_location)

display(df_master)

# COMMAND ----------

#Ouput Schema so we have a reference of fields and types whilst working 
df_master.printSchema()

# COMMAND ----------

# MAGIC %md
# MAGIC We have alaredy explored the data fields and shape so lets start looking at some of the features and releationships within the data

# COMMAND ----------

#Run an action command , this will allow loading to be executed, and enable better optimisation and distribution  of the DAG before further analytics
df_master.describe().show()

# COMMAND ----------

# MAGIC %md
# MAGIC ##Lets observe the pattern of loans in the book 
# MAGIC 
# MAGIC We have a lot of date over a long period we can observe how the loan book has grown over time and any trend on the data

# COMMAND ----------

#Field issue_d, holds Date/Period information, but exsist as a string field, we have a lot of date over a number of years, so will extract the year, so we can observe behaviour over years. 
from pyspark.sql import functions as F
from pyspark.sql.functions import isnan, when, count, col, avg, split

#Split the year out of the data DEC-2015 using the second field of the _
df_master= df_master.withColumn('issue_d_year', (split(df_master['issue_d'], '-').getItem(1)).cast('int'))

# COMMAND ----------

# MAGIC %md

# COMMAND ----------

# Create a temporary table to hold the data for optimised queries via Spark SQL and Temp tables etc..
temp_table_name = "loantempdata"
df_master.createOrReplaceTempView(temp_table_name)


# COMMAND ----------

# configure a matix  of plots to observe the loans issued
fig, ax =plt.subplots(2,2, figsize=(18,12))
plt.subplots_adjust(hspace = 0.4, top = 0.8)

# Gather and graph the Median Loan Amount, by each issue year
medianloanOverTime = sqlContext.sql("select issue_d_year, percentile_approx(loan_amnt, 0.5) as loan_amnt_median_value from loantempdata group by issue_d_year order by issue_d_year").toPandas()
medianloanOverTime.columns = ["Issue Period", "Median Loan Amount"]
plot0 = sns.pointplot(x=medianloanOverTime['Issue Period'], y=medianloanOverTime["Median Loan Amount"], capsize=.2, ax=ax[1][0])
plot0.set_xticklabels(plot0.get_xticklabels(),rotation=90)

# Gather and graph the Median Interest rate, for each year 
medianIntrestOverTime = sqlContext.sql("select issue_d_year, percentile_approx(int_rate, 0.5) as int_rate_median_value from loantempdata group by issue_d_year order by issue_d_year").toPandas()
medianIntrestOverTime.columns = ["Issue Period", "Median Interest rate"]
plot1 = sns.pointplot(x=medianIntrestOverTime['Issue Period'], y=medianIntrestOverTime["Median Interest rate"], capsize=.2, ax=ax[1][1])
plot1.set_xticklabels(plot1.get_xticklabels(),rotation=90)

# Gather and graph number of loan over year
loansOverTime = df_master.sort("issue_d_year").groupBy("issue_d_year").count().toPandas()
loansOverTime.columns = ["Issue Period", "Number of Loans"]
plot2 =sns.pointplot(x=loansOverTime["Issue Period"], y=loansOverTime["Number of Loans"], ax=ax[0][0])
plot2 .set_xticklabels(plot2.get_xticklabels(),rotation=90)

# Gather and graph the total loans over each year
totalloansOverTime = df_master.sort("issue_d_year").groupBy("issue_d_year").sum('loan_amnt').toPandas()
totalloansOverTime.columns = ["Issue Period", "Total Loan Amount"]
plot3 = sns.pointplot(x=totalloansOverTime['Issue Period'], y=totalloansOverTime["Total Loan Amount"], capsize=.2, ax=ax[0][1])
plot3.set_xticklabels(plot3.get_xticklabels(),rotation=90)

# COMMAND ----------

# MAGIC %md
# MAGIC From the above graphs we can observe that the loan book grew at ever increasing growth rate until 2015, the number of loan and the size of growth of the loans issue slowed signifcantly over 2015-2017.
# MAGIC The median loan amount also increased from 2007 until 2015, where it seems the reduction in growth corresponds to a lower median rate of loan being issued.
# MAGIC The max median interest rate was observed in 2013, 

# COMMAND ----------

# MAGIC %md
# MAGIC Lets look at the relationship between Loan amount and Interest rates
# MAGIC  
# MAGIC -Frequency distribution of loan amount 
# MAGIC -Five number summary distribution of loan amount.
# MAGIC -Frequency distribution of interest rates
# MAGIC -Five number summary distribution of interest rates.

# COMMAND ----------

# Set the reduced dataframe to observe the profiles of loan amounts and interest rates with the lending loan set 
tmpLoan = df_master.select("loan_amnt", "int_rate").toPandas()

fig, ax =plt.subplots(2,2, figsize=(24,16))
plt.subplots_adjust(hspace = 0.4, top = 0.8)

# Plots for Loan amount distribution
sns.distplot(tmpLoan.loan_amnt, fit=stats.gamma, axlabel="Loan Amount", label="Loan Amount Freq Dist", ax=ax[0][0])
sns.boxplot(x=tmpLoan.loan_amnt, ax=ax[0][1])

# Plots for Interest rates distribution
sns.distplot(tmpLoan.int_rate, fit=stats.gamma, axlabel="Interest Rate", label="Interest Freq Dist", ax=ax[1][0])
sns.boxplot(x=tmpLoan.int_rate, ax=ax[1][1])

fig.show()

# COMMAND ----------

# MAGIC %md
# MAGIC Its observable that within the dataset that both Loan Amount and Interest rates are right skewed with most data falling within 10K-20K loans at an interest rate of  9-16 but it's observable there is a number of outlier customers with larger loans or paying much higher interest rates. 

# COMMAND ----------

# MAGIC %md
# MAGIC We will now explore the categories/class of loans issued by the lending club

# COMMAND ----------

# MAGIC %sql
# MAGIC /* View the different strings in the emp_length */
# MAGIC select distinct grade from loantempdata order by grade

# COMMAND ----------

# MAGIC %sql
# MAGIC /* View the different strings in the emp_length */
# MAGIC select distinct sub_grade from loantempdata order by sub_grade

# COMMAND ----------

# MAGIC %md
# MAGIC It's well defined that the Grades are ordered in 7 categories (A-G) and each has 5 sub cateogories eg. A1-A5. Lets look at the loan interest rates attribute to these categories

# COMMAND ----------

#Select a sub df to investigate
tmpLoan = df_master.select("sub_grade", "grade", "loan_amnt", "int_rate").toPandas()
#Convert to categorical data 
tmpLoan['grade'] = tmpLoan['grade'].astype('category')
tmpLoan['sub_grade'] = tmpLoan['sub_grade'].astype('category')

#Set the column wrap here to 20 to visuallise the interest increase per sub grade
subg = sns.FacetGrid(tmpLoan, col="sub_grade", sharex=False, col_wrap=20, col_order=["A1", "A2", "A3", "A4", "A5","B1", "B2", "B3", "B4", "B5","C1", "C2", "C3", "C4", "C5","D1", "D2", "D3", "D4", "D5","E1", "E2", "E3", "E4", "E5","F1", "F2", "F3", "F4", "F5","G1", "G2", "G3", "G4", "G5"])
subg.map(sns.boxplot, 'grade', 'int_rate')

# COMMAND ----------

#Perform the same step but use a col_wrap of 5 to observer the variane in interes from sub group classes e.g A1,B1,C1 etc...
subg = sns.FacetGrid(tmpLoan, col="sub_grade", sharex=False, col_wrap=5, col_order=["A1", "A2", "A3", "A4", "A5","B1", "B2", "B3", "B4", "B5","C1", "C2", "C3", "C4", "C5","D1", "D2", "D3", "D4", "D5","E1", "E2", "E3", "E4", "E5","F1", "F2", "F3", "F4", "F5","G1", "G2", "G3", "G4", "G5"])
subg.map(sns.boxplot, 'grade', 'int_rate')

# COMMAND ----------

# MAGIC %md
# MAGIC It's observable that that the interest rates increase as we go through each sub category e.g A1-A5 and the outliers increase as we increase subcategory
# MAGIC It's also obeservable that the intrest rates increase as we go through each category with the degree of outliers also increasing through Category A-G.
# MAGIC 
# MAGIC ##It wouldn't be unreasonable to think that the higher the interest rate paid the more likely a customer is to default on a loan##

# COMMAND ----------

#With that in mind lets look at the status of Loans against there grades 
tempStatus = df_master.stat.crosstab('loan_status','grade').show()

# COMMAND ----------

# MAGIC %md
# MAGIC We can see the dataset has primarily loans that are fully paid or are current which means are being paid and up to date. As we move to attempting to predict if a customer will default we will probably have to combine the various categories of being unable to pay a loan. 
# MAGIC 
# MAGIC Lets explore further the relationship of other data with defaulting/failing to pay a loan
# MAGIC 
# MAGIC Looking Back at the sample dataset, field data types and data dictionary the following fields are numerical but are present in the data as strings. These data fields would be expected to have some form of relationship with defaulting on a loan
# MAGIC -term -> has months with textual data need to convert from sting to numerical type
# MAGIC -emp_length - has years textual information , need to convert from sting to numerical type

# COMMAND ----------

##dftemp= df_master.select().describe().toPandas()#('term','int_rate','loan_amnt', 'emp_length','annual_inc', 'dti','delinq_2yrs').describe().toPandas()
#Data Profile was still quite messy to view (Pandas for spark would be useful here.)
df_master.describe('term','emp_length').show()

# COMMAND ----------

# MAGIC %sql
# MAGIC /* View the different strings in the emp_length */
# MAGIC select distinct emp_length from loantempdata limit 1000

# COMMAND ----------

#Evalutate how to clean the emp_length string
from pyspark.sql.functions import  regexp_replace, regexp_extract
from pyspark.sql.functions import col
regex_filterstring = 'years|year|\\+|\\<'
df_master.select(regexp_replace(col("emp_length"), regex_filterstring, "").alias("emplength_clean"), col("emp_length")).show(15)

# COMMAND ----------

# Evalute Getting rid of the Months string
from pyspark.sql.functions import  regexp_replace, regexp_extract
from pyspark.sql.functions import col
#Extract digits data
regex_filterstring = '\\d+'  #Alternative filter
df_master.select(regexp_extract(col("term"), regex_filterstring, 0).alias("term_clean"), col("term")).show(10)

# COMMAND ----------

## need to get these into dataset and keep clean further
df_master=df_master.withColumn('emplength_clean', regexp_replace(col("emp_length"), 'years|year|\\+|\\<', ""))
df_master=df_master.withColumn("term_clean",regexp_extract(col("term"), '\\d+', 0))

# COMMAND ----------

df_master.describe('term_clean','emplength_clean').show()

# COMMAND ----------

These Fields are now clean.

# COMMAND ----------

df_master.describe()

# COMMAND ----------

ANDY YOU ARE HERE.

# COMMAND ----------


df_cleaning = df_cleaning.withColumn('emplength_clean', df_cleaning['emplength_clean'].cast(dataType=int))
###Theres' possibly some shifty data still in your emplenght field

# COMMAND ----------


# Write a custom function to convert the data type of DataFrame columns
def convertColumn(df, names, newType):
  for name in names: 
     df = df.withColumn(name, df[name].cast(newType))
  return df 

# Assign all column names to `columns`
columns = ['emplength_clean'] #', 'emplength_clean''

# Conver the `df` columns to `FloatType()`
df_cleaning = convertColumn(df_cleaning, columns, )
    

# COMMAND ----------

#Example of grouping
# Otherwise you can first write your function
# as you can see here we have more flexibility
# I will write the function to also incorporate the total number of reviews
def watchable_udf(avg_rating, reviews):
  if avg_rating > 3.5 and reviews > 50:
    return 'yes'
  elif avg_rating > 3.5 and reviews < 50:
    return 'maybe'
  else:
    return 'no'
# and then register it as an UDF with the return type declared
watchable_udf = udf(watchable_udf, StringType())

# COMMAND ----------

stages = []
categorical_cols = ['Embarked', 'Sex'] 

for categorical_col in categorical_cols:
    string_indexer = StringIndexer(inputCol=categorical_col, outputCol=categorical_col + "_index", handleInvalid='error')
    stages += [string_indexer]

assembler_inputs = numerical_cols + [c + "_index" for c in categorical_cols]
assembler = VectorAssembler(inputCols=assembler_inputs, outputCol="features")

stages  += [assembler]
