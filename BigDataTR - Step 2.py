# Databricks notebook source
# MAGIC %md
# MAGIC 
# MAGIC ## Big Data Technical Project 1 Step 2 - Further Exploration, assess and visualize to find insights on the data.
# MAGIC The intent of this second  workbook is to injest the cleaned data file from step1 to carry out further analysis on relationships within the data to understand insights that can be identified to assist in creating a model that allows us to predict if someone will default on there loan, it is expected that further fields and data will have to be cleansed for use in any ML prediction 
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

!pip install -U pandas-profiling[notebook]

import plotly.plotly as py
import plotly.graph_objs as go
from plotly.offline import init_notebook_mode,iplot

init_notebook_mode(connected=True)

# COMMAND ----------

# Load the data file

file_location = "/FileStore/tables/step2_0402_2007_to_2018Q4_2201.csv"
#file_location = "/FileStore/tables/step2_0402_2007_to_2018Q4-2201.csv"
# file_location = "/FileStore/tables/step2_0202_2007_to_2018Q4-2201.csv"
# file_location = "/FileStore/tables/step2_2007_to_2018Q4-2201.csv"
# file_location = "/FileStore/tables/tuncated_2007_to_2018Q4-2201.csv"
file_type = "csv"

# CSV options
infer_schema = "true"
first_row_is_header = "true"
delimiter = ","

# The applied options are for CSV files. For other file types, these will be ignored.
df_master = (
    spark.read.format(file_type)
    .option("inferSchema", infer_schema)
    .option("header", first_row_is_header)
    .option("sep", delimiter)
    .load(file_location)
)

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
# MAGIC ##Observe the pattern of loans in the book 
# MAGIC 
# MAGIC We have a lot of date over a long period we can observe how the loan book has grown over time and any trend on the data

# COMMAND ----------

#Field issue_d, holds Date/Period information, but exists as a string field, we have a lot of date over a number of years, so will extract the year, so we can observe behaviour over years. 
from pyspark.sql import functions as F
from pyspark.sql.functions import isnan, when, count, col, avg, split,lit

#Split the year out of the data DEC-2015 using the second field of the _
df_master= df_master.withColumn('issue_d_year', (split(df_master['issue_d'], '-').getItem(1)).cast('int'))

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
# MAGIC From the above graphs we can observe that the loan book grew at ever increasing growth rate until 2015, the number of loans and the size of growth of the loans issue slowed signifcantly over 2015-2017.
# MAGIC The median loan amount also increased from 2007 until 2015, where it seems the reduction in growth corresponds to a lower median rate of loan being issued.
# MAGIC The max median interest rate was observed in 2013, 

# COMMAND ----------

# MAGIC %md
# MAGIC ##Observe relationship between Loan amount and Interest rates 

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
# MAGIC Its observable that within the dataset that both Loan Amount and Interest rates are right skewed with most data falling within 10K-20K loans at an interest rate of  9-16%  It's also observable there is a number of outlier customers with larger loans or paying much higher interest rates. 

# COMMAND ----------

# MAGIC %md
# MAGIC It would be reasonable to believe that the number of years someone has beem employed would have relationship to loans. Lets explore this with respect to term of the loan, the value of the loans and the interest rates provided on the loan.
# MAGIC 
# MAGIC Looking back at the sample dataset / scheme (above) and the category information in YData from STEP 1 term and emp_length are presented as strings but would be expected to be mumerical in format.
# MAGIC 
# MAGIC Let's explore this before visualising any data

# COMMAND ----------

#Show the Min/Max on term and emp_lenght to observe format of data
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

# Evalute Getting rid of the Months string in term
from pyspark.sql.functions import  regexp_replace, regexp_extract
from pyspark.sql.functions import col
#Extract digits data
regex_filterstring = '\\d+'  #Alternative filter
df_master.select(regexp_extract(col("term"), regex_filterstring, 0).alias("term_clean"), col("term")).show(10)

# COMMAND ----------

# Add these into dataset cleaned for numerical format
df_master=df_master.withColumn('emplength_clean', regexp_replace(col("emp_length"), 'years|year|\\+|\\<', ""))
df_master=df_master.withColumn("term_clean",regexp_extract(col("term"), '\\d+', 0))

# COMMAND ----------

#Check data has been updated appropriately
df_master.describe('term_clean','emplength_clean').show()

# COMMAND ----------

# Modify the updated fields from strings to ints
df_master = df_master.withColumn('emplength_clean', df_master['emplength_clean'].cast('int'))
df_master = df_master.withColumn('term_clean', df_master['term_clean'].cast('int'))

# COMMAND ----------

# MAGIC %md
# MAGIC Now our data is clean (which will help with future ML as well) we can obeserve relationships.

# COMMAND ----------

fig, ax = plt.subplots(2,1, figsize=(16,9))

df_tempLoan = df_master.select("emplength_clean", "int_rate", "loan_amnt", "term_clean", "purpose").toPandas()

plot0 = sns.boxplot(x="emplength_clean", y="loan_amnt",data=df_tempLoan, hue="term_clean",  ax=ax[0])
plot0.set(xlabel='Length of Employment',ylabel='Loan Amount')

plot1 = sns.boxplot(x="emplength_clean", y="int_rate", data=df_tempLoan,ax=ax[1])
plot1.set(xlabel='Length of Employment',ylabel='Interest Rate')

plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC The above plots illustrate that Length of Employment of the loan applicant has limited influence on the Loan Amount or the Interest rate applied to the loan

# COMMAND ----------

# MAGIC %md
# MAGIC ##Observe grade struture of loans issued by the lending club and see how it influences interest rates

# COMMAND ----------

# MAGIC %sql
# MAGIC /* View the different strings in the grade */
# MAGIC select distinct grade from loantempdata order by grade

# COMMAND ----------

# MAGIC %sql
# MAGIC /* View the different strings in the sub_grade */
# MAGIC select distinct sub_grade from loantempdata order by sub_grade

# COMMAND ----------

# MAGIC %md
# MAGIC It's well defined that the Grades are ordered in 7 categories (A-G) and each has 5 sub cateogories eg. A1-A5. Lets look at the loan interest rates attribute to these categories

# COMMAND ----------

#Select a sub df to investigate
df_tempLoan = df_master.select("sub_grade", "grade", "loan_amnt", "int_rate").toPandas()
#Convert to categorical data 
df_tempLoan['grade'] = df_tempLoan['grade'].astype('category')
df_tempLoan['sub_grade'] = df_tempLoan['sub_grade'].astype('category')

subgradeOrder = ["A1", "A2", "A3", "A4", "A5", "B1", "B2", "B3", "B4", "B5", "C1", "C2", "C3", "C4", "C5", "D1", "D2", "D3", "D4", "D5",
                "E1", "E2", "E3", "E4", "E5", "F1", "F2", "F3", "F4", "F5", "G1", "G2", "G3", "G4", "G5"]

#Set the column wrap here to 20 to visuallise the interest increase per sub grade
subgrade = sns.FacetGrid(df_tempLoan, col="sub_grade", sharex=False, col_wrap=20, col_order=subgradeOrder)
subgrade.map(sns.boxplot, 'grade', 'int_rate')

# COMMAND ----------

#Perform the same step but use a col_wrap of 5 to observer the variane in interes from sub group classes e.g A1,B1,C1 etc...
subg = sns.FacetGrid(df_tempLoan, col="sub_grade", sharex=False, col_wrap=5, col_order=subgradeOrder)
subg.map(sns.boxplot, 'grade', 'int_rate')

# COMMAND ----------

# MAGIC %md
# MAGIC It's observable that that the interest rates increase as we go through each sub category e.g A1-A5 and the outliers increase as we increase subcategory
# MAGIC It's also obeservable that the intrest rates increase as we go through each category with the degree of outliers also increasing through Category A-G.
# MAGIC 
# MAGIC It wouldn't be unreasonable to think that the higher the interest rate paid the more likely a customer is to default on a loan.
# MAGIC 
# MAGIC Let's see if there is any relationship between the loan amounts taken and sub grade.

# COMMAND ----------

fig, ax = plt.subplots(1,1, figsize=(16,9))
plt.subplots_adjust(hspace = 0.4, top = 0.8)

sLoanSub = sns.barplot(x="sub_grade", y="loan_amnt", data=df_tempLoan, order = subgradeOrder)
sLoanSub.set(xlabel='Sub Grade',ylabel='Loan Amount')

# COMMAND ----------

# MAGIC %md
# MAGIC The visualisation indicates that in most cases as the Sub Grade increases, so does the loan amount. There appears to be a pattern of increasing interest rates and loan amount as you increase the sub grade.
# MAGIC 
# MAGIC Let' look at what constitutes a bad loan outcome and see if we can observere any further patterns. The field loan_status contains the state of a loan

# COMMAND ----------

# MAGIC %md
# MAGIC ##Observe what constitutes a bad loan outcome 
# MAGIC 
# MAGIC and see if we can observere any further patterns. 
# MAGIC 
# MAGIC The field loan_status contains the state of a loan

# COMMAND ----------

#With that in mind lets look at the status of Loans against associated grade 
tempStatus = df_master.stat.crosstab('loan_status','grade').show()

# COMMAND ----------

# MAGIC %md
# MAGIC We can see the dataset has primarily loans that are fully paid or are current which means are being paid and up to date. As we move to attempting to predict if a customer will default we will need to combine the various categories of being unable to pay a loan. 

# COMMAND ----------

# MAGIC %sql
# MAGIC /* View the different strings in the sub_grade */
# MAGIC select distinct loan_status from loantempdata order by loan_status

# COMMAND ----------

 #Need to clean the balances on some of the fields before we create a single / new category for loan_status, the  Majority of the missing data in "tot_cur_bal", "tot_coll_amt" columns can be set to 0 as their loan status is "Fully Paid" OR "Charged Off"
    
from pyspark.sql.functions import isnan, when, count, col, avg, split,lit
   
df_master = df_master.withColumn("tot_coll_amt", when((col("tot_coll_amt").isNull() & col("loan_status").isin("Fully Paid", "Charged Off", "Does not meet the credit policy. Status:Fully Paid")), lit("0")).otherwise(col("tot_coll_amt")))
df_master = df_master.withColumn("tot_cur_bal", when((col("tot_cur_bal").isNull() & col("loan_status").isin("Fully Paid", "Charged Off")), lit("0")).otherwise(col("tot_cur_bal")))

#Create a Boolean "Class" to use for future models and observe defaulters  (0 no debt , 1 being in debt )
df_master = df_master.withColumn("target", when(col("loan_status").isin("Default","Charged Off", "Late (31-120 days)", "Late (16-30 days)", "Does not meet the credit policy. Status:Charged Off","In Grace Period"), 1).otherwise(0))


# COMMAND ----------

#Lets visualise where the largest number of defaulted loans exist
import plotly.express as px

total_loans_by_state = df_master.groupBy("addr_state").sum('target').toPandas()

fig = px.choropleth(total_loans_by_state,
                    locations='addr_state', 
                    locationmode="USA-states", 
                    scope="usa",
                    color='sum(target)',
                    color_continuous_scale="Viridis_r", 
                    
                    )
fig.show()

# COMMAND ----------

# MAGIC %md
# MAGIC The number of columns are extensive and it's going to be hard to define coreletation between field so we will lool to unserpivsed learning to help. Before we start we need to remove redundant fields we have replaced 

# COMMAND ----------

df_master.printSchema()

# COMMAND ----------

# Dropping further fields, member_id contains no data, zip code replicates data with addr_code and URL is a link to the loan info that provides no further value )
df_master = df_master.drop("emplength")
df_master = df_master.drop("term")
df_master = df_master.drop("class")
df_master = df_master.drop("loan_status")
df_master = df_master.drop("issue_d")
 
# Refresht the temporary table to hold the data for optimised queries etc..can always return to this points
temp_table_name = "loantempdata"
df_master.createOrReplaceTempView(temp_table_name)

# COMMAND ----------

# Refresh temporary table to hold the data for optimised queries via Spark SQL and Temp tables etc..
temp_table_name = "loantempdata"
df_master.createOrReplaceTempView(temp_table_name)

# COMMAND ----------

## Writing cleaned pyspark dataframe to csv file
df_master.write.option("header",True) \
 .csv("/FileStore/tables/step2_0502_2007_to_2018Q4-2201.csv")

# COMMAND ----------

#Preserve this information to produce data copy for local use or to ensure we can return to this stage with minimum effort. locally
df_master.coalesce(1).write.format('com.databricks.spark.csv').option('header', 'true').save('dbfs:/FileStore/backup/tables/step2_0502_2007_to_2018Q4-2201.csv')
 
# How to download to PC
# https://adb-8855045224243626.6.azuredatabricks.net/files/backup/tables/step2_0402_2007_to_2018Q4-2201.csv/part-00000-tid-8010089884215906983-bb4e8ab1-d100-4fad-b574-e0cfa00480ba-202-1-c000.csv?o=8855045224243626
