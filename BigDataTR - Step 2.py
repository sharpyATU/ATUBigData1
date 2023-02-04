# Databricks notebook source
# MAGIC %md
# MAGIC 
# MAGIC ## Overview
# MAGIC 
# MAGIC 
# MAGIC 
# MAGIC This notebook will show you how to create and query a table or DataFrame that you uploaded to DBFS. [DBFS](https://docs.databricks.com/user-guide/dbfs-databricks-file-system.html) is a Databricks File System that allows you to store data for querying inside of Databricks. This notebook assumes that you have a file already inside of DBFS that you would like to read from.
# MAGIC 
# MAGIC This notebook is written in **Python** so the default cell type is Python. However, you can use different languages by using the `%LANGUAGE` syntax. Python, Scala, SQL, and R are all supported.

# COMMAND ----------

#One of the known limitations in pandas is that it does not scale with your data volume linearly due to single-machine processing. For example, pandas fails with out-of-memory if it attempts to read a dataset that is larger than the memory available in a single machine.
#pandas API on Spark overcomes the limitation, enabling users to work with large datasets by leveraging Spark:
#https://www.databricks.com/blog/2021/10/04/pandas-api-on-upcoming-apache-spark-3-2.html

# COMMAND ----------

import pandas as pd
import numpy as np
import pyspark.pandas as ps

import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(color_codes=True)
from scipy import stats

import plotly.plotly as py

import plotly.graph_objs as go
from plotly.offline import init_notebook_mode,iplot
init_notebook_mode(connected=True)


!pip install -U pandas-profiling[notebook]



# COMMAND ----------

## Writing cleaned pyspark dataframe to csv file
#df_master.write.option("header",True) \
# .csv("/FileStore/tables/step2_0202_2007_to_2018Q4-2201.csv")
## Define the File (cleaned) 

file_location = "/FileStore/tables/step2_0202_2007_to_2018Q4-2201.csv"
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

#Preserve this information to produce data copies at stages locally
# df_master.coalesce(1).write.format('com.databricks.spark.csv').option('header', 'true').save('dbfs:/FileStore/backup/tables/step2_0202_2007_to_2018Q4-2201.csv')
# How to download to PC
# https://adb-8855045224243626.6.azuredatabricks.net/files/backup/tables/step2_0202_2007_to_2018Q4-2201.csv/part-00000-tid-8010089884215906983-bb4e8ab1-d100-4fad-b574-e0cfa00480ba-202-1-c000.csv?o=8855045224243626


# COMMAND ----------

df_master.coalesce(1).write.format('com.databricks.spark.csv').option('header', 'true').save('dbfs:/FileStore/backup/tables/step2_0202_2007_to_2018Q4-2201.csv')

# COMMAND ----------

# Create a temporary table to hold the data for optimised queries etc..
temp_table_name = "loantempdata"
df_master.createOrReplaceTempView(temp_table_name)
## ANDY YOU ARE HERE. FOLLOW THIS APPROACH

# COMMAND ----------

# Can see from quickly looking at the data theres a few fields of significant interest that will likely need cleaned
#The following fields are numerical but are present in the data as strings
# term -> has months with textual data need to convert from sting to numerical type
# emp_length - > has years textual informatiomn , need to convert from sting to numerical type
# int-rate -> is stored as a string but is a numerical value
# dti is stored as a string (would expect it to be a key value), observered a few empty values so maybe further cleaning is required

#emp_title is mixed case. with repeats of the same word but spelt differently. if it is of use for categorical/hot encoding would need converting.

#loan_status --> this will be the result we are loooking to predict... but its' messy data

#daPurpose and title are the same
#grade may require encoding
#sub grade may require encoding

#********* ANDY THIS IS NOT REQUIRED. DOING THE LITE BIT NEXT WOULD BE BETTER ** 
#df_cleaning = df_master.select("term","int_rate","loan_amnt", "grade", "sub_grade", "emp_title","emp_length","home_ownership","annual_inc", "loan_status", "desc","purpose","title","addr_state","dti","delinq_2yrs")
# Create a lighter width data frame to investigate /parse with

#df_cleaning.describe().show()

# COMMAND ----------

# MAGIC %md
# MAGIC Lets Explore the data further, I will perform some minor visualisations on data fields that give us an understanding of some key features of the data.
# MAGIC It is quicker and less burdensome to use the pandas profileing tools, so will re-execute the profiling in step 1

# COMMAND ----------

# MAGIC %sql
# MAGIC /* View the different strings in the emp_length */

# COMMAND ----------

##dftemp= df_master.select().describe().toPandas()#('term','int_rate','loan_amnt', 'emp_length','annual_inc', 'dti','delinq_2yrs').describe().toPandas()
#Data Profile was still quite messy to view (Pandas for spark would be useful here.)
df_master.describe('term','int_rate','loan_amnt', 'emp_length','annual_inc', 'dti','delinq_2yrs').show()

# COMMAND ----------

# MAGIC %sql
# MAGIC /* View the different strings in the emp_length */
# MAGIC select distinct emp_length from loantempdata limit 1000

# COMMAND ----------

# MAGIC %md
# MAGIC Clean the employee length to integers using regular expresssions

# COMMAND ----------

#Evalutate how to clean the emp_length string
from pyspark.sql.functions import  regexp_replace, regexp_extract
from pyspark.sql.functions import col
regex_filterstring = 'years|year|\\+|\\<'
df_master.select(regexp_replace(col("emp_length"), regex_filterstring, "").alias("emplength_clean"), col("emp_length")).show(15)

# COMMAND ----------

# MAGIC %sql
# MAGIC select distinct term from loantempdata limit 1000

# COMMAND ----------

# Evalute Getting rid of the Months string
from pyspark.sql.functions import  regexp_replace, regexp_extract
from pyspark.sql.functions import col
#Extract digits data
regex_filterstring = '\\d+'  #Alternative filter
df_master.select(regexp_extract(col("term"), regex_filterstring, 0).alias("term_clean"), col("term")).show(10)

# COMMAND ----------

## need to get these into dataset and keep clean further
df_master=df_master.withColumn("term_clean",regexp_extract(col("term"), '\\d+', 0)).withColumn('emplength_clean', regexp_replace(col("emp_length"), 'years|year|\\+|\\<', ""))

 
##Andy you are here, next step would be to create a refreshed dataset, need to cast the objects back, next cell error indicates we may have some further rouge data.


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

Objective after updating the data fields would be 

# COMMAND ----------

df_master.stat.crosstab('loan_status','grade').show()

# COMMAND ----------

loanRows = df.count()

# find list of columns which has more than 50% of data missing.
def findMissingValueCols(df):
    # df.select([count(when(isnan(c) | col(c).isNull(), c)).alias(c) for c in df.columns]).show()
    missingValueColumns = []
    for column in df.columns:
        nullRows = df.where(col(column).isNull()).count()
        print(column, "--", nullRows)
        if nullRows > loanRows * 0.5:  # i.e. if ALL values are NULL
            missingValueColumns.append(column)
    return missingValueColumns


# columns names which has more than 50% data missing
missingValueColList = findMissingValueCols(df)

print(missingValueColList)

# COMMAND ----------



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

# COMMAND ----------

# MAGIC %sql
# MAGIC 
# MAGIC /* Query the created temp table in a SQL cell */
# MAGIC 
# MAGIC /*select * from `accepted_2007_to_2018Q4_csv`*/

# COMMAND ----------

# With this registered as a temp view, it will only be available to this particular notebook. If you'd like other users to be able to query this table, you can also create a table from the DataFrame.
# Once saved, this table will persist across cluster restarts as well as allow various users across different notebooks to query this data.
# To do so, choose your table name and uncomment the bottom line.

#permanent_table_name = "accepted_2007_to_2018Q4_csv"

# df.write.format("parquet").saveAsTable(permanent_table_name)
