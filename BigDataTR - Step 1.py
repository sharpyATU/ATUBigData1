# Databricks notebook source
# MAGIC %md
# MAGIC ##Technical Project 1 - Introduction
# MAGIC 
# MAGIC Although the idea of innovation in finance is not new, over the past ten years there has been a wave of new ideas and businesses that have opened access to financial services to a wider range of people. The businesses' inventions have a strong technological foundation in Big data analytics, AI, and blockchain technologies. These have paved the way for new business models, applications, workflows, and goods.  These new companies, products and their sector have been labelled as FinTech (FINancial TECHnology) [Talonen et al, 2017] 
# MAGIC 
# MAGIC Big data analytics are a key component of many fintech applications, especially those based on peer-to-peer (P2P) financial transactions including peer-to-peer lending, crowdfunding, and invoice trading. Peer-to-peer does away with the necessity for a central middleman. With peer-to-peer lending, the fintech company enables direct communication between lenders and borrowers, using the platform as an information source that, among other things, evaluates the credit risk of borrowers. The focus of this technical project is to investigate the data of one on the earliest large scale P2P Lenders “The Lending Club”.  
# MAGIC The lending club business concept offers borrowers a platform to request loans on its website in 39 US States. It also provides investors the ability to choose loans to invest in. By charging investors a "service fee" and borrowers an "origination fee," the lending company generates revenue. The Lending Club interest rates are often better for lenders and borrowers than either would receive from a traditional financial institution, which is the products' unique selling feature.
# MAGIC 
# MAGIC Whilst there are advantages in peer-to-peer lending there are also many risks that are greater than in established financial institutions; a relevant example of this for P2P Lending is the underestimation of creditworthiness, and indeed the resultant risk of losses to the lenders. The general goal for this project is to provide analytics that will give new insight into risk exposure too Lenders off existing Loans in the Lending Club 
# MAGIC 
# MAGIC The core dataset to be used in this project is Lending Club: All Lending Club loan data 2007 through 2018 found in Kaggle , https://www.kaggle.com/datasets/ethon0426/lending-club-20072020q1 

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 1 - Data Sourcing , Ingesting,  Inital Cleaning, Visualisation and Transformation 
# MAGIC The intent of this workbook is to injest the data file, carry out intial exploration to find defects, irregularities, visualise data issues and drop/clean to provide a data file that can be used in a subsequent steps for further investigation to make predictions via Machine Learning

# COMMAND ----------

# Note : One of the known limitations in pandas is that it does not scale with your data volume linearly due to single-machine processing. For example, pandas fails with out-of-memory if it attempts to read a dataset that is larger than the memory available in a single machine.
#pandas API on Spark overcomes the limitation, enabling users to work with large datasets by leveraging Spark:
#https://www.databricks.com/blog/2021/10/04/pandas-api-on-upcoming-apache-spark-3-2.html

# COMMAND ----------

# Import the base libaries expected to be used
import pandas as pd
import numpy as np
import pyspark.pandas as ps
from pyspark.sql import SparkSession
from scipy import stats


# COMMAND ----------

# MAGIC %md
# MAGIC Clean the file as Extended characters such as emojis etc are known to be present in the dataset (this is a once of task)

# COMMAND ----------

#import io
#file_location = "//dbfs/FileStore/tables/accepted_2007_to_2018Q4.csv"
#cleaned_file = "//dbfs/FileStore/tables/cleaned_2007_to_2018Q4.csv"

#with io.open(file_location,'r',encoding='utf-8',errors='ignore') as infile, \
#     io.open(cleaned_file,'w',encoding='ascii',errors='ignore') as outfile:
#    for line in infile:
#        print(*line.split(), file=outfile)

# COMMAND ----------

# MAGIC %md
# MAGIC Load the cleansed file into Spark darta frame for investigation

# COMMAND ----------


# Define the File location and type
file_location = "/FileStore/tables/cleaned_2007_to_2018Q4.csv"
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

#Lets observe how many data rows we have
df_master.count()

# COMMAND ----------

#Lets look at the struture of the dataset
df_master.printSchema()

# COMMAND ----------

# MAGIC %md
# MAGIC There's over 2.25 milltion recordsd in the table with >150 columns, a number of fields look from the sample set and data dictionary to be more likely to be numeric but are stored as string types. 
# MAGIC Lets investigtae further as this is an indication of dirty data.have a look to see what's going on with these fields

# COMMAND ----------

#Run an action command , this will allow loading to be executed, and enable better optimisation and distribution  of the DAG before further analytics
df_master.describe().show()

# COMMAND ----------

# Lets look at some of the key  fields within the dataset
df_master.describe('term','int_rate','loan_amnt', 'emp_title' ,'emp_length','annual_inc', 'addr_state' , 'dti','delinq_2yrs').show()

# COMMAND ----------

# MAGIC %md
# MAGIC Theres bad data littered through the dataset where key metric data fields are being shifted to the right due to data that has been entered via text prompt systems not escaping or cleansing data. Lets create temp view to make it easier to query the data to get an indication of whats needs removed.

# COMMAND ----------

# Create a temporary table to hold the data for optimised queries etc..
temp_table_name = "loantempdata"
df_master.createOrReplaceTempView(temp_table_name)

# COMMAND ----------

# MAGIC %sql
# MAGIC /* dti would be expected to be populated for every customer as its' a essential metric to consider for a loan, lets use this to identify problem rows*/
# MAGIC select * from loantempdata where dti NOT REGEXP '[0-9]+'

# COMMAND ----------

# MAGIC %md
# MAGIC So there's 179 potentially bad pieces of data, due to large row count going to filter these out of the data set.

# COMMAND ----------

from pyspark.sql import functions as F

#df_master = df_master.filter(~df_master.dti.rlike("^[0-9]*$"))
#df.filter(~F.col('col1').like('%#')).show()
df_master = df_master.filter(F.col('dti').rlike('[0-9]+'))

# COMMAND ----------

df_master.describe('term','int_rate','loan_amnt', 'emp_title' ,'emp_length','annual_inc', 'addr_state' , 'dti','delinq_2yrs').show()

# COMMAND ----------

# MAGIC %md
# MAGIC There's still bad data associated on these fields, lets look at another field delinq_2yrs

# COMMAND ----------

# Create a temporary table to hold the data for optimised queries etc..
temp_table_name = "loantempdata"
df_master.createOrReplaceTempView(temp_table_name)

# COMMAND ----------

# MAGIC %sql
# MAGIC select * from loantempdata where delinq_2yrs NOT REGEXP '[0-9]+'

# COMMAND ----------

df_master = df_master.filter(F.col('delinq_2yrs').rlike('[0-9]+'))
df_master.describe('term','int_rate','loan_amnt', 'emp_title' ,'emp_length','annual_inc', 'addr_state' , 'dti','delinq_2yrs').show()

# COMMAND ----------

# The data is still not clean lets look at another field to clean
df_master = df_master.filter(F.col('addr_state').rlike('^[A-Z]{0,3}$'))
df_master.describe('term','int_rate','loan_amnt', 'emp_title' ,'emp_length','annual_inc', 'addr_state' , 'dti','delinq_2yrs').show()


# COMMAND ----------

# MAGIC %md
# MAGIC That was painful. , the data presented looks better, lets analyse for some key missing metics (missing data, icompatible types and replicated data) with the intent of creating a cleaner file for future processing. 

# COMMAND ----------

import plotly.plotly as py
import plotly.graph_objs as go
from plotly.offline import init_notebook_mode,iplot
init_notebook_mode(connected=True)
!pip install -U pandas-profiling[notebook]



# COMMAND ----------

# MAGIC %md
# MAGIC Create a CSV file from the data that is lighter (aprox 10% of data). will use this for early development ...  to accelerated understanding of data and how to parse and not wait on repsonses

# COMMAND ----------

##Temporary space to analyse smaller section of data -- 
#df_master = spark.sql("select * from `accepted_2007_to_2018Q4_csv` LIMIT 200000")
## Writing pyspark dataframe to csv file
#df_master.write.option("header",True) \
# .csv("/FileStore/tables/truncated_2007_to_2018Q4-2201.csv")


# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC **We have >2.25 million data points, the Data has significant allocation of sting types, also from first observations a lot of cleaning will be required, lets delve a bit deeper into the data using a Profile Report to understand the data where fields have a large degree of missing data, so along with review of the fields from the data dictionary from the project these may be removed.  

# COMMAND ----------

#Convert the spark data frame to enable intial analysis for field deletion
result_tmp_pd = df_master.select("*").toPandas()

#Utilse Y-data profiling to quickly undertand a high number of data fields to drop   :https://ydata-profiling.ydata.ai/docs/master/index.html
from pandas_profiling import ProfileReport
profile = ProfileReport(result_tmp_pd,progress_bar=True, minimal=True, title = 'Loans Defaults Prediction', html = {'style': {'full_width': True }})
profile



# COMMAND ----------

# MAGIC %md
# MAGIC The following fields have a high proportion of data missing, guidance is to consider removing columns > 5-10% data missing that have appear to have limited business value.
# MAGIC Data assesed through ditionary for removal
# MAGIC 
# MAGIC member_id has 2260701 (100.0%) missing values	Missing
# MAGIC 
# MAGIC desc has 2134633 (94.4%) missing values	Missing
# MAGIC 
# MAGIC mths_since_last_delinq has 1158374 (51.2%) missing values	Missing
# MAGIC 
# MAGIC mths_since_last_record has 1901354 (84.1%) missing values	Missing
# MAGIC 
# MAGIC next_pymnt_d has 1345119 (59.5%) missing values	Missing
# MAGIC 
# MAGIC mths_since_last_major_derog has 1679686 (74.3%) missing values	Missing
# MAGIC 
# MAGIC annual_inc_joint has 2139788 (94.7%) missing values	Missing
# MAGIC 
# MAGIC dti_joint has 2139829 (94.7%) missing values	Missing
# MAGIC 
# MAGIC verification_status_joint has 2144839 (94.9%) missing values	Missing
# MAGIC 
# MAGIC open_acc_6m has 866068 (38.3%) missing values	Missing
# MAGIC 
# MAGIC open_act_il has 866092 (38.3%) missing values	Missing
# MAGIC 
# MAGIC open_il_12m has 866117 (38.3%) missing values	Missing
# MAGIC 
# MAGIC open_il_24m has 866124 (38.3%) missing values	Missing
# MAGIC 
# MAGIC mths_since_rcnt_il has 909923 (40.2%) missing values	Missing
# MAGIC 
# MAGIC total_bal_il has 866133 (38.3%) missing values	Missing
# MAGIC 
# MAGIC il_util has 1068859 (47.3%) missing values	Missing
# MAGIC 
# MAGIC open_rv_12m has 866146 (38.3%) missing values	Missing
# MAGIC 
# MAGIC open_rv_24m has 866142 (38.3%) missing values	Missing
# MAGIC 
# MAGIC max_bal_bc has 866150 (38.3%) missing values	Missing
# MAGIC 
# MAGIC all_util has 866374 (38.3%) missing values	Missing
# MAGIC 
# MAGIC inq_fi has 866155 (38.3%) missing values	Missing
# MAGIC 
# MAGIC total_cu_tl has 866153 (38.3%) missing values	Missing
# MAGIC 
# MAGIC inq_last_12m has 866161 (38.3%) missing values	Missing
# MAGIC 
# MAGIC mths_since_recent_bc_dlq has 1740974 (77.0%) missing values	Missing
# MAGIC 
# MAGIC mths_since_recent_inq has 295446 (13.1%) missing values	Missing
# MAGIC 
# MAGIC mths_since_recent_revol_delinq has 1520323 (67.3%) missing values	Missing
# MAGIC 
# MAGIC revol_bal_joint has 2152654 (95.2%) missing values	Missing
# MAGIC 
# MAGIC sec_app_fico_range_low has 2152660 (95.2%) missing values	Missing
# MAGIC 
# MAGIC sec_app_fico_range_high has 2152655 (95.2%) missing values	Missing
# MAGIC 
# MAGIC sec_app_earliest_cr_line has 2152657 (95.2%) missing values	Missing
# MAGIC 
# MAGIC sec_app_inq_last_6mths has 2152665 (95.2%) missing values	Missing
# MAGIC 
# MAGIC sec_app_mort_acc has 2152667 (95.2%) missing values	Missing
# MAGIC 
# MAGIC sec_app_open_acc has 2152671 (95.2%) missing values	Missing
# MAGIC 
# MAGIC sec_app_revol_util has 2154514 (95.3%) missing values	Missing
# MAGIC 
# MAGIC sec_app_open_act_il has 2152678 (95.2%) missing values	Missing
# MAGIC 
# MAGIC sec_app_num_rev_accts has 2152678 (95.2%) missing values	Missing
# MAGIC 
# MAGIC sec_app_chargeoff_within_12_mths has 2152673 (95.2%) missing values	Missing
# MAGIC 
# MAGIC sec_app_collections_12_mths_ex_med has 2152673 (95.2%) missing values	Missing
# MAGIC 
# MAGIC sec_app_mths_since_last_major_derog has 2224757 (98.4%) missing values	Missing
# MAGIC 
# MAGIC hardship_type has 2249725 (99.5%) missing values	Missing
# MAGIC 
# MAGIC hardship_reason has 2249738 (99.5%) missing values	Missing
# MAGIC 
# MAGIC hardship_status has 2249744 (99.5%) missing values	Missing
# MAGIC 
# MAGIC deferral_term has 2249760 (99.5%) missing values	Missing
# MAGIC 
# MAGIC hardship_amount has 2249766 (99.5%) missing values	Missing
# MAGIC 
# MAGIC hardship_start_date has 2249775 (99.5%) missing values	Missing
# MAGIC 
# MAGIC hardship_end_date has 2249772 (99.5%) missing values	Missing
# MAGIC 
# MAGIC payment_plan_start_date has 2249771 (99.5%) missing values	Missing
# MAGIC 
# MAGIC hardship_length has 2249774 (99.5%) missing values	Missing
# MAGIC 
# MAGIC hardship_dpd has 2249777 (99.5%) missing values	Missing
# MAGIC 
# MAGIC hardship_loan_status has 2249777 (99.5%) missing values	Missing
# MAGIC 
# MAGIC orig_projected_additional_accrued_interest has 2252048 (99.6%) missing values	Missing
# MAGIC 
# MAGIC hardship_payoff_balance_amount has 2249783 (99.5%) missing values	Missing
# MAGIC 
# MAGIC hardship_last_payment_amount has 2249783 (99.5%) missing values	Missing
# MAGIC 
# MAGIC debt_settlement_flag_date has 2226353 (98.5%) missing values	Missing
# MAGIC 
# MAGIC settlement_status has 2226370 (98.5%) missing values	Missing
# MAGIC 
# MAGIC settlement_date has 2226393 (98.5%) missing values	Missing
# MAGIC 
# MAGIC settlement_amount has 2226417 (98.5%) missing values	Missing
# MAGIC 
# MAGIC settlement_percentage has 2226431 (98.5%) missing values	Missing
# MAGIC 
# MAGIC settlement_term has 2226435 (98.5%) missing values	Missing

# COMMAND ----------

# MAGIC %md
# MAGIC Before we are too hasty and chop the data. lets look at the key field that demonstrates if someone is in debt or not "Loan_status". 

# COMMAND ----------

# MAGIC %sql
# MAGIC /* View the different groupings in loan_status */
# MAGIC select distinct loan_status from loantempdata 

# COMMAND ----------

# MAGIC %md
# MAGIC The fields Im particualtrly interested in are whether the hardship and settlement fields to the RHS that are spartially populated contain quality data for the charged off 

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT *
# MAGIC FROM loantempdata
# MAGIC WHERE loan_status = 'Charged Off';

# COMMAND ----------

# MAGIC %md
# MAGIC The data is appears sparsely populated even when charged off. lets check the proportion empty

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT COUNT(loan_status)
# MAGIC FROM loantempdata
# MAGIC WHERE loan_status = 'Charged Off'; 

# COMMAND ----------

# MAGIC %sql
# MAGIC /* View the different groupings in loan_status */
# MAGIC SELECT COUNT(loan_status)
# MAGIC FROM loantempdata
# MAGIC WHERE loan_status = 'Charged Off' and settlement_status  IS NULL

# COMMAND ----------

# MAGIC %md
# MAGIC There is only data present in the settlement fields for aprox 12% of the charged of caetgory. we can delete this too.

# COMMAND ----------

#Removing all the features which less than 90% complete data

lowDataColList = ['desc','mths_since_last_delinq','mths_since_last_record','next_pymnt_d','mths_since_last_major_derog','annual_inc_joint','dti_joint','verification_status_joint'
,'open_acc_6m','open_act_il','open_il_12m','open_il_24m','mths_since_rcnt_il','total_bal_il','il_util has','open_rv_12m','open_rv_24m','max_bal_bc','all_util','inq_fi','total_cu_tl'
,'inq_last_12m','mths_since_recent_bc_dlq','mths_since_recent_inq','mths_since_recent_revol_delinq','revol_bal_joint','sec_app_fico_range_low','sec_app_fico_range_high','sec_app_earliest_cr_line'
,'sec_app_inq_last_6mths','sec_app_mort_acc','sec_app_open_acc','sec_app_revol_util','sec_app_open_act_il','sec_app_num_rev_accts','sec_app_chargeoff_within_12_mths','sec_app_collections_12_mths_ex_med'
,'sec_app_mths_since_last_major_derog','hardship_type','hardship_reason','hardship_status','deferral_term','hardship_amount','hardship_start_date','hardship_end_date','payment_plan_start_date'
,'hardship_length','hardship_dpd','hardship_loan_status','orig_projected_additional_accrued_interest','hardship_payoff_balance_amount','hardship_last_payment_amount','debt_settlement_flag_date'
,'settlement_status','settlement_date','settlement_amount','settlement_percentage','settlement_term' ]

df_master = df_master.drop(*lowDataColList)




# COMMAND ----------

# Dropping further fields, member_id contains no data, zip code replicates data with addr_code and URL is a link to the loan info that provides no further value )
df_master = df_master.drop("member_id")
df_master = df_master.drop("zip_code")
df_master = df_master.drop("url")

# Refresht the temporary table to hold the data for optimised queries etc..can always return to this points
temp_table_name = "loantempdata"
df_master.createOrReplaceTempView(temp_table_name)

# COMMAND ----------

#Lets observe how many data rows we have
df_master.count()

# COMMAND ----------

# MAGIC %md
# MAGIC I have reduced the data set from 2,260,701 to 2,258,672 rows, removing incorrelctly formated date that would have prevented casting back to intended numeric types
# MAGIC 
# MAGIC I have also reduced the width of the data set from 149 colums to 89 due to redundant or ducplicate information, this will improve performance in rest of the analysis
# MAGIC 
# MAGIC Lets write the CSV file out to the DBFS to preserve a cleaned file in our pipeline to be used for next steps.

# COMMAND ----------

## Writing cleaned pyspark dataframe to csv file
df_master.write.option("header",True) \
 .csv("/FileStore/tables/step2_0402_2007_to_2018Q4-2201.csv")


# COMMAND ----------

#Preserve this information to produce data copy for local use or to ensure we can return to this stage with minimum effort. locally
df_master.coalesce(1).write.format('com.databricks.spark.csv').option('header', 'true').save('dbfs:/FileStore/backup/tables/step2_0402_2007_to_2018Q4-2201.csv')

# How to download to PC
# https://adb-8855045224243626.6.azuredatabricks.net/files/backup/tables/step2_0402_2007_to_2018Q4-2201.csv/part-00000-tid-8010089884215906983-bb4e8ab1-d100-4fad-b574-e0cfa00480ba-202-1-c000.csv?o=8855045224243626
