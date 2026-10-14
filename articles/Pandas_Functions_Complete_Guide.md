# Pandas Functions from Scratch: Complete Guide with Real-Time Examples <!-- omit in toc -->

Pandas is a Python library used for data manipulation, data cleaning,
analysis, and reporting. It is widely used in Data Science, Machine
Learning, Data Analytics, and ETL pipelines.

This guide organizes useful Pandas functions from beginner to advanced
level. Each function includes a one-line explanation and a real-world
example.

## Table of Contents <!-- omit in toc -->

- [1. Importing Pandas and Creating Data](#1-importing-pandas-and-creating-data)
- [2. Reading and Writing Files](#2-reading-and-writing-files)
- [3. Exploring Datasets](#3-exploring-datasets)
- [4. Selecting and Filtering Data](#4-selecting-and-filtering-data)
- [5. Cleaning Missing and Duplicate Data](#5-cleaning-missing-and-duplicate-data)
- [6. Adding and Modifying Columns](#6-adding-and-modifying-columns)
- [7. Sorting and Ranking](#7-sorting-and-ranking)
- [8. Aggregation and Grouping](#8-aggregation-and-grouping)
- [9. Combining Datasets](#9-combining-datasets)
- [10. String Operations](#10-string-operations)
- [11. Date and Time Operations](#11-date-and-time-operations)
- [12. Statistical Functions](#12-statistical-functions)
- [13. Reshaping Data](#13-reshaping-data)
- [14. Applying Custom Functions](#14-applying-custom-functions)
- [15. Index Management and Advanced Operations](#15-index-management-and-advanced-operations)
- [16. End-to-End Real-Time Example: Data Annotation EDA](#16-end-to-end-real-time-example-data-annotation-eda)
- [17. Quick-Reference Cheat Sheet](#17-quick-reference-cheat-sheet)
- [18. Recommended Learning Order](#18-recommended-learning-order)
- [Final Advice](#final-advice)

## 1. Importing Pandas and Creating Data

Start by importing Pandas and creating a sample business dataset.

``` python
import pandas as pd
import numpy as np

df = pd.DataFrame({
    "Employee": ["Ravi", "Priya", "Amit", "Ravi"],
    "Department": ["IT", "HR", "IT", "IT"],
    "Salary": [60000, 50000, 70000, 60000],
    "Experience": [3, 5, 7, 3]
})
```

**Real-world scenario:** An HR analyst creates a table containing
employee names, departments, salaries, and experience.

| Employee | Department | Salary | Experience |
| --- | --- | ---: | ---: |
| Ravi | IT | 60000 | 3 |
| Priya | HR | 50000 | 5 |
| Amit | IT | 70000 | 7 |
| Ravi | IT | 60000 | 3 |

## 2. Reading and Writing Files

These functions load data from different sources and save processed
results.

| Function | One-line explanation | Real-time example |
| --- | --- | --- |
| `pd.read_csv()` | Reads data from a CSV file. | Load customer transactions. |
| `pd.read_excel()` | Reads data from Excel. | Import monthly sales reports. |
| `pd.read_json()` | Reads data stored in JSON format. | Load API response data. |
| `pd.read_sql()` | Reads query results from a database. | Extract orders from MySQL. |
| `pd.read_parquet()` | Reads column-oriented Parquet files. | Load large analytics datasets. |
| `pd.read_html()` | Extracts tables from HTML pages. | Import a published HTML table. |
| `pd.read_xml()` | Reads XML data into a DataFrame. | Process XML product feeds. |
| `pd.read_fwf()` | Reads fixed-width formatted text files. | Import legacy banking records. |
| `df.to_csv()` | Exports a DataFrame to CSV. | Save cleaned customer data. |
| `df.to_excel()` | Exports a DataFrame to Excel. | Deliver a report to management. |
| `df.to_json()` | Converts a DataFrame to JSON. | Prepare records for an application. |
| `df.to_sql()` | Writes DataFrame records to a SQL table. | Store processed orders in a database. |
| `df.to_parquet()` | Saves data in Parquet format. | Store an efficient analytics dataset. |
| `df.to_dict()` | Converts data into Python dictionaries. | Pass customer records to application code. |
| `df.to_records()` | Converts rows into a NumPy record array. | Transfer tabular data to NumPy workflows. |

``` python
df = pd.read_csv("sales.csv")
df.to_csv("cleaned_sales.csv", index=False)
```

## 3. Exploring Datasets

Use these functions to understand a dataset's structure, size, and
quality.

| Function | One-line explanation | Real-time example |
| --- | --- | --- |
| `df.head()` | Displays the first five rows by default. | Preview imported sales records. |
| `df.tail()` | Displays the last five rows by default. | Inspect the latest appended records. |
| `df.sample()` | Returns random rows. | Review a sample of annotated URLs. |
| `df.info()` | Displays column types and non-null counts. | Identify columns containing missing values. |
| `df.describe()` | Summarizes numeric columns statistically. | Understand salary distributions. |
| `df.shape` | Returns the number of rows and columns. | Count URLs and fields in a dataset. |
| `df.columns` | Returns column names. | Verify imported column headers. |
| `df.dtypes` | Displays each column's data type. | Check whether dates were imported as strings. |
| `df.index` | Returns the row index. | Inspect row labels after filtering. |
| `df.size` | Returns the total number of cells. | Calculate the size of a table. |
| `df.ndim` | Returns the number of dimensions. | Confirm that a DataFrame is two-dimensional. |
| `df.memory_usage()` | Reports memory usage by column. | Find memory-heavy columns in a large dataset. |
| `df.nunique()` | Counts distinct values per column. | Count unique languages or categories. |
| `df.value_counts()` | Counts occurrences of each value in a Series. | Count completed and pending tasks. |
| `df.dtypes.value_counts()` | Counts columns by data type. | Audit a dataset's schema. |

``` python
print(df.head())
print(df.shape)
print(df.info())
print(df["Department"].value_counts())
```

## 4. Selecting and Filtering Data

| Function or syntax | One-line explanation | Real-time example |
| --- | --- | --- |
| `df["Salary"]` | Selects a single column as a Series. | Retrieve employee salaries. |
| `df[["Employee", "Salary"]]` | Selects multiple columns as a DataFrame. | Create an employee salary report. |
| `df.loc[]` | Selects rows and columns by labels or Boolean conditions. | Retrieve employees from IT. |
| `df.iloc[]` | Selects rows and columns by integer position. | Extract the first 10 records. |
| `df.at[]` | Gets or sets one value using row and column labels. | Read an employee's salary by index. |
| `df.iat[]` | Gets or sets one value using integer positions. | Update a particular cell by position. |
| `df.filter()` | Selects rows or columns by labels or patterns. | Select columns containing `Sales` in their names. |
| `df.query()` | Filters rows using a string expression. | Find employees earning over 60,000. |
| `df.where()` | Keeps values meeting a condition and replaces others. | Mask invalid sales values. |
| `df.mask()` | Replaces values where a condition is true. | Mask salaries below a validation threshold. |
| `df.isin()` | Checks whether values belong to a specified set. | Find orders from selected regions. |
| `df.between()` | Checks whether values fall within a range (Series method). | Find products priced between ₹500 and ₹2,000. |
| `df.nlargest()` | Returns rows with the largest values. | Identify the five highest-paid employees. |
| `df.nsmallest()` | Returns rows with the smallest values. | Find the five lowest-priced products. |

``` python
df[df["Salary"] > 60000]
df[["Employee", "Salary"]]
df.query("Salary >= 60000 and Experience >= 3")
df.nlargest(2, "Salary")
```

## 5. Cleaning Missing and Duplicate Data

| Function | One-line explanation | Real-time example |
| --- | --- | --- |
| `df.isna()` | Identifies missing values. | Find orders without customer IDs. |
| `df.isnull()` | Alias for checking missing values. | Detect blank fields in annotations. |
| `df.notna()` | Identifies values that are not missing. | Find records with available email addresses. |
| `df.notnull()` | Alias for checking non-missing values. | Select rows with populated dates. |
| `df.dropna()` | Removes rows or columns containing missing values according to the chosen rule. | Remove records missing a mandatory identifier. |
| `df.fillna()` | Replaces missing values with specified values. | Fill missing quantities with zero when appropriate. |
| `df.ffill()` | Fills missing values using the preceding valid value. | Carry forward a status within a time series. |
| `df.bfill()` | Fills missing values using the following valid value. | Fill a missing reading using the next reading. |
| `df.drop_duplicates()` | Removes duplicate records. | Remove duplicate URLs. |
| `df.duplicated()` | Identifies duplicate rows or key combinations. | Flag repeated transaction IDs. |
| `df.replace()` | Replaces specified values with alternatives. | Convert `"N/A"` text into missing values. |
| `df.isna().sum()` | Counts missing values in each column. | Measure data quality by field. |
| `df.interpolate()` | Estimates missing values between observations. | Fill gaps in sensor measurements. |
| `df.convert_dtypes()` | Converts columns to more suitable Pandas data types. | Use nullable integer types for incomplete columns. |
| `df.astype()` | Converts a column or DataFrame to specified data types. | Convert a year column from text to integer. |
| `df.rename()` | Renames rows or columns. | Change `Emp_Name` to `Employee`. |
| `df.drop()` | Removes specified rows or columns. | Remove an unnecessary temporary column. |
| `df.select_dtypes()` | Selects columns matching specified data types. | Select numeric features for modeling. |

``` python
df.isna().sum()
df = df.drop_duplicates()
df["Salary"] = df["Salary"].fillna(df["Salary"].median())
df = df.rename(columns={"Salary": "Monthly_Salary"})
```

**Important:** Do not automatically fill every missing value with zero.
Choose a strategy based on what the missing value means in the business
context.

## 6. Adding and Modifying Columns

| Function or syntax | One-line explanation | Real-time example |
| --- | --- | --- |
| `df.assign()` | Creates new columns and returns a DataFrame. | Add a calculated tax column. |
| `df.insert()` | Inserts a column at a specified position. | Add an employee ID before the name column. |
| `df.pop()` | Removes a column and returns its values. | Extract a temporary processing column. |
| `df.eval()` | Evaluates expressions involving DataFrame columns. | Calculate revenue minus cost. |
| `df.apply()` | Applies a function along rows or columns. | Calculate a custom employee bonus. |
| `Series.map()` | Maps Series values using a function or mapping. | Convert department codes to names. |
| `DataFrame.map()` | Applies a function element-wise to a DataFrame (Pandas 2.1+). | Transform every cell in a small table. |
| `df.clip()` | Restricts values to specified lower and upper limits. | Cap extreme sensor readings. |
| `df.round()` | Rounds numeric values. | Round financial metrics to two decimal places. |
| `df.abs()` | Returns absolute values. | Calculate absolute differences from a target. |
| `df.diff()` | Calculates differences between adjacent rows. | Compare today's sales with yesterday's. |
| `df.pct_change()` | Calculates fractional change between observations. | Measure period-over-period sales growth. |
| `df.cumsum()` | Calculates a cumulative sum. | Track running monthly revenue. |
| `df.cummax()` | Calculates a cumulative maximum. | Track the highest sales reached so far. |
| `df.cummin()` | Calculates a cumulative minimum. | Track the lowest price recorded so far. |

``` python
df = df.assign(Annual_Salary=df["Salary"] * 12)

department_map = {"IT": "Technology", "HR": "Human Resources"}
df["Department_Name"] = df["Department"].map(department_map)

df["Bonus"] = df["Salary"] * 0.10
```

## 7. Sorting and Ranking

| Function | One-line explanation | Real-time example |
| --- | --- | --- |
| `df.sort_values()` | Sorts rows by one or more columns. | Rank orders by revenue. |
| `df.sort_index()` | Sorts rows or columns by index labels. | Restore chronological index order. |
| `df.rank()` | Assigns ranks to values. | Rank sales representatives by revenue. |
| `df.nlargest()` | Selects rows with the highest values. | Find top-selling products. |
| `df.nsmallest()` | Selects rows with the lowest values. | Find least expensive suppliers. |
| `Series.argsort()` | Returns indices that would sort a Series' values. | Determine the sorted order of readings. |

``` python
df.sort_values(by="Salary", ascending=False)
df["Salary_Rank"] = df["Salary"].rank(ascending=False, method="dense")
```

## 8. Aggregation and Grouping

| Function | One-line explanation | Real-time example |
| --- | --- | --- |
| `df.groupby()` | Groups records by categories for analysis. | Calculate salary by department. |
| `df.agg()` | Applies one or more aggregation functions. | Calculate total, average, and maximum sales. |
| `df.aggregate()` | Alias for `agg()`. | Generate multiple business metrics. |
| `df.sum()` | Calculates totals. | Calculate total sales revenue. |
| `df.mean()` | Calculates the arithmetic average. | Find average customer spending. |
| `df.median()` | Calculates the middle value. | Find the median transaction amount. |
| `df.mode()` | Returns the most frequent value or values. | Find the most common product category. |
| `df.min()` | Returns minimum values. | Find the lowest order value. |
| `df.max()` | Returns maximum values. | Find the highest order value. |
| `df.count()` | Counts non-missing values per column. | Count completed records per column. |
| `GroupBy.size()` | Counts rows in each group, including rows with missing values in other columns. | Count records per department. |
| `df.nunique()` | Counts unique values. | Count distinct customers. |
| `df.std()` | Calculates standard deviation. | Measure variation in delivery times. |
| `df.var()` | Calculates variance. | Analyze variability in product prices. |
| `df.quantile()` | Returns specified quantiles. | Calculate the 90th percentile of delivery times. |
| `df.corr()` | Calculates pairwise correlations among numeric columns. | Explore relationships between advertising and sales. |
| `df.cov()` | Calculates pairwise covariance. | Analyze how financial metrics vary together. |
| `df.prod()` | Multiplies values together. | Calculate combined growth factors. |
| `df.sem()` | Calculates the standard error of the mean. | Estimate uncertainty around average delivery time. |
| `df.skew()` | Measures distribution asymmetry. | Check whether transaction amounts are right-skewed. |
| `df.kurt()` | Measures distribution tail weight and peakedness. | Investigate extreme financial transactions. |
| `GroupBy.cumcount()` | Numbers rows within each group. | Number each customer's purchases. |
| `df.transform()` | Returns transformed values aligned to the original rows. | Add each employee's department-average salary. |
| `Series.value_counts()` | Counts unique values in a Series. | Count records for each status. |

``` python
df.groupby("Department")["Salary"].mean()

df.groupby("Department").agg(
    Average_Salary=("Salary", "mean"),
    Maximum_Salary=("Salary", "max"),
    Employee_Count=("Employee", "count")
)
```

## 9. Combining Datasets

| Function | One-line explanation | Real-time example |
| --- | --- | --- |
| `pd.concat()` | Stacks or combines objects along an axis. | Combine January and February sales files. |
| `pd.merge()` | Joins DataFrames using common columns or indexes. | Combine customers and orders using `Customer_ID`. |
| `df.join()` | Joins DataFrames primarily using their indexes. | Add department details to employee records. |
| `df.combine_first()` | Fills missing values in one DataFrame using aligned values from another. | Combine primary and backup customer records. |
| `df.update()` | Updates matching values using another DataFrame. | Apply corrected product information. |
| `df.compare()` | Shows differences between similarly shaped DataFrames. | Compare two versions of a report. |
| `df.align()` | Aligns two objects by row and column labels. | Align financial reports before comparison. |

``` python
combined = pd.concat([january_df, february_df], ignore_index=True)

orders = pd.merge(
    customers,
    transactions,
    on="Customer_ID",
    how="left"
)
```

Common join types:

-   `inner`: only matching records.
-   `left`: all records from the left table and matching records from
    the right.
-   `right`: all records from the right table and matching records from
    the left.
-   `outer`: all records from both tables.

## 10. String Operations

Pandas provides the `.str` accessor for processing text columns.

| Function | One-line explanation | Real-time example |
| --- | --- | --- |
| `.str.lower()` | Converts text to lowercase. | Standardize email addresses. |
| `.str.upper()` | Converts text to uppercase. | Standardize country codes. |
| `.str.title()` | Capitalizes the first letter of each word. | Format customer names. |
| `.str.capitalize()` | Capitalizes the first character and lowercases the rest. | Normalize a category label. |
| `.str.strip()` | Removes leading and trailing whitespace. | Clean imported names. |
| `.str.lstrip()` | Removes leading whitespace. | Clean spaces before product codes. |
| `.str.rstrip()` | Removes trailing whitespace. | Clean spaces after URLs. |
| `.str.replace()` | Replaces matching text or patterns. | Remove unwanted symbols from IDs. |
| `.str.contains()` | Checks whether text contains a substring or pattern. | Find descriptions mentioning "refund". |
| `.str.startswith()` | Checks whether strings begin with specified text. | Identify URLs beginning with HTTPS. |
| `.str.endswith()` | Checks whether strings end with specified text. | Find files ending in `.pdf`. |
| `.str.split()` | Splits strings into parts. | Separate first and last names. |
| `.str.rsplit()` | Splits strings from the right. | Extract a file extension. |
| `.str.extract()` | Extracts groups matching a regular expression. | Extract invoice numbers from text. |
| `.str.findall()` | Finds all regular-expression matches. | Extract reference codes from descriptions. |
| `.str.len()` | Returns string lengths. | Find unusually long URLs. |
| `.str.count()` | Counts occurrences of a pattern. | Count separators in product codes. |
| `.str.slice()` | Extracts portions of strings by position. | Extract a year from a structured code. |
| `.str.get()` | Gets an element from each string-like value. | Retrieve the first token after splitting. |
| `.str.pad()` | Pads strings to a specified width. | Format numeric IDs with leading spaces or characters. |
| `.str.zfill()` | Pads strings on the left with zeros. | Convert ID `123` to `000123`. |
| `.str.isnumeric()` | Checks whether characters are numeric. | Validate numeric-only text fields. |
| `.str.isalpha()` | Checks whether characters are alphabetic. | Validate a text-only field. |
| `.str.match()` | Checks whether text matches a pattern at the start. | Validate a product-code prefix. |

``` python
df = pd.DataFrame({
    "Email": [" RAVI@EXAMPLE.COM ", "priya@example.com "]
})
df["Email"] = df["Email"].str.strip().str.lower()
```

**Real-world scenario:** A marketing team standardizes email addresses
before deduplicating a mailing list.

## 11. Date and Time Operations

Use `.dt` to extract and analyze components of datetime columns.

| Function | One-line explanation | Real-time example |
| --- | --- | --- |
| `pd.to_datetime()` | Converts supported values to datetime. | Convert order-date text into dates. |
| `pd.to_timedelta()` | Converts values to time durations. | Represent delivery time in hours. |
| `pd.date_range()` | Generates a sequence of dates or timestamps. | Create a daily reporting calendar. |
| `.dt.year` | Extracts the year. | Group sales by year. |
| `.dt.month` | Extracts the month number. | Analyze monthly revenue. |
| `.dt.day` | Extracts the day of the month. | Analyze daily orders. |
| `.dt.day_name()` | Returns the weekday name. | Identify the busiest weekday. |
| `.dt.month_name()` | Returns the month name. | Label monthly charts. |
| `.dt.hour` | Extracts the hour. | Analyze hourly website traffic. |
| `.dt.dayofweek` | Returns weekday numbers, Monday = 0. | Compare weekday and weekend activity. |
| `.dt.quarter` | Extracts the calendar quarter. | Prepare quarterly business reports. |
| `.dt.is_month_end` | Checks whether a date is month-end. | Identify month-end transactions. |
| `.dt.is_month_start` | Checks whether a date is month-start. | Identify first-day records. |
| `.dt.days_in_month` | Returns the number of days in each month. | Calculate monthly reporting coverage. |
| `.dt.strftime()` | Formats datetime values as strings. | Display dates as `13-Oct-2026`. |
| `df.resample()` | Aggregates time-series data into time intervals. | Calculate weekly or monthly sales. |
| `df.asfreq()` | Converts a time series to a specified frequency. | Create a daily series with gaps represented. |
| `df.shift()` | Shifts values by rows or time periods. | Compare today's sales with the previous day. |
| `df.rolling()` | Calculates rolling-window statistics. | Calculate a seven-day moving average. |
| `df.expanding()` | Calculates statistics over an expanding window. | Track cumulative average sales. |
| `df.ewm()` | Calculates exponentially weighted statistics. | Smooth daily demand fluctuations. |

``` python
df = pd.DataFrame({
    "Order_Date": ["2026-10-01", "2026-10-02", "2026-10-03"],
    "Sales": [1000, 1500, 1200]
})

df["Order_Date"] = pd.to_datetime(df["Order_Date"])
df["Month"] = df["Order_Date"].dt.month
df["Weekday"] = df["Order_Date"].dt.day_name()
```

## 12. Statistical Functions

| Function | One-line explanation | Real-time example |
| --- | --- | --- |
| `df.mean()` | Calculates the average. | Find average order value. |
| `df.median()` | Calculates the middle value. | Summarize typical household income. |
| `df.std()` | Calculates standard deviation. | Measure variation in product demand. |
| `df.var()` | Calculates variance. | Compare sales variability across stores. |
| `df.min()` | Finds minimum values. | Identify the lowest daily temperature. |
| `df.max()` | Finds maximum values. | Find peak daily revenue. |
| `df.quantile()` | Calculates percentiles or quantiles. | Find the 95th percentile of response time. |
| `df.corr()` | Calculates correlations. | Explore relationships between model features. |
| `df.cov()` | Calculates covariance. | Explore joint changes in financial measures. |
| `df.skew()` | Measures distribution asymmetry. | Check whether prices have a long right tail. |
| `df.kurt()` | Measures distribution tail characteristics. | Investigate unusually extreme values. |
| `df.sem()` | Calculates the standard error of the mean. | Estimate uncertainty in average service time. |
| `df.mad()` | Removed in Pandas 2.0; historically calculated mean absolute deviation. | Calculate mean absolute deviation with a manual formula. |

``` python
print(df["Sales"].mean())
print(df["Sales"].median())
print(df["Sales"].std())
print(df["Sales"].quantile(0.95))
```

## 13. Reshaping Data

| Function | One-line explanation | Real-time example |
| --- | --- | --- |
| `df.pivot()` | Reshapes data using unique index/column combinations. | Create a region-by-month sales table. |
| `df.pivot_table()` | Creates a pivot table with aggregation for repeated combinations. | Summarize revenue by region and category. |
| `pd.crosstab()` | Calculates a frequency table for categorical variables. | Count task statuses by language. |
| `df.melt()` | Converts wide-format data into long-format data. | Convert monthly sales columns into month-value rows. |
| `df.stack()` | Moves column levels into the row index. | Reshape a multi-column report for analysis. |
| `df.unstack()` | Moves index levels into columns. | Turn grouped results into a matrix. |
| `pd.wide_to_long()` | Converts similarly named wide columns into long format. | Reshape year-specific sales columns. |
| `df.explode()` | Converts list-like elements into separate rows. | Expand a list of tags attached to each URL. |
| `df.squeeze()` | Converts a one-column or one-row object into a lower-dimensional object when possible. | Extract a Series from a one-column DataFrame. |
| `df.transpose()` | Swaps rows and columns. | Flip a small summary table for presentation. |

``` python
sales = pd.DataFrame({
    "Product": ["Laptop", "Phone"],
    "January": [100, 200],
    "February": [150, 250]
})

long_sales = sales.melt(
    id_vars="Product",
    var_name="Month",
    value_name="Sales"
)
```

## 14. Applying Custom Functions

| Function | One-line explanation | Real-time example |
| --- | --- | --- |
| `df.apply()` | Applies a custom function along an axis. | Calculate a score using multiple columns. |
| `Series.map()` | Applies mappings or functions to a Series. | Standardize labels using a dictionary. |
| `df.pipe()` | Passes an object into a function to chain transformations. | Organize a multi-step cleaning pipeline. |
| `df.agg()` | Applies aggregation functions to a Series, DataFrame, or grouped object. | Generate multiple KPI metrics. |
| `df.transform()` | Applies transformations while preserving the relevant output shape. | Normalize values within groups. |
| `df.groupby().filter()` | Retains groups satisfying a condition. | Keep departments with at least five employees. |

``` python
def calculate_bonus(salary):
    return salary * 0.10

df["Bonus"] = df["Salary"].map(calculate_bonus)
```

**Performance tip:** Prefer vectorized Pandas operations, such as
`df["Salary"] * 0.10`, over row-by-row Python functions when possible.

## 15. Index Management and Advanced Operations

| Function | One-line explanation | Real-time example |
| --- | --- | --- |
| `df.set_index()` | Sets one or more columns as the index. | Use `Customer_ID` as the customer table index. |
| `df.reset_index()` | Resets the index and optionally retains it as a column. | Restore a regular index after grouping. |
| `df.reindex()` | Conforms a DataFrame to specified index and column labels. | Align monthly reports to a common calendar. |
| `df.rename_axis()` | Renames an index or column axis. | Label an index as `Transaction_ID`. |
| `df.get()` | Retrieves a column by key, with an optional default if absent. | Safely retrieve an optional field. |
| `df.copy()` | Creates a copy of a DataFrame. | Preserve the raw dataset before cleaning. |
| `df.empty` | Checks whether a DataFrame has zero rows or columns. | Check whether an API returned any records. |
| `df.equals()` | Checks whether two DataFrames contain the same data and structure. | Verify whether two processed datasets match. |
| `df.memory_usage()` | Measures memory consumption by column. | Investigate large datasets. |
| `df.select_dtypes()` | Selects columns by data type. | Select numeric model features. |
| `df.flags` | Exposes certain DataFrame flags, including copy-related behavior. | Inspect DataFrame configuration. |
| `df.pipe()` | Passes data through a function as part of a chain. | Build reusable data-processing pipelines. |
| `pd.factorize()` | Encodes unique values as integer codes. | Create categorical codes for exploration. |
| `pd.get_dummies()` | Converts categorical values into indicator columns. | One-hot encode product categories. |
| `pd.cut()` | Divides continuous values into specified intervals. | Group customers into spending bands. |
| `pd.qcut()` | Divides values into quantile-based groups. | Segment customers into spending quartiles. |
| `pd.to_numeric()` | Converts values to numeric types where possible. | Convert imported price strings into numbers. |
| `pd.api.types.is_numeric_dtype()` | Checks whether a data type is numeric. | Validate model input columns. |

``` python
raw_df = df.copy()

encoded_df = pd.get_dummies(
    df,
    columns=["Department"],
    dtype=int
)

df["Experience_Band"] = pd.cut(
    df["Experience"],
    bins=[0, 2, 5, 10],
    labels=["Junior", "Mid-level", "Senior"]
)
```

## 16. End-to-End Real-Time Example: Data Annotation EDA

Suppose you have a dataset tracking URL annotation tasks across multiple
languages. Your goal is to identify total URLs, completed URLs, pending
URLs, and language-wise progress.

``` python
import pandas as pd

# Load annotation data
df = pd.read_csv("annotation_data.csv")

# 1. Inspect the dataset
print(df.head())
print(df.shape)
print(df.info())

# 2. Clean column names and text fields
df.columns = df.columns.str.strip()
df["Language"] = df["Language"].str.strip()
df["Status"] = df["Status"].str.strip().str.title()

# 3. Remove duplicate URLs
df = df.drop_duplicates(subset=["URLs"])

# 4. Check missing values
print(df.isna().sum())

# 5. Count records by status
print(df["Status"].value_counts(dropna=False))

# 6. Calculate language-wise annotation metrics
language_report = df.groupby("Language").agg(
    Total_URLs=("URLs", "nunique"),
    Completed_URLs=("Status", lambda s: s.eq("Success").sum())
).reset_index()

# 7. Calculate pending URLs
language_report["Pending_URLs"] = (
    language_report["Total_URLs"] - language_report["Completed_URLs"]
)

# 8. Calculate completion percentage
language_report["Completion_Percentage"] = (
    language_report["Completed_URLs"]
    / language_report["Total_URLs"].replace(0, pd.NA)
    * 100
).round(2)

# 9. Sort by completion percentage
language_report = language_report.sort_values(
    by="Completion_Percentage",
    ascending=False
)

# 10. Export the report
language_report.to_csv("language_annotation_report.csv", index=False)

print(language_report)
```

**Assumptions:** `Language`, `URLs`, and `Status` are actual column
names, and `Success` is the completion status. Adjust these to match
your source data. This example assumes each URL belongs to one language
and uses one status.

Expected report structure (illustrative numbers only):

| Language | Total URLs | Completed URLs | Pending URLs | Completion % |
| --- | ---: | ---: | ---: | ---: |
| English | 2,000 | 1,800 | 200 | 90.00 |
| Hindi | 1,500 | 1,200 | 300 | 80.00 |
| Kannada | 1,000 | 700 | 300 | 70.00 |

You can load the exported CSV into Power BI and create KPI cards,
language-wise bar charts, completion percentages, and status breakdowns.

## 17. Quick-Reference Cheat Sheet

 | Task | Function |
|------|----------|
| Read a file | `pd.read_csv()` |
| Preview records | `df.head()` |
| Check dataset size | `df.shape` |
| Inspect data types | `df.info()` |
| Summarize numerical data | `df.describe()` |
| Find missing values | `df.isna().sum()` |
| Remove duplicates | `df.drop_duplicates()` |
| Fill missing values | `df.fillna()` |
| Filter rows | `df.loc[]` |
| Select columns | `df[["col1", "col2"]]` |
| Sort records | `df.sort_values()` |
| Group and summarize | `df.groupby()` |
| Join tables | `pd.merge()` |
| Combine files | `pd.concat()` |
| Clean text | `Series.str.strip()` |
| Convert dates | `pd.to_datetime()` |
| Create columns | `df.assign()` |
| Create a pivot table | `pd.pivot_table()` |
| Encode categories | `pd.get_dummies()` |
| Export a report | `df.to_csv()` |

## 18. Recommended Learning Order

1.  **Beginner:** Create DataFrames, read CSV files, and use `head()`,  `shape`, `info()`, `columns`, and `dtypes`.
2.  **Data cleaning:** Practice `isna()`, `fillna()`, `dropna()`, `drop_duplicates()`, `replace()`, and `astype()`.
3.  **Data analysis:** Learn filtering, sorting, `groupby()`, `agg()`, `value_counts()`, and `describe()`.
4.  **Data transformation:** Practice `apply()`, `map()`, `merge()`, `concat()`, `pivot_table()`, and `melt()`.
5.  **Time series and advanced EDA:** Learn datetime operations,  `rolling()`, `resample()`, correlations, and performance optimization.
6.  **Real-world projects:** Build a sales EDA report, customer segmentation analysis, and data annotation dashboard dataset.

## Final Advice

You do not need to memorize every Pandas function. First become comfortable with data inspection, cleaning, filtering, grouping, merging, and exporting. Those skills cover a large proportion of
day-to-day Data Analyst and Data Scientist tasks.

For complete API details and version-specific information, refer to the
[official Pandas documentation](https://pandas.pydata.org/docs/).
