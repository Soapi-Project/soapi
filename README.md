# soapi
### A Python tool that automatically preprocesses data sets and prepares them for analysis. Enjoy soaping it up!<br>

## Usage Incentives

The purpose of the **soapi package** is to speed up the data cleaning process which would significantly reduce machine learning training time.  
<br>
The **soapi package** stores most of the generic functions data scientists use such as remove outliers, remove duplicates, fill in missing values, etc. 
<br>

## Installation
soapi is built to use pandas DataFrames and some scikit-learn modules for data preprocessing. We recommend installing the [Anaconda Python distribution](https://www.anaconda.com/products/distribution) prior to installing soapi. <br>

Link to soapi: https://test.pypi.org/project/soapii/0.1.0/

### Dependencies <br> 
Requirements: <br>
* numpy (>=1.22)
* pandas (>=1.3.5)
* pandas_profiling (>=3.2.0)
* scikit_learn (>=0.21.3)
* setuptools>=41.4.0


Once the prerequisites are installed, run from your command line the following:
<br >

``pip install --extra-index-url https://test.pypi.org/simple/ soapi``
<br >

It will also install all the dependent packages such as pandas, pandas_profiling, numpy, etc. <br>

## Function Call References

### Data Cleaning 

| Function        | Description      |
| ----------- | ------------ |
| clean(dataframe, column_threshold, row_threshold, do_outlier, fill)         | Upper-level function that calls drop_bad_categories, fill_missing_values, and replace_outliers at once for easier implementation.   |
| drop_bad_categories(dataframe, column_threshold, row_threshold, in_place) | Drops static columns, duplicate columns, and columns/rows in the dataframe that do not meet the percentage thresholds of non-null cells.  |
| fill_missing(data)                                                    | Fills missing cells in the dataframe with the mean of its respective column.  |
| fill_missing_values(dataframe)                                        | Fills missing cells with the median of its respective column.  |
| impute_missing(dataframe, type)                                       | Fills missing cells with the specified method (mean, median, most frequent, or constant).  |
| replace_outliers(dataframe, columns, factor, method, treatment, do)   | Caps the numerical outliers and replaces them via the specified method or removes them according to the user’s preference.  |
| combine_categories(dataframe, column, category_name, threshold)       | Combines categories with percentages of instances below the threshold into one category for representation.  |
| fill_missing_numeric(dataframe, use_limit, limit_percentage, column_names) | Fill missing numeric values with -999|
| fill_missing_categorical(dataframe, use_limit, limit_percentage, column_names) | Fill missing categorical values with "None"|

### Data Transformation 

| Function      | Description      |
| ----------- | ------------ |
| transform(dataframe, column_headers)                                  | Upper-level function that calls transform_categorical with its given parameters.  |
| transform_categorical(dataframe, column_headers, return_copy)         | Transforms all object datatypes into numerical using built-in algorithm, so it can be run through ML algorithm and converted back to previous datatype afterwards.  |
| preprocess_dates(dataframe, column_name, epoch)                          | Converts many different methods of representing dates to one uniform method or to epoch time.|

### Data Postprocessing 

| Function        | Description      |
| ----------- | ------------ |
| postprocess(original_dataframe, dataframe, column_headers)            | Upper-level function that sets up and calls inverse_transform_categorical. Used after ML has taken place  |
| inverse_transform_categorical( original_dataset, copy, column_headers)| Converts the dataframe that has been previously transformed back from numerical values to categorical values for reading and interpretation after ML has taken place.  |
| postprocess_dates(dataframe, columns, format)                               | Converts epoch time into a readable date in a format that the user chooses.|
| inverse_transform_numeric(dataframe, column_names) |     Converts transformed numeric values from -999.0 to NaN|
| inverse_transform_none_categorical(dataframe, column_names) |     Converts transformed categorical values from "None" to NaN|

    
The soapi preprocessing library is unique in that it can do reverse transformation

![outputDates](readme_pics/outputDates.png) <br>

The 'LocationAbbr' column shows that the first two rows locations are "US" <br>

![outputReverseDates](readme_pics/outputReverseDates.png) <br>

The transform function was called and the rows converted the strings into numerical values. For example, US became the value '44.0' <br>


### Helper Functions 

| Function        | Description      |
| ----------- | ------------ |
| load_csv(file_path, missing_headers)                                  | Loads the CSV file into the program and converts it into a dataframe.  |
| data_copy(dataframe)                                                  | Creates a copy of the dataframe and returns it.  |
| get_categorical_columns(dataframe)                                    | Returns a list of categorical columns in the dataframe.  |
| get_numerical_columns(dataframe)                                      | Returns a list of numerical columns in the dataframe.  |
| columns_type(dataframe, verbose)                                      | Returns the number of categorical and numerical columns in an array.  |
| compare_columns(original_dataframe, final_dataframe, as_list)         | Compares two dataframes and the columns that are lost between them.  |
| get_profile(dataframe, file_name, show)                               | Saves the Pandas Profile report for the dataframe as a file and offers an option to display it|


    
## Contributing to soapi <br>
We welcome everyone to check the library for bugs or enhancements to work on. 
<br>
<br>

### Turn that dirty data into shiny data after soaping it up! 
