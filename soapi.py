import pandas as pd
import numpy as np
from sklearn.preprocessing import OrdinalEncoder
from sklearn.impute import SimpleImputer
from pandas_profiling import ProfileReport
import os

#Read data as csv and return as pandas data frame
def load_csv(file_path : str, missing_headers: bool = False):
    '''
    Loads the CSV file into the program and converts it into a dataframe.

    Parameters:
        - file_path: A string of the desired CSV file's path. This could be it's path relative to the document that this is being run in or it can be its total path.
        - missing_headers: A boolean specifying if the dataset has a header row, true or false. Default is false.
    
    Returns:
        A Pandas DataFrame version of the inputted CSV file
    '''
    if missing_headers:
        data = pd.read_csv(file_path, header=None)
    else:
        data = pd.read_csv(file_path, header=0)

    # make shape of data frame global
    global rows, cols
    rows, cols = data.shape

    return data


def data_copy(dataframe: pd.DataFrame):
    '''
    Creates a copy of the dataframe and returns it.

    Parameters:
        - dataframe: Pandas DataFrame that is to be copied.
    
    Returns:
        A copy of the original dataframe. This is useful for when keeping an original copy of the data is needed.
    '''
    return dataframe.copy(deep = True)


# Data Cleaning Functions

def drop_bad_catagories(dataframe: pd.DataFrame, column_threshold: float = 0.75, row_threshold: float = 0.1, in_place = True):
    '''
    Drops static columns, duplicate columns, and columns/rows in the dataframe that do not meet the percentage thresholds of non-null cells.

    Parameters: 
        - dataframe: The Pandas DataFrame that is to be cleaned.
        - column_threshold:  A float between 0 and 1 representing the percentage limit of null cells allowed in each cell. Columns that don't meet this threshold will be dropped. Default is 0.75 (75%)
        - row_threshold: A float between 0 and 1 rrepresenting the percentage limit of null cells allowed in each row. Rows that don't meet this threshold will be dropped. Default is 0.1 (10%)
        - in_place: If true then the inputted Pandas DataFrame is edited in place. If false a copy will be returned.

    returns: 
        If in_place is true then this will return nothing but the original DataFrame will be edited. If in_place is false then this will return a copy of the DataFrame with the appropriate edits.
    '''

    while column_threshold > 1.0:
        column_threshold/= 10
        
    while row_threshold > 1.0:
        row_threshold/= 10
    
    if in_place:
        dataframe.dropna(axis=1, thresh=int((1-column_threshold)*len(dataframe)), inplace=True)
        dataframe.dropna(axis=0, thresh=int((1-row_threshold)*len(dataframe.columns)), inplace=True)
        dataframe.drop([e for e in dataframe.columns if dataframe[e].nunique() == 1], axis=1, inplace=True)
        dataframe.drop_duplicates(subset=None, keep='first', inplace=True)
    else:
        copy = data_copy(dataframe)
        copy = dataframe.dropna(axis=1, thresh=int((1-column_threshold)*len(dataframe)), inplace=False)
        copy = dataframe.dropna(axis=0, thresh=int((1-row_threshold)*len(dataframe.columns)), inplace=False)
        copy = dataframe.drop([e for e in dataframe.columns if dataframe[e].nunique() == 1], axis=1, inplace=False)
        copy = dataframe.drop_duplicates(subset=None, keep='first', inplace=False)
        return copy


def fill_missing(dataframe: pd.DataFrame):
    '''
    Fills missing cells in the dataframe with the median of its respective column.
    
    Parameters:
        - dataframe: The Pandas Dataframe with missing values that is to be edited.
    
    Returns:
        A copy of the inputted Pandas DataFrame with its missing values filled with the cell's column's median.
    '''
    copy = data_copy(dataframe)
    copy = copy.fillna(copy.median())
    return copy

##caps numerical outlier values or removes them
def replace_outliers(dataframe: pd.DataFrame, columns: list[str] = None, factor: float = 1.5, method: str = 'IQR', treament: str = 'cap', do_outliers: bool = True): 
    '''
    Caps the numerical outliers and replaces them via the specified method or removes them according to user preference.

    Parameters:
        - dataframe: The Pandas DataFrame that the user would like to implement this function on
        - columns: Optional. If desired, it is a list of strings representing column names to apply this function to. If left empty, the function will be applied to every column.
        - factor: The float tolerance amount of how many standard deviations from the mean that a numerical value can be. Default is 1.5
        - method: A string of which method to use for capping outliers, 'STD' (standard deviation method) or 'IQR' (Inter-Quartile Range). Default is 'IQR'
        - treatment: A string of whether to cap outliers or remove them. Use 'remove' to remove and 'cap' to cap.
        - do_outliers: A boolean to give the user the option to bypass the whole process and return the original DataFrame. This is useful for the clean function that uses this but should be left as the default true otherwise.
    
    Returns:
        The original Pandas DataFrame if do_outliers is false or the DataFrame with outliers removed or capped if do_outliers is true. 
    '''
    if do_outliers == True:
        if not columns:
            columns = dataframe.columns
        
        for column in columns:      ##iterates through columns
            if dataframe[column].dtype == int or dataframe[column].dtype == float:     ##type checks data in column is numerical
                
                if method == 'STD':     ##STD method of capping outliers
                    permissable_std = factor * dataframe[column].std()
                    col_mean = dataframe[column].mean()
                    floor, ceil = col_mean - permissable_std, col_mean + permissable_std
                elif method == 'IQR':       ##IQR method of capping outliers
                    Q1 = dataframe[column].quantile(0.25)
                    Q3 = dataframe[column].quantile(0.75)
                    IQR = Q3 - Q1
                    floor, ceil = Q1 - factor * IQR, Q3 + factor * IQR
                
                if treament == 'remove':    ##removes outliers
                    dataframe = dataframe[(dataframe[column] >= floor) & (dataframe[column] <= ceil)]
                elif treament == 'cap':     ##caps outliers
                    dataframe[column] = dataframe[column].clip(floor, ceil)
        
    return dataframe

#replace missing values with the mode for categorical columns and median for continuous variables 
def fill_missing_values(dataframe: pd.DataFrame):
    '''
    Fills missing cells with the median of its respective column.

    Parameters:
        - dataframe: The Pandas DataFrame with missing values that is to be filled. 
    
    Returns:
        The dataframe with filled in values. For categorical columns, missing values are filled with mode and for continuous 
        variables, missing values is filled with median. 
    '''
     
    for column in dataframe.columns.values:
        try:
            dataframe[column].fillna(dataframe[column].median(), inplace=True)
        except TypeError:
            most_frequent = dataframe[column].mode()

            if len(most_frequent) > 0:
                dataframe[column].fillna(dataframe[column].mode()[0], inplace=True)
            # else:
            #     data[column].fillna(method='bfill', inplace=True)
            #     data[column].fillna(method='ffill', inplace=True)

    return dataframe



def impute_missing(dataframe: pd.DataFrame, type: str ="mean"):
    '''
    Fills missing cells with the specified method (mean, median, most frequent, or constant).
    
    Parameters:
        - dataframe: The Pandas DataFrame with the missing values to be filled.
        - type: A string specifying the method of fillig the missing values. "mean", "median", "most_frequent", or "constant" with the default being "mean".
    
    Returns:
        The Pandas dataframe with the missing values filled in via the specified method.
    '''
    if (type=='mean'):
        si = SimpleImputer(strategy="mean")
        cols = get_numerical_columns(dataframe)
        dataframe[cols] = si.fit_transform(dataframe[cols])
        
    elif (type=='median'):
        si = SimpleImputer(strategy="median")
        cols = get_numerical_columns(dataframe)
        dataframe[cols] = si.fit_transform(dataframe[cols])
    
    elif (type=='most_frequent'):
        si = SimpleImputer(strategy="most_frequent")
        cols = get_categorical_columns(dataframe)
        dataframe[cols] = si.fit_transform(dataframe[cols])
    
    elif (type=='constant'):
        si = SimpleImputer(strategy="constant")
        cols = dataframe.columns
        dataframe[cols] = si.fit_transform(dataframe[cols])
    return dataframe

def fill_missing_numeric(dataframe: pd.DataFrame, use_limit: bool = False, limit_percentage: float = .20, column_names: list[str] = []):
    '''
    Fill missing numeric values with -999
    Choose what percentage of missing values you want to fill from .00 - .99, default is 20% 
    In order to reflect real data, we must have missing values to model accurately

    Parameters:
        - dataframe: The Pandas DataFrame to be changed
        - use_limit: Boolean to see if you want to use a limit or not.
        - limit_percentage: The float value that represents the percentage you
                            want to fill. By default fills 20%
        - column_names: String list of column names you specifically want to fill

    Returns:
        The Pandas DataFrame that has been changed
    '''
    cols = get_numerical_columns(dataframe=dataframe)

    if (len(column_names)):

        if (use_limit == True):
            for column in cols:
                nan = dataframe[column].isna().sum()
                limit = int(nan*limit_percentage)

                dataframe[column].fillna(value=-999, inplace=True, limit=limit)

        elif (use_limit == False):
            for column in cols:
                nan = dataframe[column].isna().sum()

                dataframe[column].fillna(value=-999, inplace=True)
    else:
        if (use_limit == True):
            for column in column_names:
                nan = dataframe[column].isna().sum()
                limit = int(nan*limit_percentage)

                dataframe[column].fillna(value=-999, inplace=True, limit=limit)

        elif (use_limit == False):
            for column in column_names:
                nan = dataframe[column].isna().sum()

                dataframe[column].fillna(value=-999, inplace=True)

    return dataframe

def fill_missing_categorical(dataframe: pd.DataFrame, use_limit: bool = False, limit_percentage: float = .20, column_names: list[str] = []):
    '''
    Fill missing categorical values with "None".
    Choose what percentage of missing values you want to fill from .00 - .99, default is 20% 
    In order to reflect real data, we must have missing values to model accurately

    Parameters:
        - dataframe: The Pandas DataFrame to be changed
        - use_limit: Boolean to see if you want to use a limit or not.
        - limit_percentage: The float value that represents the percentage you
                            want to fill. By default fills 20%
        - column_names: String list of column names you specifically want to fill

    Returns:
        The Pandas DataFrame that has been changed
    '''
    cols = get_categorical_columns(dataframe=dataframe)

    if (len(column_names) == 0):

        if (use_limit == True):
            for column in cols:
                nan = dataframe[column].isna().sum()
                limit = int(nan*limit_percentage)

                dataframe[column].fillna(value="None", inplace=True, limit=limit)

        elif (use_limit == False):
            for column in cols:

                dataframe[column].fillna(value="None", inplace=True)
    else:
        if (use_limit == True):
            for column in column_names:
                nan = dataframe[column].isna().sum()
                limit = int(nan*limit_percentage)

                dataframe[column].fillna(value="None", inplace=True, limit=limit)

        elif (use_limit == False):
            for column in column_names:
                dataframe[column].fillna(value="None", inplace=True) 


    return dataframe


    #column is for column name 
    #all categories that are less than the threshold are combined under the same category under a name, ex: 'other'
    #returns a list with all the name of categories that were combined under 'other'
def combine_categories(dataframe: pd.DataFrame, column: str, category_name: str = "other", threshold: float = 0.01):
    '''
    Combines categories with percentages of instances below the threshold into one category for representation.

    Parameters:
        - dataframe: The Pandas Dataframe that have similar categories to be combined. 
        - column: name of the column in the dataframe in string format
        - category_name: string name of the column that has combined categories 
        - threshold: float value that user specifies a threshold and all categories that are less than the threshold 
        are combined under the same category with a new name.
    
    Returns:
        A list with all the names of categories that were combined under a new name 

    '''
    to_combine = dataframe[column].value_counts()[dataframe[column].value_counts(normalize = True) < threshold].index
    dataframe[column].replace(to_combine,category_name, inplace=True)
    return to_combine



# Transformation Functions



oe = OrdinalEncoder()

def transform_categorical(dataframe: pd.DataFrame, column_headers: list[str] = [], return_copy: bool = True):
    """
    Transforms all object datatypes into numerical using built-in algorithm,
    so it can be ran through ML algorithm and converted back to previous datatype
    aftewards

    Parameters: 
        - data: A Pandas DataFrame with categorical values that is to be transformed 
        - return_copy: A boolean that specifies if you want a copy of the DataFrame before transformation

    returns:
        "data" DataFrame post transformation, return_copy=True returns "data" and a copy of the DataFrame before transformation

    Use this copy of the pre-transformed DataFrame to inverseTransform back to original
    labels

    """
    copy = data_copy(dataframe)
    if len(column_headers) == 0:
        categorical_data = dataframe.select_dtypes(include=['object']).columns.tolist()
        dataframe[categorical_data] = oe.fit_transform(dataframe[categorical_data].astype(str))
    else:
        dataframe[column_headers] = oe.fit_transform(dataframe[column_headers].astype(str))

    if (return_copy == True):
        return dataframe, copy
        
    return dataframe


def inverse_transform_categorical(original_dataset: pd.DataFrame, copy: pd.DataFrame, column_headers: list[str] =[]):
    """
    Reverse transforms the encoded values back to their original categorical state

    Parameters: 
        - original_dataset: The cleaned, but not transformed Pandas DataFrame (it will NOT be altered). We need the original dataset to grab the Categorical columns.
        - copy: The Pandas DataFrame that is going to be reverse transformed,and WILL be altered and returned


    returns: 
        "copy" DataFrame after it has been reversed back to pre-transformation stage, typically after you run it through ML algorithm

    """
    if len(column_headers) == 0:
        cols = get_categorical_columns(original_dataset)
        copy[cols] = oe.inverse_transform(copy[cols])
    else:
        copy[column_headers] = oe.inverse_transform(copy[column_headers])
    
    return copy

def inverse_transform_numeric(dataframe: pd.DataFrame, column_names: list[str] = []):
    '''
    Converts transformed numeric values from -999.0 to NaN

    Parameters:
        - dataframe: The Pandas DataFrame that is to be changed
        - column_names: A list of strings of the columns in the DataFrame that need to be inversed.

    
    Returns:
         A Pandas DataFrame that has the specified columns edited. 
    '''
    cols = get_numerical_columns(dataframe=dataframe)

    if (len(column_names) == 0):
        for column in cols:
            dataframe[column] = dataframe[column].replace({-999.0 : np.nan})
    
    else:
        for column in column_names:
            dataframe[column] = dataframe[column].replace({-999.0 : np.nan})
    
    return dataframe

def inverse_transform_none_categorical(dataframe: pd.DataFrame, column_names: list[str] = []):
    '''
    Converts transformed categorical values from "None" to NaN

    Parameters:
        - dataframe: The Pandas DataFrame that is to be changed
        - column_names: A list of strings of the columns in the DataFrame that need to be inversed.

    
    Returns:
         A Pandas DataFrame that has the specified columns edited. 
    '''
    cols = get_categorical_columns(dataframe=dataframe)

    if (len(column_names) == 0):
        for column in cols:
            dataframe[column] = dataframe[column].replace({"None" : np.nan})
    
    else:
        for column in column_names:
            dataframe[column] = dataframe[column].replace({"None": np.nan})
    
    return dataframe


##formats dates in file to all be unix
def transform_dates(dataframe: pd.DataFrame, column_headers: list[str], epoch: bool = True):
    '''
    Converts many different methods of representing dates to one uniform method or to epoch time.

    Parameters:
        - dataframe: The Pandas DataFrame that has the desired date-related column. This will remain unchanged.
        - column_headers: A list of strings of the columns in the DataFrame that need to be converted.
        - epoch: A boolean specifying if the user would like the column to be in epoch time if true or in YYYY-MM-DD format if false
    
    Returns:
         A Pandas DataFrame that has the specified columns edited. 
    '''
    
    if epoch == True:
        for col in column_headers:
            dataframe[col] = (pd.to_datetime(dataframe[col])- pd.Timestamp("1970-01-01")) // pd.Timedelta('1s')
    else:
        for col in column_headers:
            dataframe[col] = (pd.to_datetime(dataframe[col]))

    return dataframe


def postprocess_dates(dataframe: pd.DataFrame, column_headers: list[str], format: str):
    '''
    Converts epoch time into a readable date in a format that the user chooses.

    Parameters:
        - dataframe: Pandas DataFrame that contains columns in epoch time that need to be converted to a readable date format. This will remain unchanged
        - column_headers: A list of strings representing the names of columns that contain dates in epoch time
        - format: A single string of the format that the user would like the date to be converted into. This will contain the order of values seperated by a forward slash("/"). Example "%m/%d/%Y"
            - %m = month
            - %d = day
            - %Y = year
            - $f = include nanoseconds
            - For more: https://docs.python.org/3/library/datetime.html#strftime-and-strptime-behavior

    Returns:
        A Pandas DataFrame that has the specified columns edited. 

    '''

    for col in column_headers:
        dataframe[col] = pd.to_datetime(dataframe[col], unit='s')
        dataframe[col] = dataframe[col].dt.strftime(format)
    
    return dataframe

# Helper Functions 

#returns list of categorical columns
def get_categorical_columns(dataframe: pd.DataFrame) -> list:
    '''
    Gets a list of categorical columns in the dataframe. 

    Parameters:
        - dataframe: The Pandas Dataframe that is being inspected.
    
    Returns:
        "categorical_cols": A list of categorical columns in the dataframe. 
    '''

    categorical_cols = dataframe.select_dtypes(include="O").columns.tolist()
    return categorical_cols

#returns list of numerical columns
def get_numerical_columns(dataframe: pd.DataFrame) -> list:
    '''
    Gets a list of numerical columns in the dataframe.
    Parameters:
        - dataframe: The Pandas DataFrame that is being inspected.
    
    Returns:
        "numerical_cols": A list of numerical columns in the dataframe.
    '''

    numerical_cols = dataframe.select_dtypes(include=np.number).columns.tolist()
    return numerical_cols


#retuns number of categorical and numerical columns
def columns_type(dataframe: pd.DataFrame, verbose: bool =False) -> tuple:
    '''
    Returns the number of categorical and numerical columns in an array.

    Parameters:
        - dataframe:
        - verbose: A boolean that, when true, specifies that the user would like the lists of categorical/ numerical columns returned. When false, only the number of categorical and numerical columns will be returned.
    
    Returns:
        When verbose is true, this returns a tuple containing a list of the categorical columns followed by a list of the numerical columns. If false then a tuple of the number of categorical and numerical columns will be returned
    '''

    cat_columns = get_categorical_columns(dataframe)
    num_columns = get_numerical_columns(dataframe)

    if verbose:
        print(cat_columns)
        print(num_columns)

    return(len(cat_columns), len(num_columns))

def compare_columns(original_dataframe: pd.DataFrame, final_dataframe: pd.DataFrame, as_list: bool = False):
    '''
    Compares two dataframes and the columns that are lost between them.

    Parameters:
        - original_dataframe: The original Pandas Dataframe.
        - final_dataframe: The cleaned Pandas DataFrame. 
        - as_list: Boolean that specifies whether to print the list of columns that were lost in cleaning if false or return a list objects of lost columns if true. Defaults to false.
    
    Returns:
        A list of colunns in the first DataFrame but not in the second one if as_list is true or a printed version if false.
    '''

    removed = []
    for col in original_dataframe.columns:
            if col not in final_dataframe.columns:
                removed.append(col)
    
    
    # Option for the user to return a list or print the same list for visual inspection 
    if as_list == True:
        return removed
    else:
        if len(removed) <= 0:
            print("No columns were removed\n")
            print("Either completley different dataframes, the second dataframe was empty or the dataframes were flipped ")
        elif len(removed) < len(original_dataframe.columns): 
            print("These are the columns that were removed:\n")
            for number, letter in enumerate(removed):
                print(number+1, letter)
        else:
            print("Dataframes were flipped otherwise I don't know how this is possible")


def get_profile(dataframe: pd.DataFrame, file_name: str ="Pandas Profiling Report", show: bool = False):
    ''' 
    Saves the Pandas Profile report for the dataframe as a file and offers an option to display it

    Parameters:
        dataframe: The Pandas Dataframe to be profiled
        file_name: A string of what the user would like the profile to be called
        show: A boolean that, if true, will display a Pandas Profile Report in an iframe but nothing otherwise

    Returns:
        None but the profile file is saved in the project and a profile iframe may appear if specified

    '''
    dataframe_profile = ProfileReport(dataframe, title= file_name)
    dataframe_profile.to_file(os.path.dirname(os.path.abspath(__file__))+"/" +file_name)
    if show:
        dataframe_profile.to_notebook_iframe()

# Main Functions for calling 
        
def clean(dataframe: pd.DataFrame, column_threshold: float = 0.75, row_threshold: float = 0.1, do_outlier: bool = True, fill_missing: bool= True):
    '''
    Upper-level function that calls drop_bad_categories, fill_missing_values, and replace_outliers at once for easier implementation. 

    Parameters:
        - dataframe: The original dataframe that needs to be cleaned.
        - column_threshold:  A float between 0 and 1 representing the percentage limit of null cells allowed in each cell. Columns that don't meet this threshold will be dropped. Default is 0.75 (75%)
        - row_threshold: A float between 0 and 1 rrepresenting the percentage limit of null cells allowed in each row. Rows that don't meet this threshold will be dropped. Default is 0.1 (10%)
        - do_outlier: A boolean to specify if the user would like to use the default replace_outlierss function.
        - fill_missing: Allows the user to opt out of using the default fill_missing_values function and instead lets them use one of the other options for filling values
        
    
    Returns:
        The cleaned dataframe.
    '''
    drop_bad_catagories(dataframe, column_threshold, row_threshold)
    if fill_missing:
        fill_missing_values(dataframe)
    replace_outliers(dataframe, do_outliers = do_outlier)

    return dataframe

def transform(dataframe: pd.DataFrame, column_headers: list[str] = []):
    '''
    Upper-level function that calls transform_categorical with its given parameters.

    Parameters:
        - dataframe: Pandas DataFrame with categorical values.
        - column_headers: A list of strings (of column names) that the user specifically wants to do the transformation on. If empty, all categorical columns will be transformed.

    Returns:
        The original Pandas DataFrame but with the categorical columns transformed into numerical values.
    '''
    dataframe, copy = transform_categorical(dataframe, column_headers=column_headers)

    return dataframe, copy


def postprocess(original_dataframe: pd.DataFrame, dataframe: pd.DataFrame, column_headers: list[str] = []):
    '''
    Upper-level function that sets up and calls inverse_transform_categorical. Used after ML has taken place.

    Parameters:
        - original_dataframe: the original Pandas DataFrame. This is needed to pull the original data for conversion and will not be edited.
        - dataframe: The tranformed Pandas Dataframe that needs to be converted back. This will be edited.
        - column_headers: A list of strings (of column names) that the user specifically wants to do the transformation on. If empty, all categorical columns will be transformed back.
    
    Returns:
        A Pandas DataFrame that has the formerly categorical columns converted from their numerical equivalent back to their categorical value. 
    '''
    copied_dataframe = data_copy(original_dataframe)
    drop_bad_catagories(copied_dataframe)
    inverse_transform_categorical(copied_dataframe, dataframe, column_headers=column_headers)

    return dataframe


