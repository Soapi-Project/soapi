import soapi
import pandas as pd
import numpy as np
from sklearn.preprocessing import OrdinalEncoder
from sklearn.impute import SimpleImputer

df = soapi.load_csv('testData.csv')


def test_importCsv():
    assert soapi.load_csv('testData.csv').equals(pd.read_csv('testData.csv'))


def test_data_copy():
    assert soapi.data_copy(df).equals(df)


def test_cleanAllNumeric():
    data = pd.DataFrame({'1': np.random.rand(1000),
                         '2': np.random.rand(1000),
                         '3': np.random.randint(0, 3, 1000)})
    data.loc[12:74, '1'] = np.nan
    data.loc[2:10, '2'] = np.nan
    data.loc[20:35, '3'] = np.nan

    data = soapi.fill_missing(data)

    copy = soapi.data_copy(data)
    copy['1'].fillna(copy['1'].median(), inplace=True)
    copy['2'].fillna(copy['2'].median(), inplace=True)
    copy['3'].fillna(copy['3'].median(), inplace=True)

    assert data.equals(copy)


def test_transformAndInverse():
    data = soapi.load_csv('testData.csv')
    copy = soapi.data_copy(data)

    data, data_copy = soapi.transform_categorical(data, return_copy=True)
    copy, copy2 = soapi.transform_categorical(copy)

    data = soapi.inverse_transform_categorical(data_copy, data)
    copy = soapi.inverse_transform_categorical(copy2, copy)

    # Copy is the DataFrame before transformation, so if we inverseTransform,
    # data SHOULD equal copy if it inverses correctly
    assert data.equals(copy)


def test_clean():
    data = soapi.load_csv('testData.csv')

    copy = soapi.data_copy(data)

    data = soapi.clean(data)

    soapi.drop_bad_catagories(copy)
    copy = soapi.fill_missing_values(copy)
    copy = soapi.replace_outliers(copy, do_outliers=True)

    assert data.equals(copy)



def test_allStagesTest():
    data = soapi.load_csv('testData.csv')
    copy = soapi.data_copy(data)

    data = soapi.clean(data)
    data, data_copy = soapi.transform(data)
    data = soapi.postprocess(data_copy, data)

    soapi.drop_bad_catagories(copy)
    copy = soapi.fill_missing_values(copy)
    copy = soapi.replace_outliers(copy, do_outliers=True)

    copy, copyBeforeTransform = soapi.transform_categorical(copy)

    copy = soapi.inverse_transform_categorical(copyBeforeTransform, copy)

    assert data.equals(copy)


def test_transformWithGivenColumns():
    data = soapi.load_csv('testData.csv')

    copy = soapi.data_copy(data)

    data = soapi.clean(data)
    data = soapi.transform_categorical(data, return_copy=False)

    copy = soapi.clean(copy)
    categorical_col = soapi.get_categorical_columns(copy)
    copy = soapi.transform_categorical(copy, column_headers=categorical_col, return_copy=False)

    assert data.equals(copy)


def test_cleanWithStrings():
    data = pd.DataFrame({'1': np.random.randint(1, 4, 1000)})

    strings = {1: "soap", 2: "wash", 3: "clean"}
    data['1'] = data['1'].apply(lambda x: strings[x])

    copy = data[:500].copy()

    copy = soapi.clean(copy)

    data = soapi.clean(data)

    assert data.equals(copy)


def test_cleanWithNanStrings():
    data = pd.DataFrame({'1': np.random.randint(1, 4, 1000)})

    copy = soapi.data_copy(data)

    strings = {1: "soap", 2: "wash", 3: "clean"}
    data['1'] = data['1'].apply(lambda x: strings[x])

    data.loc[120:280, '1'] = np.nan
    copy = soapi.data_copy(data)

    data = soapi.clean(data)
    copy = soapi.clean(copy)

    assert data.equals(copy)


def test_dateTimes():
    data = pd.DataFrame(
        {'date': ['December 18th, 2001', '12/18/2001', '2001-12-18', '18 December 2001', '2001 December 18']})
    # 12/18/2001 in unix time is 1008633600, so all of these date inputs should give the same output
    unix = pd.DataFrame({'date': [1008633600, 1008633600, 1008633600, 1008633600, 1008633600]})

    soapi.transform_dates(data, column_headers = ['date'])


    assert data.equals(unix)


def test_categoricalColAfterTransform():
    data = soapi.load_csv('testData.csv')

    soapi.clean(data)

    data, copy = soapi.transform(data)

    assert soapi.get_categorical_columns(data) == []


def test_getNumericColumn():
    data = soapi.load_csv('testData.csv')

    assert soapi.get_numerical_columns(data) == ['Age', 'Paid']


def test_getNumericColumnAfterTransform():
    data = soapi.load_csv('testData.csv')

    soapi.clean(data)
    soapi.transform(data)

    cols = soapi.get_numerical_columns(data)

    assert soapi.get_numerical_columns(data) == cols


def test_getCategoricalColumn():
    data = soapi.load_csv('testData.csv')


    assert soapi.get_categorical_columns(data) == ['Location', 'Gender', 'Type', 'Date']


def test_getCategoricalColumnAfterTransform():
    data = soapi.load_csv('testData.csv')

    soapi.clean(data)
    soapi.transform(data)

    assert soapi.get_categorical_columns(data) == []


def test_columnTypes():
    data = soapi.load_csv('testData.csv')

    numCols = soapi.columns_type(data)  # Index 0 gives Categorical Columns, Index 1 gives Numerical

    assert numCols[0] == 4 and numCols[1] == 2


def test_combineCategorical():
    data = pd.DataFrame({'cat': ['male', 'male', 'male', 'male', 'male', 'female', 'female', 'female', 'female',
                                 'overall', 'overall', 'overall', 'overall', 'overall', 'overall',
                                 '65+', '65+', '75+', '75+']})
    soapi.combine_categories(data, 'cat', category_name='Senior Citizen', threshold=0.2)
    # Should make 65+ and 75+ turn into "Senior Citizens" in DataFrame

    data2 = pd.DataFrame({'cat': ['male', 'male', 'male', 'male', 'male', 'female', 'female', 'female', 'female',
                                  'overall', 'overall', 'overall', 'overall', 'overall', 'overall',
                                  'Senior Citizen', 'Senior Citizen', 'Senior Citizen', 'Senior Citizen']})
    assert data.equals(data2)


def test_compareRemovedCols():
    data = soapi.load_csv('testData.csv')
    data2 = soapi.data_copy(data)

    soapi.clean(data, column_threshold=.4)  # Should remove "Type", "Paid", "Date"

    removed = soapi.compare_columns(data2, data, as_list=True)

    assert removed == ['Type', 'Paid', 'Date']

