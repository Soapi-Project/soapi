import soapi as sp

df = sp.load_csv('/test/testData.csv')
copy = sp.data_copy(df)

sp.get_profile(copy, file_name="Original Demo Report")

print("\n Original Dataset: \n" + str(df))

sp.clean(df)
print("\n After Cleaning: \n" + str(df))

sp.transform(df,column_headers=['Location','Gender','Type'])
sp.transform_dates(df,column_headers=['Date'])

print("\n After Transforming: \n" + str(df))

sp.postprocess(copy,df,column_headers=['Location','Gender','Type'])
sp.postprocess_dates(df, column_headers=['Date'], format='%m/%d/%Y')

print("\n After Postprocessing: \n" + str(df))

sp.get_profile(df, file_name="Cleaned Demo Report")






