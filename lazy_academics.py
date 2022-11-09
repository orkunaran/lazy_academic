from classes import lazy_academic
import streamlit as st
import pandas as pd

la: lazy_academic = lazy_academic()
# setting up streamlit page
# st.set_page_config(page_title="Lazy Academic: A tool for building tables and Hypothesis testing",
#                   initial_sidebar_state="expanded")

st.write("# Welcome to Lazy Academic app!!")

input_file = st.file_uploader("Upload you data file in csv, excel or spss sav file forms",
                              type={".csv", ".xlsx", ".sav"})

# show nothing if no data is uploaded
if not input_file:
    st.warning('Waiting for data to be uploaded.')
    st.stop()

df = la.read_file(input_file)
st.success('Data Uploaded.')
# print the data
st.dataframe(df)


# create a table for missing values and data types
st.write('### Data Info - missing values')
missing_df = pd.DataFrame(columns=['column', 'n_missing', 'data_type'])
for col in df.columns:
    df_new_row = pd.DataFrame.from_records({'column': [col],
                                            'n_missing': [df[col].isna().sum()],
                                            'data_type': [str(df[col].dtypes)]})
    missing_df = pd.concat([missing_df, df_new_row])

st.dataframe(missing_df)


st.write('### Data Cleaning - Change Column Types')
change = la.change_data()

st.write('### Data Cleaning - Drop Columns')
drops = la.columns_to_drop

# Let the users choose their actions
# create a selection dictionary

selection = st.selectbox(label="Please select an option to run",
                         options=['1 - None',
                                  '2 - Descriptive Statistics',
                                  '3 - Compare two or more groups',
                                  '4 - Compare pre-post treatment/intervention results',
                                  '5 - Compare pre-pst treatment/intervention results within groups']

                         )

if selection == '1 - None':
    pass
elif selection == '2 - Descriptive Statistics':
    category = la.define_category
    if category is None or len(category) < 2:
        table = la.descriptive_nums
        st.write(table)
        la.download_table()
    else:
        table = la.descriptive_multiple_nums
        st.write(table)
        la.download_table()
elif selection == '3 - Compare two or more groups':
    table = la.compare_groups()
    st.write(table)
    la.download_table()
elif selection == '4 - Compare pre-post treatment/intervention results':
    table = la.test_treatment
    st.write(table)
    la.download_table()
else:
    st.warning('Will be implemented in the future')

