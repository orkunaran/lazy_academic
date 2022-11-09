"""
Lazy academics is a tool to create summary statistics, hypothesis testing and tables for your academic
manuscript.
Please check requirements.txt for required packages
"""
import pandas as pd
import streamlit as st
from io import BytesIO
from collections import Counter as c
import numpy as np

# statistical packages
from scipy.stats import mannwhitneyu, wilcoxon, chisquare, f_oneway, kruskal, ttest_ind

# functions to run
# create objects to run functions
df = pd.DataFrame()
table = pd.DataFrame()
category: None = None
categories = []
n_categories = 0


def read_file(input_file):
    """
    Function to read data file and save it as a dataframe
    :param input_file: uploaded data file, can be xlsx, csv or SPSS file
    :return: dataframe object df
    """
    global df
    if input_file.name.endswith(".xlsx"):
        df = pd.read_excel(input_file)
    elif input_file.name.endswith(".csv"):
        df = pd.read_csv(input_file)
    elif input_file.name.endswith(".sav"):
        df = pd.read_spss(input_file)

    return df


def columns_to_drop():
    """
    Removes unwanted columns from the dataframe
    :return: dataframe object
    """

    columns = st.multiselect(label='Select columns to drop',
                             options=df.columns)

    drop = st.checkbox('drop columns')
    if drop:
        df.drop(columns, axis=1, inplace=True)

    return df


def drop_nan():
    """
    :return:
    """
    try:
        missings = df[category].isna().sum()
        if missings > 1:
            df.dropna(subset=[category],
                      inplace=True)
    except:
        st.write("No columns selected, no samples will be dropped")
        pass
    return df


def handle_nan(column):
    """

    :return:
    """
    if df[column].isna().sum() > 0:
        df.dropna(subset=column, how='all', inplace=True)
    return df


def change_data():
    """
    A function to change dtypes of df columns
    :return:
    """
    objects = st.multiselect(label='Select columns that includes text data or grouping variable ',
                             options=df.columns)

    nums = st.multiselect(label='Select columns that includes numeric data',
                          options=df.columns)

    dates = st.multiselect(label='Select columns that includes dates',
                           options=df.columns)

    convert = st.checkbox('convert selected columns')
    if convert:
        for col in objects:
            df[col] = df[col].astype('object')
        for col in nums:
            df[col] = df[col].astype('float32')
        for col in dates:
            df[col] = df[col].astype('datetime64[ns]')
    else:
        pass

    return df


cols_dict = {}


def change_col_names():
    """
    A function to change column names. Best practices are for pre-post treatment comparison. The column names should be
    ordered.
    :return: changed column name of the df
    """

    # notes for myself
    ## can restart after changing one column name, need to find how to reset the selection box without
    ## deleting the cols_dict

    global cols_dict
    columns = st.multiselect(label='Select column to change name',
                             options=df.columns)
    new_col_name = st.text_input(label='Enter the desired name for that column')

    new_name = list(zip(columns, new_col_name))

    add = st.button(label='Click to add your selection to the changing list')

    if add:
        for i in range(len(new_name)):
            cols_dict[new_name[i][0]] = new_name[i][1]
        st.write(f"{columns} : {new_col_name} added to change list.")

    st.write(cols_dict)

    change = st.checkbox('change column names')
    if change:
        df.rename(columns=cols_dict, inplace=True)
    return df


def compare_groups():
    """
    this function receives data and gets numeric columns.
    :return: an Excel file that includes each parameter's mean ± sd, (min - max) and hypothesis testing if selected
    """
    global table
    global category
    global categories
    global n_categories
    # errors
    column_names = [i for i in df.columns]
    column_names.append(None)

    if category not in column_names:
        raise ValueError(f"{category} not in dataframe columns")

    # define a category
    # select object columns
    object_columns = [col for col in df.select_dtypes('object')]
    # add None to object_columns
    object_columns.append(None)
    # select category to group data
    category = st.selectbox(label='Select the grouping variable',
                            options=object_columns)

    # drop rows if category == NA exists
    drop_nan()

    # define categories and n_categories
    if category is None:
        pass
    else:
        categories = df[category].unique()
        n_categories = len(categories)
        st.write(f"Selected category = {category}, \n "
                 f"Unique values = {categories}, \n"
                 f"Elements in each category : {c(df[category])}")

    hypothesis = st.checkbox(label='Check this box if you want to test the Null hypothesis')

    if hypothesis:
        if category is None:
            raise ValueError("You need a grouping variable to compare means")
        elif n_categories <= 1:
            raise ValueError(
                f"Category classes must be other than 1. The classes in {category} : {df[category].unique()}")
        else:
            if n_categories == 2:
                selection = st.selectbox(label='Select the test you want to run',
                                         options=['Student t test (parametric)',
                                                  'Mann Whitney U test (non-parametric)'])

                test = mannwhitneyu
                if selection == 'Student t test (parametric)':
                    test = ttest_ind
                else:
                    pass

                table['variable_name'] = df.select_dtypes(exclude='object').columns
                for index, cat in enumerate(categories):
                    for col in df.select_dtypes(exclude='object').columns:
                        handle_nan(col)
                        table.loc[table.variable_name == col, f"{category}: {cat} \n mean ± sd "] = \
                            f"{df[df[category] == cat][col].mean():.2f} ± {df[df[category] == cat][col].std():.2f}"

                p_value = []
                for col in df.select_dtypes(exclude='object').columns:
                    p_value.append(
                        test(df[df[category] == categories[0]][col], df[df[category] == categories[1]][col])[1])

                table['p value'] = p_value

            else:
                selection = st.selectbox(label='Select the test you want to run',
                                         options=['ANOVA (parametric)',
                                                  'Kruskal Wallis (non-parametric)'])

                table['variable_name'] = df.select_dtypes(exclude='object').columns
                for index, cat in enumerate(categories):
                    for col in df.select_dtypes(exclude='object').columns:
                        handle_nan(col)
                        table.loc[table.variable_name == col, f"{category}: {cat} \n mean ± sd "] = \
                            f"{df[df[category] == cat][col].mean():.2f} ± {np.std(df[df[category] == cat][col]):.2f}"

                if selection == 'ANOVA (parametric)':
                    data_list = []
                    for column in df.select_dtypes(exclude='object').columns:
                        #  handle_nan(column)
                        for cat in categories:
                            data_list.append(df[df[category] == cat][column])
                        p = f_oneway(*data_list)[1]
                    table['p_value'] = p
                else:

                    data_list = []
                    for column in df.select_dtypes(exclude='object').columns:
                        #  handle_nan(column)
                        for cat in categories:
                            st.write(df[df[category]])
                            data_list.append(df[df[category] == cat][column])
                        p = kruskal(*np.array(data_list), nan_policy='omit')[1]
                    table['p_value'] = p

    else:
        if category is None:
            for col in df.select_dtypes(exclude='object').columns:
                table = table.append(
                    {"variable_name": col,
                     "mean ± standard deviation": f"{df[col].mean():.2f} ± {df[col].std(ddof=0):.2f}"
                     },
                    ignore_index=True)
        else:
            if n_categories <= 1:
                raise ValueError(f"Category classes must be other than 1. The classes in {category} : {categories}")
            else:
                table['variable_name'] = df.select_dtypes(exclude='object').columns
                for index, cat in enumerate(categories):
                    for col in df.select_dtypes(exclude='object').columns:
                        table.loc[table.variable_name == col, f"{category}: {cat} \n mean ± sd "] = \
                            f"{df[df[category] == cat][col].mean():.2f} ± {df[df[category] == cat][col].std():.2f}"

    return table





def post_hoc():
    """
    
    :return:
    """
    # check if the p values are lower than 0.05
    p_values = [p for p in table['p']]
    indexes = [index for index, p in enumerate(p_values) if p < 0.05]
    # retrive column names from table
    columns_for_posthoc = [table.loc[i, 'variable_name'] for i in indexes]


# setting up streamlit page
st.set_page_config(page_title="Lazy Academic: A tool for building tables and Hypothesis testing",
                   initial_sidebar_state="expanded")

st.write("# Welcome to Lazy Academic app!!")

input_file = st.file_uploader("Upload you data file in csv, excel or spss sav file forms",
                              type={".csv", ".xlsx", ".sav"})

if input_file is not None:
    read_file(input_file)

if input_file:
    st.write('## Data Preview')
    st.write(df.head())

    st.write('### Data Info- Missing count and dtypes')
    missing_df = pd.DataFrame()
    for col in df.columns:
        missing_df = missing_df.append({'column': col,
                                        'n_missing': df[col].isna().sum()
                                        },
                                       ignore_index=True)
    st.dataframe(missing_df)

    st.write('Choose columns to drop')
    columns_to_drop()

    st.write('## Define and Change Column Data Types')
    change_data()

    # st.write('Check and correct column names')
    # change_col_names()

    st.write('### Print Summary Statistics Table')
    st.write('Choose a category if available, else choose None')
    st.write(compare_groups())

    # save table as xlsx

    # download the data
    output = BytesIO()

    # Write files to in-memory strings using BytesIO
    table.to_excel(output, sheet_name='Sheet1', index=False, header=True)
    output.seek(0)

    st.download_button(
        label="Download Excel workbook",
        data=output.getvalue(),
        file_name="table.xlsx",
        mime="application/vnd.ms-excel"
    )
