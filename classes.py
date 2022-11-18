"""
Lazy academics is a tool to create summary statistics, hypothesis testing and tables for your academic
manuscript.
Please check requirements.txt for required packages
"""
from typing import Any, List

import pandas as pd
import streamlit as st
from io import BytesIO
from collections import Counter as c, Counter
import numpy as np

# statistical packages
from scipy.stats import mannwhitneyu, wilcoxon, chisquare, f_oneway, kruskal, ttest_ind, ttest_rel


class lazy_academic:

    def __init__(self):
        self.table = pd.DataFrame()
        self.df = pd.DataFrame()
        self.category = None
        self.n_categories = 0
        self.categories = []
        self.test = None
        self.table_control = pd.DataFrame()
        self.table_treatment = pd.DataFrame()
        self.paired_columns = []

    # noinspection PyTypeChecker
    def read_file(self, input_file: object = None) -> object:
        """
        Function to read data file and save it as a dataframe
        :param input_file: uploaded data file, can be xlsx, csv or SPSS file
        :return: dataframe object df
        """
        if input_file.name.endswith(".xlsx"):
            self.df = pd.read_excel(input_file)
        elif input_file.name.endswith(".csv"):
            self.df = pd.read_csv(input_file)
        elif input_file.name.endswith(".sav"):
            self.df = pd.read_spss(input_file)

        return self.df

    @property
    def columns_to_drop(self):
        """
        Removes selected columns from the dataframe
        :return: dataframe object
        """

        columns = st.multiselect(label='Select columns to drop',
                                 options=self.df.columns)

        drop = st.checkbox('drop columns')
        if drop:
            self.df.drop(columns, axis=1, inplace=True)
            st.success('Columns dropped')
        return self.df

    def drop_nan(self) -> object:
        """
        This function checks the missing values and drops the observations from th df
        :return: DataFrame object - df
        """
        try:
            missing = self.df[self.category].isna().sum()
            if missing > 1:
                self.df.dropna(subset=[self.category],
                               inplace=True)
        finally:
            pass

        return self.df

    def change_data(self) -> object:
        """
        A function to change dtypes of df columns
        :return:
        """
        objects = st.multiselect(label='Select columns that includes text data or grouping variable ',
                                 options=self.df.columns)

        nums = st.multiselect(label='Select columns that includes numeric data',
                              options=self.df.columns)

        dates = st.multiselect(label='Select columns that includes dates',
                               options=self.df.columns)

        convert = st.checkbox('convert selected columns')
        if convert:
            for col in objects:
                self.df[col] = self.df[col].astype('object')
            for col in nums:
                self.df[col] = self.df[col].astype('float32')
            for col in dates:
                self.df[col] = self.df[col].astype('datetime64[ns]')

        if convert:
            st.success('Data types have been changed')
        return self.df

    @property
    def define_category(self):
        self.category = None
        column_names = [i for i in self.df.columns]
        column_names.append(None)

        # select category to group data
        self.category = st.selectbox(label='Select the grouping variable',
                                     options=column_names)
        if self.category:
            self.categories = self.df[self.category].unique()
            self.n_categories = len(self.categories)
        else:
            self.category = None

        return self.category, self.categories, self.n_categories

    def download_table(self) -> object:
        # save table as xlsx
        # download the data
        output = BytesIO()

        # Write files to in-memory strings using BytesIO
        self.table.to_excel(output, sheet_name='Sheet1', index=False, header=True)
        output.seek(0)

        button = st.download_button(
            label="Download Excel workbook",
            data=output.getvalue(),
            file_name="table.xlsx",
            mime="application/vnd.ms-excel"
        )
        return button

    @property
    def descriptive_nums(self):
        self.table = pd.DataFrame()
        for col in self.df.select_dtypes(exclude='object').columns:
            mean_std = f"{self.df[col].mean():.2f} ± {self.df[col].std(ddof=0):.2f}"
            df_new_row = pd.DataFrame.from_records([{'variable_name': col,
                                                     'mean ± standard deviation': mean_std}])
            self.table = pd.concat([self.table, df_new_row], ignore_index=True, sort=False)
            del mean_std

        return self.table

    def descriptive_cats(self):
        for col in self.df.select_dtypes('object').columns:
            n_unique: Counter[Any] = c(self.df[col])
            uniques = self.df[col].unique()

            for i in uniques:
                s = f"{i}: n= {n_unique[i]} ({n_unique[i] / len(self.df) * 100:.2f}%) \n"
                freq = pd.DataFrame.from_records([{'variable_name': col,
                                                   'n (%)': s}])
                self.table = pd.concat([self.table, freq])

        return self.table

    @property
    def descriptive_multiple_nums(self):
        self.table = pd.DataFrame()
        self.table['variable_name'] = self.df.select_dtypes(exclude='object').columns
        for index, cat in enumerate(self.categories):
            for col in self.df.select_dtypes(exclude='object').columns:
                self.table.loc[self.table.variable_name == col, f"{self.category}: {cat} \n mean ± sd "] = \
                    f"{self.df[self.df[self.category] == cat][col].mean():.2f} ± " \
                    f"{self.df[self.df[self.category] == cat][col].std():.2f}"

        return self.table

    def descriptive_multiple_cats(self):
        self.table = pd.DataFrame()
        for index, cat in enumerate(self.categories):
            for col in self.df.select_dtypes('object').columns:
                n_unique = c(self.df[self.df[self.category] == cat][col])
                uniques = self.df[self.df[self.category] == cat][col].unique()

                for i in uniques:
                    s = f"{i}: n= {n_unique[i]} ({n_unique[i] / len(self.df) * 100:.2f}%) \n"
                    freq = pd.DataFrame.from_records([{'variable_name': col,
                                                       f"{self.category} = {cat}: \n n (%)": s}])
                    self.table = pd.concat([self.table, freq])

        return self.table

    def calculate_p(self) -> object:
        p_value = []
        for col in self.df.select_dtypes(include='number').columns:
            p_value.append(
                self.test(self.df[self.df[self.category] == self.categories[0]][col],
                          self.df[self.df[self.category] == self.categories[1]][col])[1])

        self.table['p value'] = p_value
        return self.table

    def calculate_multiple_p(self) -> object:
        data_list = []
        for col in self.df.select_dtypes(include='number').columns:
            for cat in self.categories:
                data_list.append(self.df[self.df[self.category] == cat][col])
            p = self.test(*data_list)[1]

        self.table['p_value'] = p
        return self.table

    def compare_groups(self):
        # reset table df
        self.table = pd.DataFrame()
        # select object columns
        var = self.category, self.categories, self.n_categories = self.define_category

        # define categories and n_categories
        if self.category is None:
            pass
        else:
            self.categories = self.df[self.category].unique()
            self.n_categories = len(self.categories)

        if self.category is None:
            raise ValueError("You need a grouping variable to compare means")
        elif self.n_categories <= 1:
            raise ValueError(
                f"Category classes must be other than 1. The classes in {self.category} : "
                f"{self.df[self.category].unique()}")
        else:
            if self.n_categories == 2:
                tests = {'Student t test (parametric)': ttest_ind,
                         'Mann Whitney U test (non-parametric)': mannwhitneyu}
                selection = st.selectbox(label='Select the test you want to run',
                                         options=['Student t test (parametric)',
                                                  'Mann Whitney U test (non-parametric)'])

                self.test = tests[selection]
                self.table = self.descriptive_multiple_nums
                self.calculate_p()

                return self.table

            else:
                tests = {'ANOVA (parametric)': f_oneway,
                         'Kruskal Wallis (non-parametric)': kruskal}
                selection = st.selectbox(label='Select the test you want to run',
                                         options=['ANOVA (parametric)',
                                                  'Kruskal Wallis (non-parametric)'])

                self.test = tests[selection]
                self.table = self.descriptive_multiple_nums
                self.table = self.calculate_multiple_p()
                return self.table

    @property
    def test_treatment(self):
        """
        A function to apply hypothesis testing on the pre-post treatment data
        :return: p values and summary table
        """

        st.write('#### Imporant ')
        st.write('In this section you are required to choose paired columns.Please select them in order to prevent '
                 'mis-comparisons. Ie. BMI1 in col1, BMI2 should be in col2 in the same indices')
        col1 = st.multiselect(label='Select the pre treatment columns',
                              options=self.df.columns)
        col2 = st.multiselect(label='Select the post treatment columns',
                              options=self.df.columns)

        if len(col1) != len(col2):
            raise ValueError('Length mismatch. Two selections must contain same amount of columns')
        else:
            for i in range(len(col1)):
                self.paired_columns.append((col1[i], col2[i]))

        tests = {'Paired samples t test': ttest_rel,
                 'Wilcoxon': wilcoxon}
        selection = st.selectbox(label='Select the test you want to run',
                                 options=['Paired samples t test',
                                          'Wilcoxon'])

        self.test = tests[selection]
        for i in range(len(self.paired_columns)):
            mean1 = f"{self.df[self.paired_columns[i][0]].mean():.2f} ± {self.df[self.paired_columns[i][0]].std(ddof=0):.2f}"
            mean2 = f"{self.df[self.paired_columns[i][1]].mean():.2f} ± {self.df[self.paired_columns[i][1]].std(ddof=0):.2f}"
            new_row = pd.DataFrame.from_dict({'Variable': self.paired_columns[i],
                                              'Pre mean ± std:': f"{mean1}",
                                              'Post mean ± std:': f"{mean2}",
                                              'p': f"{self.test(self.df[self.paired_columns[i][0]], self.df[self.paired_columns[i][1]])[1]:.2f}"})
            self.table = pd.concat([self.table, new_row])
            self.table.drop_duplicates(subset=['Pre mean ± std:', 'Post mean ± std:'], inplace=True)

        return self.table


    @property
    def test_with_control(self):
        # get unique values of controlling parameter
        self.category, self.categories, self.n_categories = self.define_category

        for i in range(self.n_categories):
            self.table = pd.concat([self.table, self.test_treatment])

        return self.table


    def post_hoc(self):
        st.write("Post-hoc tests will be implemented in the future versions")
        st.write('#### Imporant ')
        st.write('In this section you are required to choose paired columns.Please select them in order to prevent '
                 'mis-comparisons. Ie. BMI1 in col1, BMI2 should be in col2 in the same indices')
        col1 = st.multiselect(label='Select the pre treatment columns',
                              options=self.df.columns)
        col2 = st.multiselect(label='Select the post treatment columns',
                              options=self.df.columns)

        self.paired_columns = []

        if len(col1) != len(col2):
            raise ValueError('Length mismatch. Two selections must contain same amount of columns')
        else:
            for i in range(len(col1)):
                self.paired_columns.append((col1[i], col2[i]))

        for col in self.df.select_dtypes(include='number').columns:
            for cat in self.categories:
                data_list.append(self.df[self.df[self.category] == cat][col])
            p = self.test(*data_list)[1]

        self.table['p_value'] = p
        return self.table