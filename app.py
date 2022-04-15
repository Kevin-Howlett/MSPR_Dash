import streamlit as st
import numpy as np
import pandas as pd
import os
import plotly.express as px
from io import StringIO
import datetime
from sklearn.impute import SimpleImputer
from lightgbm import LGBMClassifier
import pickle



# ======================== #
st.set_page_config(
    page_title="Predicting NCF Academic Success",
    layout="wide",
    initial_sidebar_state="expanded",
)

PAGE_STYLE = """
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    </style>
    """

st.markdown(PAGE_STYLE, unsafe_allow_html=True)
# ======================== #


def main():

    if 'button_pressed' not in st.session_state:
        st.session_state['button_pressed'] = False

    # Main panel
    st.title("Predicting NCF Academic Success")
    

    # ======================== #

    # Side panel
    st.sidebar.title("Data Upload")

    

    mspr_file = st.sidebar.file_uploader("Upload MSPR course file:", key=1)
    if mspr_file:
         # Can be used wherever a "file-like" object is accepted:
         mspr_bycourse = load_data(mspr_file)
         #st.write(dataframe)

    mspr_file2 = st.sidebar.file_uploader("Upload MSPR contract file:", key=2)
    if mspr_file2:
         # Can be used wherever a "file-like" object is accepted:
         mspr_bycontract = load_data(mspr_file2)
         #st.write(dataframe)


    # Analysis button
    run_analysis = st.sidebar.button('Run analysis')

    if run_analysis:
        st.session_state.button_pressed = True

    if not mspr_file or not mspr_file2 or not run_analysis:
        st.markdown("### Text describing dataset schemas/formats will go here")
        st.markdown("texttexttexttexttextext")
        st.markdown("blahblahblahblah")


    # ======================== #
    # Output after running analysis

    if st.session_state['button_pressed'] and mspr_file2 and mspr_file:

        st.write('Analysis Complete!')


        # MSPR Plotting DF
        mspr_plotting = mspr_bycourse.copy()

        # Convert TERM to date type
        mspr_plotting['TERM'] = pd.to_datetime(mspr_plotting['TERM'], format='%Y%m', errors='coerce').dt.date

        # Store variable with current term
        current_term = mspr_plotting.TERM.max()

        # Store new dfs for plotting MSPR where completed
        mspr_current = mspr_plotting.loc[(mspr_plotting.MSPR_COMPL_IND == 1) & (mspr_plotting.TERM == current_term)]
        mspr_old = mspr_plotting.loc[(mspr_plotting.MSPR_COMPL_IND == 1) & (mspr_plotting.TERM != current_term)]

        # Dropping outliers
        treatoutliers(mspr_plotting, columns = ['GPA_HIGH_SCHOOL', 'TOTAL_FUNDS'])


        #st.write(mspr_plotting)
        #st.write(mspr_current)


        # Plotting

        # Create dicts for mspr bar charts

        mspr_old_dict = create_mspr_dict(mspr_old)
        mspr_current_dict = create_mspr_dict(mspr_current)

        # Plot MSPR bar charts

        mspr_plot_old = px.bar(x=mspr_old_dict.keys(), y=[i for i in mspr_old_dict.values()],
                text=[round(i,2) for i in mspr_old_dict.values()],
                labels={
                    "x": "Indicator",
                    "y": "Percentage of Students w/ Indicator"
                },
                title='MSPR Indicators for Previous Terms',
                width = 600, height = 500)

        mspr_plot_old.update_traces(width=0.8)
        mspr_plot_old.update_xaxes(type='category', categoryorder="total ascending")
        mspr_plot_old.update_layout(yaxis_ticksuffix = '%')


        mspr_plot = px.bar(x=mspr_current_dict.keys(), y=[i for i in mspr_current_dict.values()],
                text=[round(i,2) for i in mspr_current_dict.values()],
                labels={
                    "x": "Indicator",
                    "y": "Percentage of Students w/ Indicator"
                },
                title='MSPR Indicators for Current Term',
                width = 600, height = 500)

        mspr_plot.update_traces(width=0.8)
        mspr_plot.update_xaxes(type='category', categoryorder="total ascending")
        mspr_plot.update_layout(yaxis_ticksuffix = '%')
        #st.write(mspr_plot)


        # Title for MSPR Plotting
        st.write("## MSPR Features")

        # First row of plots
        col1, col2 = st.columns(2)

        col1.header("")
        col1.plotly_chart(mspr_plot_old, use_column_width=True)

        col2.header("")
        col2.plotly_chart(mspr_plot, use_column_width=True)


        
        # Mspr metrics comparing current term to previous terms
        st.markdown('#### Metrics for Current Term')

        col1, col2, col3, col4, col5, col6, col7 = st.columns(7)

        mspr_metrics = []
        for k in mspr_current_dict.keys():
            l = []
            l.append(k)
            l.append(mspr_current_dict[k])
            l.append(mspr_current_dict[k]-mspr_old_dict[k])
            mspr_metrics.append(l)


        cols = [col1, col2, col3, col4, col5, col6, col7]
        for i in range(len(cols)):
            if mspr_metrics[i][0] == 'No Concerns':
                delta_color = 'normal'
            else:
                delta_color = 'inverse'
            cols[i].metric(label=mspr_metrics[i][0], 
                value=str(round(mspr_metrics[i][1], 1))+"%",
                delta=str(round(mspr_metrics[i][2], 2)),
                delta_color=delta_color)


        # Term selector

        # terms = st.multiselect(
        #     'Select terms to plot',
        #     mspr_plotting.TERM.unique().tolist(),
        #     mspr_plotting.TERM.unique().tolist())


        # mspr_plotting = mspr_plotting.loc[mspr_plotting.TERM.isin(terms)]

        # Aggregate to term-level (for each student)for plotting changes each term
        mspr_term_level = mspr_plotting[['IR_ID','TERM','TOTAL_FUNDS','RANK_PERCENTILE','GPA_HIGH_SCHOOL','TEST_SCORE_N']].groupby(['IR_ID','TERM']).agg({
            'TOTAL_FUNDS':'max',
            'RANK_PERCENTILE':'max',
            'GPA_HIGH_SCHOOL':'max',
            'TEST_SCORE_N':'max'
            }).reset_index()

        # Aggregate data for time-series
        mspr_time_series_df = mspr_plotting[['TERM','TOTAL_FUNDS','RANK_PERCENTILE','GPA_HIGH_SCHOOL','TEST_SCORE_N']].groupby('TERM').agg({
            'TOTAL_FUNDS':'mean',
            'RANK_PERCENTILE':'mean',
            'GPA_HIGH_SCHOOL':'mean',
            'TEST_SCORE_N':'mean'
            }).reset_index()

        
        # Convert to percents
        mspr_time_series_df['RANK_PERCENTILE'] = mspr_time_series_df['RANK_PERCENTILE']*100
        mspr_term_level['RANK_PERCENTILE'] = mspr_term_level['RANK_PERCENTILE']*100

        # Write the table of averages over terms
        st.write("## Continuous/Numeric Features")
        st.write("#### Averages Over Terms")

        st.write(mspr_time_series_df.rename(columns={
            'TOTAL_FUNDS':'Avg Scholarship Amnt',
            'RANK_PERCENTILE':'Avg HS Rank',
            'GPA_HIGH_SCHOOL':'Avg HS GPA',
            'TEST_SCORE_N':'Avg SAT Score'
            }).round(2).astype(str))



        # Plot GPAs
        st.write("### High School GPA")

        # Plot GPA over time
        gpa_time_series = px.line(mspr_time_series_df, y="GPA_HIGH_SCHOOL", x='TERM',
                 labels={
                     "GPA_HIGH_SCHOOL": "GPA",
                     "TERM": "Date"
                 },
                title="High School GPA Average Over Time",
                width = 600, height = 500)

        # Plot GPA dist for current term
        gpa_current = px.histogram(mspr_term_level.loc[mspr_term_level.TERM == current_term], x="GPA_HIGH_SCHOOL",
                   marginal="box",
                 labels={
                     "GPA_HIGH_SCHOOL": "GPA",
                     "count": "Frequency"
                 },
                title="High School GPA (Current Term)",
                width = 600, height = 500)




        # Second row of plots
        col1, col2 = st.columns(2)

        col1.header("")
        col1.plotly_chart(gpa_time_series, use_column_width=True)

        col2.header("")
        col2.plotly_chart(gpa_current, use_column_width=True)




        # Plot HS ranks
        st.write("### High School Rank")

        # Plot Rank over time
        rank_time_series = px.line(mspr_time_series_df, y="RANK_PERCENTILE", x='TERM',
                 labels={
                     "RANK_PERCENTILE": "Rank Percentile",
                     "TERM": "Date"
                 },
                title="High School Rank Average Over Time",
                width = 600, height = 500)
        rank_time_series.update_layout(yaxis_ticksuffix = '%')

        # Plot Rank dist for current term
        rank_current = px.histogram(mspr_term_level.loc[mspr_term_level.TERM == current_term], x="RANK_PERCENTILE",
                   marginal="box",
                 labels={
                     "RANK_PERCENTILE": "Rank Percentile",
                     "count": "Frequency"
                 },
                title="High School Rank (Current Term)",
                width = 600, height = 500)
        rank_current.update_layout(xaxis_ticksuffix = '%')




        # Third row of plots
        col1, col2 = st.columns(2)

        col1.header("")
        col1.plotly_chart(rank_time_series, use_column_width=True)

        col2.header("")
        col2.plotly_chart(rank_current, use_column_width=True)





        # Plot SAT Scores
        st.write("### SAT Scores")

        # Plot SAT over time
        sat_time_series = px.line(mspr_time_series_df, y="TEST_SCORE_N", x='TERM',
                 labels={
                     "TEST_SCORE_N": "Score",
                     "TERM": "Date"
                 },
                title="SAT Score Average Over Time",
                width = 600, height = 500)

        # Plot SAT dist for current term
        sat_current = px.histogram(mspr_term_level.loc[mspr_term_level.TERM == current_term], x="TEST_SCORE_N",
                   marginal="box",
                 labels={
                     "TEST_SCORE_N": "Score",
                     "count": "Frequency"
                 },
                title="SAT Scores (Current Term)",
                width = 600, height = 500)




        # Fourth row of plots
        col1, col2 = st.columns(2)

        col1.header("")
        col1.plotly_chart(sat_time_series, use_column_width=True)

        col2.header("")
        col2.plotly_chart(sat_current, use_column_width=True)





        # Plot Scholarships
        st.write("### Total Scholarships per Student/Term")

        # Plot Scholarships over time
        fund_time_series = px.line(mspr_time_series_df, y="TOTAL_FUNDS", x='TERM',
                 labels={
                     "TOTAL_FUNDS": "Scholarship Amount",
                     "TERM": "Date"
                 },
                title="Total Scholarship Average Over Time",
                width = 600, height = 500)
        fund_time_series.update_layout(yaxis_tickprefix = '$')

        # Plot Scholarship dist for current term
        fund_current = px.histogram(mspr_term_level.loc[mspr_term_level.TERM == current_term], x="TOTAL_FUNDS",
                   marginal="box",
                 labels={
                     "TOTAL_FUNDS": "Scholarship Amount",
                     "count": "Frequency"
                 },
                title="Total Scholarships per Student (Current Term)",
                width = 600, height = 500)
        fund_current.update_layout(xaxis_tickprefix = '$')




        # Fifth row of plots
        col1, col2 = st.columns(2)

        col1.header("")
        col1.plotly_chart(fund_time_series, use_column_width=True)

        col2.header("")
        col2.plotly_chart(fund_current, use_column_width=True)



        # ======================== #
        # Imputing for predictions

        # Subset current term for prediction
        mspr_course_current = mspr_bycourse.loc[mspr_bycourse.TERM == mspr_bycourse.TERM.max()]
        mspr_contract_current = mspr_bycontract.loc[mspr_bycontract.TERM == mspr_bycontract.TERM.max()]


        # Subset features
        mspr_course_current = mspr_course_current[['IR_ID', 'CRN', 'TERM', 'MSPR_COMPL_IND', 'ATTENDANCE', 
        'LOW PARTICIPATION', 'LATE/MISSING ASSIGNMENTS', 'OTHER ASSIGNMENTS CONCERNS', 
        'LOW TEST SCORES', 'DANGER of UNSATING', 'contract_criteria_percent', 
        'contract_number', 'TOTAL_FUNDS', 'RANK_PERCENTILE', 'AP_IB_TEST_FLAG', 'TEST_SCORE_N',
        'GPA_HIGH_SCHOOL', 'total_classes' ,'COURSE_LEVEL', 'DIVS_Humanities', 
        'DIVS_Natural_Science', 'DIVS_Other', 'DIVS_Social_Sciences', 'course_grade']]

        mspr_contract_current = mspr_contract_current[['IR_ID', 'TERM', 'MSPR_COMPL_IND', 'ATTENDANCE', 
        'LOW PARTICIPATION', 'LATE/MISSING ASSIGNMENTS', 'OTHER ASSIGNMENTS CONCERNS', 
        'LOW TEST SCORES', 'DANGER of UNSATING', 'contract_criteria_percent', 
        'contract_number', 'TOTAL_FUNDS', 'RANK_PERCENTILE', 'AP_IB_TEST_FLAG', 'TEST_SCORE_N',
        'GPA_HIGH_SCHOOL', 'total_classes' ,'COURSE_LEVEL', 'DIVS_Humanities', 
        'DIVS_Natural_Science', 'DIVS_Other', 'DIVS_Social_Sciences', 'contract_grade']]

        # Take IDs for prediction output
        mspr_course_ids = mspr_course_current[['IR_ID','CRN']]
        mspr_contract_ids = mspr_contract_current[['IR_ID']]

        mspr_course_current = mspr_course_current.drop(columns=['IR_ID','CRN','TERM','course_grade'])
        mspr_contract_current = mspr_contract_current.drop(columns=['IR_ID','TERM','contract_grade'])



        # Read in pickled imputers
        current_path = os.getcwd()

        course_imputer_path = os.path.join(current_path, 'static/course_imputer.pkl')
        with open(course_imputer_path, 'rb') as handle:
            course_imputer = pickle.load(handle)

        contract_imputer_path = os.path.join(current_path, 'static/contract_imputer.pkl')
        with open(contract_imputer_path, 'rb') as handle:
            contract_imputer = pickle.load(handle)

        # Imputing
        test_imputed = course_imputer.transform(mspr_course_current)
        mspr_course_current = pd.DataFrame(data = test_imputed,
            columns = mspr_course_current.columns)

        test_imputed = contract_imputer.transform(mspr_contract_current)
        mspr_contract_current = pd.DataFrame(data = test_imputed,
            columns = mspr_contract_current.columns)


        # Read in pickeled models
        course_model_path = os.path.join(current_path, 'static/course_model.pkl')
        with open(course_model_path, 'rb') as handle:
            course_model = pickle.load(handle)

        contract_model_path = os.path.join(current_path, 'static/contract_model.pkl')
        with open(contract_model_path, 'rb') as handle:
            contract_model = pickle.load(handle)

        # Predicting
        course_preds = course_model.predict_proba(mspr_course_current)
        contract_preds = contract_model.predict_proba(mspr_contract_current)

        # Take prob of unsat
        course_preds = [item[1] for item in course_preds]
        contract_preds = [item[1] for item in contract_preds]

        mspr_course_ids['Prob of Unsat'] = course_preds
        mspr_contract_ids['Prob of Unsat'] = contract_preds

        mspr_course_ids = mspr_course_ids.sort_values(by='Prob of Unsat', ascending=False)
        mspr_contract_ids = mspr_contract_ids.sort_values(by='Prob of Unsat', ascending=False)


        col1, col2 = st.columns(2)

        col1.header("Course-Level Predictions")
        col1.write(mspr_course_ids, use_column_width=True)

        col2.header("Contract-Level Predictions")
        col2.write(mspr_contract_ids, use_column_width=True)

        # st.write(mspr_course_ids)
        # st.write(mspr_contract_ids)
        
        # Convert preds to csv and download
        mspr_course_csv = mspr_course_ids.to_csv(index=False).encode('utf-8')
        mspr_contract_csv = mspr_contract_ids.to_csv(index=False).encode('utf-8')


        st.write('### Download Predictions')
        course_download = st.download_button(
            "Download Course-Level",
            mspr_course_csv,
            "course_preds.csv",
            "text/csv",
            key='download-course-csv'
            )
        contract_download = st.download_button(
            "Download Contract-Level",
            mspr_contract_csv,
            "contract_preds.csv",
            "text/csv",
            key='download-contract-csv'
            )



    return None






# ======================== #

# Functions

@st.cache(hash_funcs={StringIO: StringIO.getvalue}, suppress_st_warning=True)
def load_data(file_uploaded):
    if file_uploaded.name.split('.')[1] == 'csv':
        return pd.read_csv(file_uploaded, sep=',', encoding='utf-8')
    else:
        return pd.read_excel(file_uploaded)


def treatoutliers(df, columns=None, factor=1.5, method='IQR', treament='cap'):
    """
    Removes the rows from self.df whose value does not lies in the specified standard deviation
    :param columns:
    :param in_stddev:
    :return:
    """
#     if not columns:
#         columns = self.mandatory_cols_ + self.optional_cols_ + [self.target_col]
    if not columns:
        columns = df.columns
    
    for column in columns:
        if method == 'STD':
            permissable_std = factor * df[column].std()
            col_mean = df[column].mean()
            floor, ceil = col_mean - permissable_std, col_mean + permissable_std
        elif method == 'IQR':
            Q1 = df[column].quantile(0.25)
            Q3 = df[column].quantile(0.75)
            IQR = Q3 - Q1
            floor, ceil = Q1 - factor * IQR, Q3 + factor * IQR
        
        if treament == 'remove':
            df = df[(df[column] >= floor) & (df[column] <= ceil)]
        elif treament == 'cap':
            df[column] = df[column].clip(floor, ceil)
            
    return None

def create_mspr_dict(df):
    my_dict = dict()
    my_dict["Has Concern(s)"] = 100-(df['NO CONCERNS'].value_counts() / len(df))[1]*100
    my_dict["Low Attendance"] = (df['ATTENDANCE'].value_counts() / len(df))[1]*100
    my_dict["Low Particpation"] = (df['LOW PARTICIPATION'].value_counts() / len(df))[1]*100
    my_dict["Late/Missing Assignments"] = (df['LATE/MISSING ASSIGNMENTS'].value_counts() / len(df))[1]*100
    my_dict["Other Assignment Concerns"] = (df['OTHER ASSIGNMENTS CONCERNS'].value_counts() / len(df))[1]*100
    my_dict["Low Test Scores"] = (df['LOW TEST SCORES'].value_counts() / len(df))[1]*100
    my_dict["Danger of Unsatting"] = (df['DANGER of UNSATING'].value_counts() / len(df))[1]*100
    #mspr_dict["MSPR Completed"] = (mspr_compl['MSPR_COMPL_IND'].value_counts() / len(mspr_compl))[1]*100
    return my_dict



if __name__ == "__main__":
    main()

