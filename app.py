import streamlit as st
import numpy as np
import pandas as pd
import os
import re
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

    
    # File uploaders
    mspr_file = st.sidebar.file_uploader("Upload MSPR file:", key=1)
    if mspr_file:
         # Can be used wherever a "file-like" object is accepted:
         mspr = load_data(mspr_file)
         #st.write(dataframe)

    course_desig_file = st.sidebar.file_uploader("Upload Course Designations file:", key=2)
    if course_desig_file:
         # Can be used wherever a "file-like" object is accepted:
         course_desig = load_data(course_desig_file)
         #st.write(dataframe)

    scholarships_file = st.sidebar.file_uploader("Upload Scholarships file:", key=3)
    if scholarships_file:
         # Can be used wherever a "file-like" object is accepted:
         scholarships = load_data(scholarships_file)
         #st.write(dataframe)

    gpa_file = st.sidebar.file_uploader("Upload GPAs file:", key=4)
    if gpa_file:
         # Can be used wherever a "file-like" object is accepted:
         gpas = load_data(gpa_file)
         #st.write(dataframe)

    sat_file = st.sidebar.file_uploader("Upload SAT/ACT file:", key=5)
    if sat_file:
         # Can be used wherever a "file-like" object is accepted:
         sat_act = load_data(sat_file)
         #st.write(dataframe)

    tests_file = st.sidebar.file_uploader("Upload AP-IB-AICE file:", key=6)
    if tests_file:
         # Can be used wherever a "file-like" object is accepted:
         tests = load_data(tests_file)
         #st.write(dataframe)

    rank_file = st.sidebar.file_uploader("Upload HS Ranks file:", key=7)
    if rank_file:
         # Can be used wherever a "file-like" object is accepted:
         rank = load_data(rank_file)
         #st.write(dataframe)




    # Analysis button
    run_analysis = st.sidebar.button('Run analysis')

    if run_analysis:
        st.session_state.button_pressed = True

    if not mspr_file or not course_desig_file or not scholarships_file or not gpa_file or not sat_file or not tests_file or not rank_file or not run_analysis:
        st.markdown("### Text describing dataset schemas/formats will go here")
        st.markdown("texttexttexttexttextext")
        st.markdown("blahblahblahblah")


    # ======================== #
    # Code to run after all files uploaded and user hit "Run Analysis" button

    if st.session_state['button_pressed'] and mspr_file and course_desig_file and scholarships_file and gpa_file and sat_file and tests_file and rank_file:

        # Munging MSPR data

        mspr.rename(columns={"STUDENT_ID":"ID"}, inplace=True)

        # Encoding indicators as binary, 0 or 1
        mspr['MSPR_COMPL_IND'] = np.where(mspr['MSPR_COMPL_IND'] == "Y", 1, 0)

        mspr['NO CONCERNS'] = np.where(mspr['NO CONCERNS'] == "Y", 1, 0)

        mspr['ATTENDANCE'] = np.where(mspr['ATTENDANCE'] == "Y", 1, 0)

        mspr['LOW PARTICIPATION'] = np.where(mspr['LOW PARTICIPATION'] == "Y", 1, 0)

        mspr['LATE/MISSING ASSIGNMENTS'] = np.where(mspr['LATE/MISSING ASSIGNMENTS'] == "Y", 1, 0)

        mspr['OTHER ASSIGNMENTS CONCERNS'] = np.where(mspr['OTHER ASSIGNMENTS CONCERNS'] == "Y", 1, 0)

        mspr['LOW TEST SCORES'] = np.where(mspr['LOW TEST SCORES'] == "Y", 1, 0)

        mspr['DANGER of UNSATING'] = np.where(mspr['DANGER of UNSATING'] == "Y", 1, 0)

        mspr['COMMENT TEXT'] = np.where(mspr['COMMENT TEXT'] == "Y", 1, 0)

        # Calculating contract criteria percent
        mspr['contract_criteria_percent'] = mspr['CRITERIA'].str.extract('([0-9]+\.?[0-9]?\s?\/\s?[0-9]\.?[0-9]?)')

        mspr['contract_criteria_percent'] = mspr['contract_criteria_percent'].str.split('/').str[0].astype(float)/mspr['contract_criteria_percent'].str.split('/').str[1].astype(float)

        mspr.rename(columns={'ID':"STUDENT_ID"}, inplace=True)

        # mspr.tail()


        # ======================== #

        # Munging Course Designations data

        # Extract course_level from CRS_NUMB
        course_desig['COURSE_LEVEL'] = [int(str(x)[0]) for x in course_desig.CRS_NUMB.tolist()]
        # Group course_level 5 & 6 values in with 4
        course_desig['COURSE_LEVEL'] = np.where(course_desig['COURSE_LEVEL'].gt(4), 4, course_desig['COURSE_LEVEL'])

        # Create Dummies for Course Divisions
        top_n = ['New College Of Florida', 'Humanities', 'Natural Science', 'Other', 'Social Sciences']
        course_desig['CRS_DIVS_DESC'] = np.where(course_desig['CRS_DIVS_DESC'].isin(top_n), course_desig['CRS_DIVS_DESC'], "Other")

        # Encode CRS_DIVS_DESC as one-hot variables
        for n in top_n:
            dummy_colname = n.replace(" ", "_")
            dummy_colname = "DIVS_" + dummy_colname
            course_desig[dummy_colname] = np.where(course_desig['CRS_DIVS_DESC'] == n, 1, 0)
            
        # Drop Divison column after dummies have been created
        course_desig.drop(columns = "CRS_DIVS_DESC", inplace = True)


        ### CONTRACT DESIGNATIONS  SECTION ###

        # extract contract number
        contract_desig = course_desig.loc[course_desig['CLASS_TITLE'].str.contains('Contract ')]
        contract_desig['contract_number'] = contract_desig['CLASS_TITLE'].str.extract('(\d+)')[0]

        course_desig = course_desig.rename(columns={'SQ_COUNT_STUDENT_ID':'STUDENT_ID',
                                    'ACAD_HIST_GRDE_DESC':'course_grade'})

        course_desig = course_desig[['STUDENT_ID','TERM','CRN', 'COURSE_LEVEL', 'course_grade',
                                    'DIVS_New_College_Of_Florida', 'DIVS_Humanities',
                                    'DIVS_Natural_Science', 'DIVS_Other', 'DIVS_Social_Sciences']]

        contract_desig = contract_desig.rename(columns={'SQ_COUNT_STUDENT_ID':'STUDENT_ID'})
        contract_desig = contract_desig[['STUDENT_ID','TERM','contract_number']]
        contract_desig = contract_desig.drop_duplicates(subset=['STUDENT_ID','TERM'])

        # contract_desig.tail()
        

        # ======================= #

        # Munging Scholarships data

        scholarships.rename(columns={'TermCode':'TERM', 'SPRIDEN_ID':'STUDENT_ID'}, inplace=True)
        # Filter to only accepted scholarships
        scholarships = scholarships[~scholarships['Accept_Date'].isna()]

        # Group and sum scholarships by term/student
        final_scholarships = scholarships.groupby(["STUDENT_ID", 'TERM'])['FORMATTED_PAID_AMT'].agg(sum).reset_index(name='TOTAL_FUNDS')

        # final_scholarships.tail()



        # ======================= #

        # Munging GPA data

        gpas = gpas.rename(columns={'SPRIDEN_ID':'STUDENT_ID'})

        
        # ======================= #

        # Munging SAT/ACT data

        filter_tests = ['AE', 'ARE', 'AS', 'AM',
               'S2M','S2RW']
        sat_act = sat_act.loc[sat_act.TEST_CODE.isin(filter_tests)]

        cats = {'TEST_CODE':{
            'AE': 'ACT',
            'ARE': 'ACT',
            'AS': 'ACT',
            'AM': 'ACT',
            'S2M': 'SAT',
            'S2RW': 'SAT'
        }}

        sat_act = sat_act.replace(cats)

        act = sat_act.loc[sat_act.TEST_CODE=='ACT']

        # Store student/demo_time pairs with less than 4 ACT scores
        grouped_act = act.groupby(['DEMO_TIME_FRAME','SPRIDEN_ID']).size().reset_index()
        to_remove = grouped_act.loc[grouped_act[0] <4][['DEMO_TIME_FRAME','SPRIDEN_ID']]

        act = act.groupby(['SPRIDEN_ID','DEMO_TIME_FRAME']).sum('TEST_SCORE_N').reset_index()

        # Remove student/time pairs with less than 4 ACT scores
        act = pd.merge(act, to_remove, on=['SPRIDEN_ID','DEMO_TIME_FRAME'], 
                how='outer', indicator=True).query('_merge=="left_only"').drop('_merge',axis=1)

        # Take highest score
        # Could sub for highest or average
        act = act.loc[act.groupby('SPRIDEN_ID').TEST_SCORE_N.idxmax()]
        act['TEST_SCORE_N'] = round(act['TEST_SCORE_N']/4)

        # Safe guard so no scores are under 9
        act = act.loc[act.TEST_SCORE_N>=9]

        encodings = {'TEST_SCORE_N': {
            36 : 1590, 35 : 1540, 34 : 1500,
            33 : 1460, 32 : 1430, 31 : 1400, 
            30 : 1370, 29 : 1340, 28 : 1310,
            27 : 1280, 26 : 1240, 25 : 1210,
            24 : 1180, 23 : 1140, 22 : 1110,
            21 : 1080, 20 : 1040, 19 : 1010,
            18 : 970, 17 : 930, 16 : 890,
            15 : 850, 14 : 800, 13 : 760,
            12 : 710, 11 : 670, 10 : 630,
            9 : 590    
        }}

        act = act.replace(encodings)

        sat = sat_act.loc[sat_act.TEST_CODE=="SAT"]

        # Store student/demo_time pairs with less than 2 SAT scores
        grouped_sat = sat.groupby(['DEMO_TIME_FRAME','SPRIDEN_ID']).size().reset_index()
        to_remove = grouped_sat.loc[grouped_sat[0] <2][['DEMO_TIME_FRAME','SPRIDEN_ID']]

        sat = sat.groupby(['SPRIDEN_ID','DEMO_TIME_FRAME']).sum('TEST_SCORE_N').reset_index()

        # Remove student/time pairs with less than 2 SAT scores
        sat = pd.merge(sat, to_remove, on=['SPRIDEN_ID','DEMO_TIME_FRAME'], 
                how='outer', indicator=True).query('_merge=="left_only"').drop('_merge',axis=1)

        # Take latest score
        # Could sub for highest or average
        sat = sat.loc[sat.groupby('SPRIDEN_ID').DEMO_TIME_FRAME.idxmax()]

        sat_final = pd.concat([sat,act])

        sat_final = sat_final.groupby('SPRIDEN_ID').max('TEST_SCORE_N').reset_index()

        sat_final = sat_final.drop('DEMO_TIME_FRAME',axis=1)

        sat_final = sat_final.rename(columns={'SPRIDEN_ID':'STUDENT_ID'})

        # sat_final.tail()



        # ======================= #

        # Munging AP/IB/AICE data

        tests.rename(columns={'SPRIDEN_ID':'STUDENT_ID','SWVLACC_CLASS_TITLE':'AP_IB_CLASS_TITLE', 'AICE/AP/IB Indicator':'AP_IB_TEST_FLAG'},inplace=True)
        tests['AP_IB_TEST_FLAG'] = np.where(tests['AP_IB_TEST_FLAG'] == "Y", 1,0)
        tests = tests[tests['AP_IB_CLASS_TITLE'].str.contains(pat="AP|IB|AICE")]
        tests = tests.drop_duplicates(subset=['STUDENT_ID'])
        tests.drop(columns="AP_IB_CLASS_TITLE", inplace=True)
        # tests.tail()

        # ======================= #

        # Munging HS Rank data

        rank.rename(columns={'SPRIDEN_ID': 'STUDENT_ID'}, inplace=True)
        rank.drop_duplicates(inplace=True)

        rank["RANK_PERCENTILE"] = 1-(rank['SORHSCH_CLASS_RANK']/rank['SORHSCH_CLASS_SIZE'])

        rank = rank[['STUDENT_ID','RANK_PERCENTILE']]
        # rank.tail()


        # ======================= #

        # Combinging dfs

        mspr = pd.merge(mspr, course_desig, on=['STUDENT_ID','TERM','CRN'], how="left")

        mspr = pd.merge(mspr, contract_desig, on=['STUDENT_ID','TERM'], how="left")

        mspr = pd.merge(mspr, final_scholarships, on=['STUDENT_ID','TERM'], how="left")

        mspr = pd.merge(mspr, rank, on=['STUDENT_ID'], how="left")

        mspr = pd.merge(mspr, tests, on=['STUDENT_ID'], how="left")

        mspr = pd.merge(mspr, sat_final, on=['STUDENT_ID'], how="left")

        mspr = pd.merge(mspr, gpas, on=['STUDENT_ID'], how="left")

        mspr = mspr.replace({'TRM_DESC': {'1':1, '1MC':0.5, 'M1':0.5}})

        # Calculate class per student/semester pair
        mspr['total_classes'] = mspr.groupby(['TERM','STUDENT_ID'])['TRM_DESC'].transform('sum')

        # Impute students without AP/IB and scholarships with 0
        mspr['AP_IB_TEST_FLAG'] = mspr['AP_IB_TEST_FLAG'].fillna(0)
        mspr['TOTAL_FUNDS'] = mspr['TOTAL_FUNDS'].fillna(0)


        # Course-level dataframe
        features = ['STUDENT_ID','CRN','TERM', 'MSPR_COMPL_IND','TITLE','INSTRUCTOR',
       'ATTENDANCE', 'LOW PARTICIPATION', 'LATE/MISSING ASSIGNMENTS',
       'OTHER ASSIGNMENTS CONCERNS', 'LOW TEST SCORES', 'DANGER of UNSATING',
       'contract_criteria_percent',
       'contract_number', 'TOTAL_FUNDS', 'RANK_PERCENTILE',
       'AP_IB_TEST_FLAG', 'TEST_SCORE_N', 'GPA_HIGH_SCHOOL', 'total_classes',
       'COURSE_LEVEL', 'DIVS_Humanities',
       'DIVS_Natural_Science', 'DIVS_Other', 'DIVS_Social_Sciences','NO CONCERNS']

        mspr_bycourse = mspr[features]

        st.write(mspr_bycourse.columns)

        # Contract-level aggregation
        mspr_bycontract = mspr_bycourse.drop(columns = ['CRN','TITLE','INSTRUCTOR','NO CONCERNS']).groupby(['STUDENT_ID','TERM']).agg({
                                                           'MSPR_COMPL_IND':'sum',
                                                           'ATTENDANCE':'sum',
                                                           'LOW PARTICIPATION':'sum',
                                                           'LATE/MISSING ASSIGNMENTS':'sum',
                                                           'OTHER ASSIGNMENTS CONCERNS':'sum',
                                                           'LOW TEST SCORES':'sum',
                                                           'DANGER of UNSATING':'sum',
                                                           'contract_criteria_percent':'max',
                                                           'contract_number':'max',
                                                           'TOTAL_FUNDS':'max',
                                                            'RANK_PERCENTILE':'max',
                                                            'AP_IB_TEST_FLAG':'max',
                                                            'TEST_SCORE_N':'max',
                                                            'GPA_HIGH_SCHOOL':'max',
                                                            'total_classes':'max',
                                                            'COURSE_LEVEL':'mean',
                                                            'DIVS_Humanities':'sum',
                                                            'DIVS_Natural_Science':'sum',
                                                            'DIVS_Other':'sum',
                                                            'DIVS_Social_Sciences':'sum',                                                    
                                                                                                     }).reset_index()






        # ==================================================== #

        # Preparing MSPR course df for plotting

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

        st.write(mspr_plotting)
        st.write(mspr_old)


        #st.write(mspr_plotting)
        #st.write(mspr_current)




        # ==================================================== #



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
        mspr_term_level = mspr_plotting[['STUDENT_ID','TERM','TOTAL_FUNDS','RANK_PERCENTILE','GPA_HIGH_SCHOOL','TEST_SCORE_N']].groupby(['STUDENT_ID','TERM']).agg({
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




        # ==================================================== #

        # LightGBM predictions

        # Imputing for predictions

        # Subset current term for prediction
        mspr_course_current = mspr_bycourse.loc[mspr_bycourse.TERM == mspr_bycourse.TERM.max()]
        mspr_contract_current = mspr_bycontract.loc[mspr_bycontract.TERM == mspr_bycontract.TERM.max()]


        # Subset features
        mspr_course_current = mspr_course_current[['STUDENT_ID', 'CRN', 'TERM', 'MSPR_COMPL_IND', 'ATTENDANCE', 
        'LOW PARTICIPATION', 'LATE/MISSING ASSIGNMENTS', 'OTHER ASSIGNMENTS CONCERNS', 
        'LOW TEST SCORES', 'DANGER of UNSATING', 'contract_criteria_percent', 
        'contract_number', 'TOTAL_FUNDS', 'RANK_PERCENTILE', 'AP_IB_TEST_FLAG', 'TEST_SCORE_N',
        'GPA_HIGH_SCHOOL', 'total_classes' ,'COURSE_LEVEL', 'DIVS_Humanities', 
        'DIVS_Natural_Science', 'DIVS_Other', 'DIVS_Social_Sciences', 'course_grade', 'TITLE','INSTRUCTOR']]

        mspr_contract_current = mspr_contract_current[['STUDENT_ID', 'TERM', 'MSPR_COMPL_IND', 'ATTENDANCE', 
        'LOW PARTICIPATION', 'LATE/MISSING ASSIGNMENTS', 'OTHER ASSIGNMENTS CONCERNS', 
        'LOW TEST SCORES', 'DANGER of UNSATING', 'contract_criteria_percent', 
        'contract_number', 'TOTAL_FUNDS', 'RANK_PERCENTILE', 'AP_IB_TEST_FLAG', 'TEST_SCORE_N',
        'GPA_HIGH_SCHOOL', 'total_classes' ,'COURSE_LEVEL', 'DIVS_Humanities', 
        'DIVS_Natural_Science', 'DIVS_Other', 'DIVS_Social_Sciences', 'contract_grade']]

        # Take IDs for prediction output
        mspr_course_ids = mspr_course_current[['STUDENT_ID','CRN','TITLE','INSTRUCTOR']]
        mspr_contract_ids = mspr_contract_current[['STUDENT_ID']]

        mspr_course_current = mspr_course_current.drop(columns=['STUDENT_ID','CRN','TERM','course_grade','TITLE','INSTRUCTOR'])
        mspr_contract_current = mspr_contract_current.drop(columns=['STUDENT_ID','TERM','contract_grade'])



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


        # Download buttons
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
    my_dict["Low Attendance"] = (df['ATTENDANCE'].value_counts() / len(df))[1]*100
    my_dict["Low Particpation"] = (df['LOW PARTICIPATION'].value_counts() / len(df))[1]*100
    my_dict["Late/Missing Assignments"] = (df['LATE/MISSING ASSIGNMENTS'].value_counts() / len(df))[1]*100
    my_dict["Other Assignment Concerns"] = (df['OTHER ASSIGNMENTS CONCERNS'].value_counts() / len(df))[1]*100
    my_dict["Low Test Scores"] = (df['LOW TEST SCORES'].value_counts() / len(df))[1]*100
    my_dict["Danger of Unsatting"] = (df['DANGER of UNSATING'].value_counts() / len(df))[1]*100
    my_dict["Has Concern(s)"] = 100-(df['NO CONCERNS'].value_counts() / len(df))[1]*100
    #mspr_dict["MSPR Completed"] = (mspr_compl['MSPR_COMPL_IND'].value_counts() / len(mspr_compl))[1]*100
    return my_dict



if __name__ == "__main__":
    main()

