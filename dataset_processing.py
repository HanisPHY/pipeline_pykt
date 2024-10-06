import pandas as pd

file_path = './data/gkt_new/cogedu_emb_raw.csv'

df1 = pd.read_csv(file_path)

print(df1.head())

df1['student_id'] = df1['student_id'].str[1:]

df1['unique_question_id'] = df1.groupby(['course_name', 'question_id', 'school']).ngroup() + 1
df1['unique_slide_id'] = df1.groupby(['course_name', 'slide_id', 'school']).ngroup() + 1

# eduAgnet dataset
# df1['unique_slide_id'] = df1.groupby(['course_name', 'slide_id']).ngroup() + 1
# df1['unique_question_id'] = df1.groupby(['course_name', 'question_id']).ngroup() + 1

df1 = df1[df1['test_type'] == 'post test']

df1_train = df1[df1['data_type'] == 'train']
df1_test = df1[df1['data_type'] == 'test']

df1_train.to_csv('cogedu_emb_raw_train.csv', index=False)
df1_test.to_csv('cogedu_emb_raw_test.csv', index=False)