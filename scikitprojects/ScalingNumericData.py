import pandas as pd
from sklearn import preprocessing

exam_data = pd.read_csv('scikitprojects/datasets/student_performance.csv')
print(pd.__version__)






#Finding Avg. of Standardized Data

# 1. Standardization
exam_data['total_score'] = preprocessing.scale(exam_data['total_score']) 
exam_data['class_participation'] = preprocessing.scale(exam_data['class_participation']) 
exam_data['attendance_percentage'] = preprocessing.scale(exam_data['attendance_percentage']) 
exam_data['weekly_self_study_hours'] = preprocessing.scale(exam_data['weekly_self_study_hours']) 

# 2. Normalization 
total_score_avg = exam_data['total_score'].mean()
class_participation_avg = exam_data['class_participation'].mean()
attendance_avg = exam_data['attendance_percentage'].mean()
weekly_self_study_hours_avg = exam_data['weekly_self_study_hours'].mean()


print(f"Avg. Total Score: {total_score_avg}    ")
print(f"Avg. Class Participation: {class_participation_avg}")
print(f"Avg. Attendance Percentage: {attendance_avg}")
print(f"Avg. Weekly Self Study Hours: {weekly_self_study_hours_avg}")


#Representing categorical data numerically
print("Representing categorical data numerically:")
le = preprocessing.LabelEncoder()
exam_data['grade'] = le.fit_transform(exam_data['grade'].astype(str))
print(exam_data.head(15))