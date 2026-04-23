import pandas as pd

def preprocess_data(df):

    # 1. Remove duplicates
    df = df.drop_duplicates()

    # 2. Handle missing values
    df['person_emp_length'] = df['person_emp_length'].fillna(df['person_emp_length'].median())
    df['loan_int_rate'] = df['loan_int_rate'].fillna(df['loan_int_rate'].median())

    # 3. Capping
    df['person_emp_length'] = df['person_emp_length'].clip(0, 60)
    df['person_age'] = df['person_age'].clip(18, 100)

    # 4. Encode binary
    df['cb_person_default_on_file'] = df['cb_person_default_on_file'].map({'N': 0, 'Y': 1})

    # 5. Encode ordinal
    grade_mapping = {'A':7,'B':6,'C':5,'D':4,'E':3,'F':2,'G':1}
    df['loan_grade'] = df['loan_grade'].map(grade_mapping)

    # 6. One-hot encoding
    df = pd.get_dummies(df, columns=[
        'person_home_ownership',
        'loan_intent'
    ], drop_first=True)

    return df