
import pandas as pd
from sklearn.preprocessing import PowerTransformer


def preprocess_columns(df):
    """
    Assumptions:
    - Remove variables with more than 50% missing values
    - Replace missing values of numerical variables with per mean
    - Remove categorical variables with more than 25 unique values
    :return: df
    """

    mv_cols = df.columns[df.isnull().sum() / len(df) > 0.5]
    df.drop(mv_cols, axis=1, inplace=True)

    cols = df.columns
    num_cols = df._get_numeric_data().columns
    cat_cols = list(set(cols) - set(num_cols))

    if len(cat_cols) > 0:
        for cat_col in cat_cols:
            if len(df[cat_col].unique()) > 25:
                df.drop(cat_col, axis=1, inplace=True)

    cols = df.columns
    num_cols = df._get_numeric_data().columns
    cat_cols = list(set(cols) - set(num_cols))

    if len(cat_cols) > 0:
        for cat_col in cat_cols:
            df[cat_col] = df[cat_col].fillna(-1)

    if len(num_cols) > 0:
        for num_col in num_cols:
            df[num_col] = df[num_col].fillna(df[num_col].mean())

    return df


def load_water_quality_data():
    # https://www.kaggle.com/adityakadiwal/water-potability
    df = pd.read_csv('../data/water_potability.csv', sep=',')
    y_df = df['Potability']
    X_df = df.drop('Potability', axis=1)
    X_df = preprocess_columns(X_df)

    y_df = y_df.astype(int)
    y_word_dict = {1: 'Potable_yes', 0: 'Potable_no'}
    dataset = {
        'problem': 'classification',
        'full': {
            'X': X_df,
            'y': y_df,
        },
    }

    return dataset, y_word_dict


def load_stroke_data():
    # https://www.kaggle.com/fedesoriano/stroke-prediction-dataset
    df = pd.read_csv('../data/healthcare-dataset-stroke-data.csv', sep=',')
    y_df = df['stroke']
    X_df = df.drop('stroke', axis=1)

    X_df['hypertension'] = X_df['hypertension'].replace({1: "Yes", 0: "No"})
    X_df['heart_disease'] = X_df['heart_disease'].replace({1: "Yes", 0: "No"})

    cols = X_df.columns
    num_cols = X_df._get_numeric_data().columns
    cat_cols = list(set(cols) - set(num_cols))
    num_cols = [num_col for num_col in num_cols if num_col in ['age', 'avg_glucose_level', 'bmi']]

    X_df = X_df[cat_cols+num_cols]
    X_df = preprocess_columns(X_df)

    y_df = y_df.astype(int)
    y_word_dict = {1: 'Stroke_yes', 0: 'Stroke_no'}
    dataset = {
        'problem': 'classification',
        'full': {
            'X': X_df,
            'y': y_df,
        },
    }

    return dataset, y_word_dict


def load_telco_churn_data():
    # https://www.kaggle.com/blastchar/telco-customer-churn/downloads/WA_Fn-UseC_-Telco-Customer-Churn.csv/1
    df = pd.read_csv('../data/WA_Fn-UseC_-Telco-Customer-Churn.csv')
    y_df = df['Churn']
    X_df = df.drop(['Churn', 'customerID'], axis=1)

    X_df['SeniorCitizen'] = X_df['SeniorCitizen'].replace({1: "Yes", 0: "No"})
    X_df['TotalCharges'] = pd.to_numeric(X_df['TotalCharges'].replace(" ", ""))

    cols = X_df.columns
    num_cols = X_df._get_numeric_data().columns
    cat_cols = list(set(cols) - set(num_cols))
    num_cols = [num_col for num_col in num_cols if num_col in ['tenure', 'MonthlyCharges', 'TotalCharges']]

    X_df = X_df[cat_cols + num_cols]
    X_df = preprocess_columns(X_df)

    y_df = y_df.replace({'Yes': 1, 'No': 0})
    y_df = y_df.astype(int)
    y_word_dict = {1: 'Churn_Yes', 0: 'Churn_No'}

    dataset = {
        'problem': 'classification',
        'full': {
            'X': X_df,
            'y': y_df,
        },
    }

    return dataset, y_word_dict


def load_fico_data():
    # https://community.fico.com/s/explainable-machine-learning-challenge?tabset-158d9=3
    df = pd.read_csv('../data/fico_heloc_dataset_v1.csv')
    X_df = df.drop(['RiskPerformance'], axis=1)

    X_df['MaxDelq2PublicRecLast12M'] = X_df['MaxDelq2PublicRecLast12M'].astype(str)
    X_df['MaxDelqEver'] = X_df['MaxDelqEver'].astype(str)

    cols = X_df.columns
    num_cols = X_df._get_numeric_data().columns
    cat_cols = list(set(cols) - set(num_cols))
    cat_cols = [cat_col for cat_col in cat_cols if cat_col in ['MaxDelq2PublicRecLast12M', 'MaxDelqEver']]

    X_df = X_df[cat_cols+num_cols.tolist()]
    X_df = preprocess_columns(X_df)

    y_df = df['RiskPerformance']
    y_df = y_df.replace({'Good': 1, 'Bad': 0})
    y_df = y_df.astype(int)
    y_word_dict = {1: 'Good', 0: 'Bad'}

    dataset = {
        'problem': 'classification',
        'full': {
            'X': X_df,
            'y': y_df,
        },
    }

    return dataset, y_word_dict


def load_bank_marketing_data():
    # https://archive.ics.uci.edu/ml/datasets/Bank+Marketing
    df = pd.read_csv('../data/bank-full.csv', sep=';')
    y_df = df['y']
    X_df = df.drop('y', axis=1)

    cols = X_df.columns
    num_cols = X_df._get_numeric_data().columns
    cat_cols = list(set(cols) - set(num_cols))
    num_cols = [num_col for num_col in num_cols if num_col in ['age', 'duration', 'campaign', 'pdays', 'previous']]

    X_df = X_df[cat_cols + num_cols]
    X_df = preprocess_columns(X_df)

    y_df = y_df.replace({'yes': 1, 'no': 0})
    y_df = y_df.astype(int)

    y_word_dict = {1: 'Deposit_subscribed_yes', 0: 'Deposit_subscribed_no'}

    dataset = {
        'problem': 'classification',
        'full': {
            'X': X_df,
            'y': y_df,
        },
    }

    return dataset, y_word_dict


def load_adult_data():
    df = pd.read_csv('../data/adult_census_income.csv')
    X_df = df.drop(['income'], axis=1)

    cols = X_df.columns
    num_cols = X_df._get_numeric_data().columns
    cat_cols = list(set(cols) - set(num_cols))
    num_cols = [num_col for num_col in num_cols if num_col in ['age', 'fnlwgt', 'education.num',
                                                               'capital.gain', 'capital.loss', 'hours.per.week']]

    X_df = X_df[cat_cols + num_cols]
    X_df = preprocess_columns(X_df)

    y_df = df["income"]
    y_df = y_df.replace({' <=50K': 0, ' >50K': 1})
    y_df = y_df.astype(int)

    y_word_dict = {0: 'Income<=50K', 1: 'Income>50K'}

    dataset = {
        'problem': 'classification',
        'full': {
            'X': X_df,
            'y': y_df,
        },
    }

    return dataset, y_word_dict


def load_airline_passenger_data():
    # https://www.kaggle.com/teejmahal20/airline-passenger-satisfaction
    df = pd.read_csv('../data/airline_train.csv', sep=',')
    y_df = df['satisfaction']
    X_df = df.drop(['Unnamed: 0', 'id', 'satisfaction'], axis=1)

    cols = X_df.columns
    num_cols = X_df._get_numeric_data().columns
    cat_cols = list(set(cols) - set(num_cols))
    cat_cols = [cat_col for cat_col in cat_cols if cat_col in ['Gender', 'Customer Type', 'Type of Travel', 'Class']]

    X_df = X_df[cat_cols + num_cols.tolist()]
    X_df = preprocess_columns(X_df)

    y_df = y_df.replace({'satisfied': 1, 'neutral or dissatisfied': 0})
    y_df = y_df.astype(int)

    dataset = {
        'problem': 'classification',
        'full': {
            'X': X_df,
            'y': y_df,
        },
    }

    return dataset


def load_car_data():
    # https: // archive.ics.uci.edu / ml / datasets / automobile
    df = pd.read_csv('../data/car.data', sep=',')
    X_df = df.drop(['price'], axis=1)

    X_df = X_df.replace("?", "")
    X_df['peak-rpm'] = pd.to_numeric(X_df['peak-rpm'])
    X_df['horsepower'] = pd.to_numeric(X_df['horsepower'])
    X_df['stroke'] = pd.to_numeric(X_df['stroke'])
    X_df['bore'] = pd.to_numeric(X_df['bore'])
    X_df['normalized-losses'] = pd.to_numeric(X_df['normalized-losses'])

    cols = X_df.columns
    num_cols = X_df._get_numeric_data().columns
    cat_cols = list(set(cols) - set(num_cols))
    num_cols = [num_col for num_col in num_cols if num_col in ['wheel-base', 'length', 'width', 'height', 'curb-weight',
                                                               'engine-size', 'bore', 'stroke', 'compression-ratio',
                                                               'horsepower', 'peak-rpm', 'city-mpg', 'highway-mpg']]

    X_df = X_df[cat_cols + num_cols]
    X_df = preprocess_columns(X_df)

    y_df = df['price']
    pt = PowerTransformer(method="box-cox")
    y_np = pt.fit_transform(y_df.to_numpy().reshape(-1, 1))
    y_df = pd.DataFrame(data=y_np, columns=["y"])

    dataset = {
        'problem': 'regression',
        'full': {
            'X': X_df,
            'y': y_df,
        },
    }

    return dataset


def load_student_grade_data():
    # https://archive.ics.uci.edu/ml/datasets/Student+Performance
    df = pd.read_csv('../data/student-por.csv', sep=';')
    X_df = df.drop(['G1', 'G2', 'G3'], axis=1)

    cols = X_df.columns
    num_cols = X_df._get_numeric_data().columns
    cat_cols = list(set(cols) - set(num_cols))
    num_cols = [num_col for num_col in num_cols if num_col in ['age', 'medu', 'fedu', 'traveltime', 'studytime',
                                                               'failures', 'famrel', 'freetime', 'goout',
                                                               'Dalc', 'Walc', 'health', 'absences']]

    X_df = X_df[cat_cols + num_cols]
    X_df = preprocess_columns(X_df)

    y_df = df['G3']
    pt = PowerTransformer()
    y_np = pt.fit_transform(y_df.to_numpy().reshape(-1, 1))
    y_df = pd.DataFrame(data=y_np, columns=["y"])

    dataset = {
        'problem': 'regression',
        'full': {
            'X': X_df,
            'y': y_df,
        },
    }

    return dataset


def load_crimes_data():
    # https://archive.ics.uci.edu/ml/datasets/Communities+and+Crime
    df = pd.read_csv('../data/communities.data', sep=',')
    X_df = df.drop(['ViolentCrimesPerPop', 'state', 'county', 'community', 'communityname string', 'fold'], axis=1)

    X_df = X_df.replace("?", "")
    X_df = preprocess_columns(X_df)

    X_df = X_df.drop(['LemasGangUnitDeploy', 'NumKindsDrugsSeiz'], axis=1)

    y_df = df['ViolentCrimesPerPop']
    pt = PowerTransformer()
    y_np = pt.fit_transform(y_df.to_numpy().reshape(-1, 1))
    y_df = pd.DataFrame(data=y_np, columns=["y"])

    dataset = {
        'problem': 'regression',
        'full': {
            'X': X_df,
            'y': y_df,
        },
    }

    return dataset


def load_bike_sharing_data():
    # https://archive.ics.uci.edu/ml/datasets/Bike+Sharing+Dataset
    df = pd.read_csv('../data/bike.csv', sep=',')
    X_df = df.drop(['instant', 'dteday', 'cnt', 'casual', 'registered'], axis=1)

    X_df['season'] = X_df['season'].astype(str)
    X_df['yr'] = X_df['yr'].astype(str)
    X_df['holiday'] = X_df['holiday'].astype(str)
    X_df['weekday'] = X_df['weekday'].astype(str)
    X_df['workingday'] = X_df['workingday'].astype(str)
    X_df['weathersit'] = X_df['weathersit'].astype(str)

    X_df = preprocess_columns(X_df)

    y_df = df['cnt']
    pt = PowerTransformer()
    y_np = pt.fit_transform(y_df.to_numpy().reshape(-1, 1))
    y_df = pd.DataFrame(data=y_np, columns=["y"])

    dataset = {
        'problem': 'regression',
        'full': {
            'X': X_df,
            'y': y_df,
        },
    }

    return dataset


def load_california_housing_data():
    # https://www.dcc.fc.up.pt/~ltorgo/Regression/cal_housing.html
    df = pd.read_csv('../data/cal_housing.data', sep=',')
    X_df = df.drop(['medianHouseValue'], axis=1)

    X_df = preprocess_columns(X_df)

    y_df = df['medianHouseValue']
    pt = PowerTransformer()
    y_np = pt.fit_transform(y_df.to_numpy().reshape(-1, 1))
    y_df = pd.DataFrame(data=y_np, columns=["y"])

    dataset = {
        'problem': 'regression',
        'full': {
            'X': X_df,
            'y': y_df,
        },
    }

    return dataset
