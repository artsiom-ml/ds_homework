import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
from sklearn.preprocessing import OneHotEncoder

pd.options.mode.chained_assignment = None

def preprocess(df):
    df.drop(['ID'], axis=1, inplace=True)
    df.drop(['age_desc'], axis=1, inplace=True)
    df['ethnicity'].replace('?', 'Others', inplace=True)
    df['ethnicity'].replace('others', 'Others', inplace=True)
    df['relation'].replace('?', 'Others', inplace=True)
    
    all_countries = df['contry_of_res'].value_counts()
    others_countries = all_countries.mask(all_countries > 20).dropna()
    others_countries = others_countries.index.to_list()
    df['contry_of_res'].loc[df['contry_of_res'].isin(others_countries)] = 'Other'
    df['contry_of_res'].value_counts()

    oe_cols = ['jaundice', 'austim', 'gender', 'used_app_before', 'relation', 'ethnicity']
    oe = OrdinalEncoder()
    df[oe_cols] = oe.fit_transform(df[oe_cols])

    ohe_cols = ['contry_of_res']
    enc = OneHotEncoder(handle_unknown = 'ignore')
    enc.fit(df[ohe_cols])
    df_enc = pd.DataFrame(enc.transform(df[ohe_cols]).toarray(), index=df.index, columns=enc.get_feature_names_out())
    df = pd.concat([df.drop(ohe_cols, axis=1), df_enc], axis=1)
    
    X = df.drop(['Class/ASD'], axis=1).values
    y = df['Class/ASD'].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, stratify=y, random_state=42)
    
    scaler = StandardScaler()

    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc= scaler.transform(X_test)
    
    return X_train_sc, y_train, X_test_sc, y_test