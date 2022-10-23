def print_corr(df):
    A = df.pop('A')
    Y = df.pop('Y')
    X = df

    print('Correlation with treatment : ')
    for col in X.columns:
        print("A - ", col, ' : ', A.corr(X[col]))

    print('Correlation with outcome : ')
    for col in X.columns:
        print("Y - ", col, ' : ', Y.corr(X[col]))
