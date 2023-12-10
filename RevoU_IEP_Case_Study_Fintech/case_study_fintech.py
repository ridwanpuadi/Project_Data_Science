# Import Library
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Import dataset
df = pd.read_csv('data_fintech.csv')

# Data summary
summary = df.describe()
data_type = df.dtypes

# Revised Column numscreens
df['screen_list'] = df.screen_list.astype(str) + ','
df['num_screens'] = df.screen_list.str.count(',')
df.drop(columns=['numscreens'], inplace=True)

# Ceking column hour
df.hour[1]
df.hour = df.hour.str.slice(1,3).astype(int)

# Defines a numeric special variable
df_numeric = df.drop(columns=['user','first_open','screen_list',
                                      'enrolled_date'], inplace=False)

# Create histogram
sns.set()
plt.suptitle('Histogram Data Numeric')
for i in range(0, df_numeric.shape[1]):
    plt.subplot(3,3,i+1)
    figure = plt.gca()
    figure.set_title(df_numeric.columns.values[i])
    count_bin = np.size(df_numeric.iloc[:,i].unique())
    plt.hist(df_numeric.iloc[:,i], bins=count_bin)

# Create correlation matrix
correlation = df_numeric.drop(columns=['enrolled'], inplace=False).corrwith(df_numeric.enrolled)
correlation.plot.bar(title='Variable correlation to enrollment decisions')

correlation_matrix = df_numeric.drop(columns=['enrolled'], inplace=False).corr()
sns.heatmap(correlation_matrix, cmap='Blues')

mask = np.zeros_like(correlation_matrix)
mask[np.triu_indices_from(mask)] = True

# Create Correlation Matrix with heatmap custom
ax = plt.axes()
my_cmap = sns.diverging_palette(200, 0, as_cmap=True)
sns.heatmap(correlation_matrix, cmap=my_cmap, mask=mask, 
            linewidths=0.5, center=0, square=True)
ax = plt.suptitle('Correlation Matrix Custom')

# FEATURE ENGINEERING
# parsing process
from dateutil import parser
df.first_open = [parser.parse(i) for i in df.first_open]
df.enrolled_date = [parser.parse(i) if isinstance(i, str) else i for i in df.enrolled_date]
df['difference'] = (df.enrolled_date - df.first_open).astype('timedelta64[h]')

# Create plot histogram df.difference
plt.hist(df.difference.dropna(), range=[0,200])
plt.suptitle('time difference between  enrolled with first open')
plt.show()

# Filtering value difference > 48 hour
df.loc[df.difference>48, 'enrolled'] = 0

# Import dataet top_screens
top_screens = pd.read_csv('top_screens.csv')
top_screens = np.array(top_screens.loc[:,'top_screens'])

# Back up data
df2 = df.copy()

# Create a column for each top_screens
for screen in top_screens:
    df2[screen] = df2.screen_list.str.contains(screen).astype(int)
    
for screen in top_screens:
    df2['screen_list'] = df2.screen_list.str.replace(screen+',', '')

# Item non top_screens in screen_list
df2['other'] = df2.screen_list.str.count(',')

top_screens.sort()

# The process of merging several similar screens (Funneling)
screen_loan = ['Loan',
              'Loan2',
              'Loan3',
              'Loan4']
df2['count_loan'] = df2[screen_loan].sum(axis=1) # axis=1 it means counting the number of items per rows
df2.drop(columns=screen_loan, inplace=True) # remove all columns from the list savings_screen

screen_saving = ['Saving1',
                'Saving2',
                'Saving2Amount',
                'Saving4',
                'Saving5',
                'Saving6',
                'Saving7',
                'Saving8',
                'Saving9',
                'Saving10']
df2['count_saving'] = df2[screen_saving].sum(axis=1) 
df2.drop(columns=screen_saving, inplace=True) 

screen_credit = ['Credit1',
                'Credit2',
                'Credit3',
                'Credit3Container',
                'Credit3Dashboard']
df2["count_kredit"] = df2[screen_credit].sum(axis=1)
df2.drop(columns=screen_credit, inplace=True)

screen_cc = ['CC1',
            'CC1Category',
            'CC3']
df2['count_cc'] = df2[screen_cc].sum(axis=1)
df2.drop(columns=screen_cc, inplace=True)

# Define the dependent variable 
var_enrolled = np.array(df2['enrolled'])

# Removed some redundant columns
df2.drop(columns = ['first_open', 'screen_list','enrolled',
                        'enrolled_date', 'difference'], inplace=True)

# Divide into training and test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(df2, var_enrolled, 
                                                    test_size=0.2,
                                                    random_state=111)
                                                    
# Save user ID for training and test set
train_id = np.array(X_train['user'])
test_id = np.array(X_test['user'])

# Remove column user in X_train and X_test
X_train.drop(columns=['user'], inplace=True)
X_test.drop(columns=['user'], inplace=True)

# Change X_train and X_test become numpy array (test set is already an array so there's no need)
X_train = np.array(X_train)
X_test = np.array(X_test)

# Preprocessing Standardization (Feature Scaling)
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

# Eliminate empty variables
X_train = np.delete(X_train, 27, 1)
X_test = np.delete(X_test, 27, 1)

# Logistic Regression
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0, solver='liblinear',
                                penalty = 'l1')
classifier.fit(X_train, y_train)

# Predict test set
y_pred = classifier.predict(X_test)

# Evaluate model with confusion matrix
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
cm = confusion_matrix(y_test, y_pred)
print(classification_report(y_test, y_pred))

# Use accuracy_score
evaluation = accuracy_score(y_test, y_pred)
print('Accuracy:{:.2f}'.format(evaluation*100))

# Use seabor for CM
cm_label = pd.DataFrame(cm, columns = np.unique(y_test),
                        index = np.unique(y_test))
cm_label.index.name = 'Actual'
cm_label.columns.name = 'Predict'
sns.heatmap(cm_label, annot=True, cmap='Reds', fmt='g')

# Validate with 10-fold cross validation
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = classifier, X=X_train, y=y_train,
                             cv=10)
accuracies.mean()
accuracies.std()
print('Logistic Regression Accuracy = {:.2f}% +/- {:.2f}%'.format(accuracies.mean()*100, accuracies.std()*100))

# Put it all together
y_pred_series = pd.Series(y_test).rename('original_table', inplace=True)
final_result = pd.concat([y_pred_series, pd.DataFrame(test_id)], axis=1).dropna()
final_result['predict'] = y_pred
final_result.rename(columns={0:'user'}, inplace = True)
final_result = final_result[['user','original_table','predict']].reset_index(drop=True)

count_1 = final_result['predict'].value_counts()[1]
count_0 = final_result['predict'].value_counts()[0]

percentage_1 = (count_1/len(final_result))*100
percentage_0 = (count_0/len(final_result))*100

print('percentage_predict1:', percentage_1, "%")
print('percentage_predict0:', percentage_0, "%")
