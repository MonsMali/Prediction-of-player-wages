import pandas as pd
import re
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Importing merged data 
final = pd.read_csv("C:/Users/Administrator/OneDrive/Ambiente de Trabalho/Master Classes/2nd Semester/Data Science project/Football project/top5 eur leagues DSP/merged_data_for_analysis.csv")

# Deleting specific columns:
# first column, as it is a counter 
# 'Player Name', as unique for each row 
# duplicate of 'Assists' 
# columns 'Nationality' and 'Club' becuase of too many unique categories
for idx, col in enumerate(final.columns):
    print(f"Index: {idx}, Column Name: {col}")
columns_to_delete_indices = [0, 1, 2, 13, 17]
all_indices = list(range(len(final.columns)))
remaining_indices = [i for i in all_indices if i not in columns_to_delete_indices]
final = final.iloc[:, remaining_indices]

# Standardizing column names, adding snake case 
def to_snake_case(s):
    s = s.strip()  # Remove leading/trailing whitespace
    s = re.sub(r'[\s\-\.\(\)]+', '_', s)  # Replace spaces, hyphens, dots, and parentheses with underscores
    s = re.sub(r'(?<!^)(?=[A-Z])', '_', s).lower()  # Add underscores before uppercase letters and convert to lowercase
    s = re.sub(r'_{2,}', '_', s)  # Replace multiple underscores with a single underscore
    s = re.sub(r'^_|_$', '', s)  # Remove leading/trailing underscores
    return s

final.columns = [to_snake_case(col) for col in final.columns]

# Manually shorten some column names 
new_column_names = {
    'annual_wage_in_e_u_r': 'annual_wage',
    'save_percentage': 'save_%',
    'clean_sheet_percentage': 'clean_sheet_%',
    'avg_length_of_goal_kicks': 'avg_length_goal_kicks',
    'avg_distance_of_def_actions': 'avg_distance_def_actions',
    'assists_x': 'assists'
}
final.rename(columns=new_column_names, inplace=True)
print(final.columns)

# Removing all rows with missing values for 'annual_wage'
missing_values_wage = final['annual_wage'].isnull().sum()
print(missing_values_wage)

final = final.dropna(subset=['annual_wage'])
final.reset_index(drop=True, inplace=True)
final

# Observing how many missing rows for other variables 
missing_values_count = final.isnull().sum()
missing_values_count

# Converting 'position' and 'league' to category data type
print(final.dtypes)
final['position'] = final['position'].astype('category')
final['league'] = final['league'].astype('category')

# Merging categories for 'position'
replacements = {
    'FW,DF': 'DF,FW',
    'DF,MF': 'MF,DF',
    'MF,FW': 'FW,MF'
}
final['position'] = final['position'].replace(replacements)

print(final.describe(include='all'))

# Additional EDA steps
# Correlation Matrix (numerical variables)
# Begin with corr. matrix as some variables are likely highly correlated
correlation_matrix = final.drop(columns = ['position', 'league']).corr()
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
# plt.show()

# Implement principal component analysis to compline highly corr. variables 
scaler = StandardScaler()

# Set 1: 'penalty_made' and 'penalty_attempted'
variables_set1 = final[['penalty_made', 'penalty_attempted']]
variables_set1_standardized = scaler.fit_transform(variables_set1)
pca = PCA(n_components=1)
final['penalty_play'] = pca.fit_transform(variables_set1_standardized)

# Set 2: 'matches', 'starts', and 'minutes'
variables_set2 = final[['matches', 'starts', 'minutes']]
variables_set2_standardized = scaler.fit_transform(variables_set2)
final['play_time'] = pca.fit_transform(variables_set2_standardized)


# Remove the variables used to create the pca componenets
remove_columns = ['penalty_made', 'penalty_attempted','matches', 'starts', 'minutes']
final.drop(columns=remove_columns, inplace=True)
print(final.columns)

# There are still a few pairs of highly correlated variables, but they can't be combined with PCA
# because techniquecan't be applied to variables with missing obsevations.
# Rest of PCA components will be extracted after splitting the data into Goalkeepers and Non-Goalkeepers 

# Distribution of numerical variables
final.hist(bins=30, figsize=(15, 10))
plt.suptitle('Histograms of Numerical Variables')
# plt.show()

# Distribution of categorical variables
categorical_columns = final.select_dtypes(include=['category', 'object']).columns
for col in categorical_columns:
    sns.countplot(y=col, data=final)
    plt.title(f'Countplot of {col}')
    # plt.show()

# Box Plots
sns.boxplot(x='league', y='annual_wage', data=final)
plt.title('Box Plot of Anual Wage by League')
plt.xticks(rotation=90)
# plt.show()

sns.boxplot(x='position', y='annual_wage', data=final)
plt.title('Box Plot of Anual Wage by Position')
plt.xticks(rotation=90)
# plt.show()

# Outlier detection (for 'annual_wage')
Q1 = final['annual_wage'].quantile(0.25)
Q3 = final['annual_wage'].quantile(0.75)
IQR = Q3 - Q1

lower_bound = Q1 - 2.5 * IQR
upper_bound = Q3 + 2.5 * IQR
outliers = (final['annual_wage'] < lower_bound) | (final['annual_wage'] > upper_bound)
sum(outliers)
# Remove outliers from the daset
final_no_outliers = final[~outliers]


# Storing the clean data for modelling
final_no_outliers.to_csv('C:/Users/Administrator/OneDrive/Ambiente de Trabalho/Master Classes/2nd Semester/Data Science project/Football project/top5 eur leagues DSP/player_stats_final.csv', index=False)
