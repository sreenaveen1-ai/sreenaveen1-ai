import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Load the CSV file0
file_path = r"C:\Users\vgsre\OneDrive\Desktop\dataset\archive\train_and_test2.csv"
data = pd.read_csv(file_path)

# Remove redundant columns (columns that start with 'zero')
columns_to_remove = [col for col in data.columns if col.startswith('zero')]
data_cleaned = data.drop(columns=columns_to_remove)

# Fill missing values in 'Embarked' with the most common value
most_common_embarked = data_cleaned['Embarked'].mode()[0]
data_cleaned['Embarked'].fillna(most_common_embarked, inplace=True)

# Rename '2urvived' to 'Survived'
data_cleaned.rename(columns={'2urvived': 'Survived'}, inplace=True)

#APPLY LABEL ENCODING TO 'SEX'
data_cleaned=LabelEncoder()
data['sex']=data_cleaned.fit_transform(data['sex'])

# Save the cleaned data to a new CSV file
cleaned_file_path = r"C:\Users\vgsre\OneDrive\Desktop\dataset\archive\train_and_test2.csv"
data_cleaned.to_csv(cleaned_file_path, index=False)

# Verify the changes
print(data_cleaned.head())
print(data_cleaned.isnull().sum())
