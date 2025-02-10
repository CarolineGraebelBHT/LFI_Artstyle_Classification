import pandas as pd
import matplotlib.pyplot as plt

df_train = pd.read_csv("Prediction_Data/TrainPredictions.csv")
df_test = pd.read_csv("Prediction_Data/TestPredictions.csv")

artstyles_dict = {
    0: 'Abstract_Expressionism',
    1: 'Baroque',
    2: 'Cubism',
    3: 'Expressionism',
    4: 'High_Renaissance',
    5: 'Impressionism',
    6: 'Realism'
}

df_train['Artstyle'] = df_train['True Values'].map(artstyles_dict)
df_train['Correct'] = (df_train['True Values'] == df_train['TrainPredictions']).astype(int)
print(df_train.head())

df_test['Artstyle'] = df_test['True Values'].map(artstyles_dict)
df_test['Correct'] = (df_test['True Values'] == df_test['TestPredictions']).astype(int)
print(df_test.head())

accuracy_per_style_train = df_train.groupby('Artstyle')['Correct'].mean().sort_values(ascending=False)

plt.figure(figsize=(10, 6))
plt.bar(accuracy_per_style_train.index, accuracy_per_style_train.values)
plt.xlabel("Art Style")
plt.ylabel("Accuracy")
plt.title("Prediction Accuracy by Art Style (Training Set)")
plt.xticks(rotation=45, ha='right')  # Rotate x-axis labels for better visibility
plt.tight_layout() # Adjust layout to prevent labels from overlapping
plt.show()

accuracy_per_style_test = df_test.groupby('Artstyle')['Correct'].mean().sort_values(ascending=False)

plt.figure(figsize=(10, 6))
plt.bar(accuracy_per_style_test.index, accuracy_per_style_test.values)
plt.xlabel("Art Style")
plt.ylabel("Accuracy")
plt.title("Prediction Accuracy by Art Style (Testing Set)")
plt.xticks(rotation=45, ha='right')  # Rotate x-axis labels for better visibility
plt.tight_layout() # Adjust layout to prevent labels from overlapping
plt.show()