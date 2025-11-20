import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

print("--- Processando dados ---")

df = pd.read_csv('../data/StudentPerformanceFactors.csv')

if 'Exam_Score' in df.columns:
    df['Pass_Fail'] = df['Exam_Score'].apply(lambda x: 1 if x >= 65 else 0)
    df = df.drop('Exam_Score', axis=1) 

if 'Student_ID' in df.columns:
    df = df.drop('Student_ID', axis=1)

X = df.drop('Pass_Fail', axis=1)
Y = df['Pass_Fail']

X = pd.get_dummies(X, drop_first=True)


X_treino, X_teste, Y_treino, Y_teste = train_test_split(X, Y, test_size=0.2, random_state=42)


scaler = StandardScaler()
X_treino_scaled = scaler.fit_transform(X_treino)
X_teste_scaled = scaler.transform(X_teste)

print(f"Dados prontos! Features: {X_treino_scaled.shape[1]}")

np.save('X_treino.npy', X_treino_scaled)
np.save('X_teste.npy', X_teste_scaled)
np.save('Y_treino.npy', Y_treino.values)
np.save('Y_teste.npy', Y_teste.values)

print("Arquivos salvos na pasta src.")