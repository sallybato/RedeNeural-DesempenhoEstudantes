import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

print("--- Processando dados ---")

# 1. Carregar o dataset direto
# O '../data/' volta uma pasta e entra na pasta data
df = pd.read_csv('../data/StudentPerformanceFactors.csv')

# 2. Criar a coluna alvo (Pass/Fail) baseada na nota
# Se a nota for maior ou igual a 65 passa (1), senão reprova (0)
if 'Exam_Score' in df.columns:
    df['Pass_Fail'] = df['Exam_Score'].apply(lambda x: 1 if x >= 65 else 0)
    df = df.drop('Exam_Score', axis=1) # Remove a nota para não enviesar a rede

# Remove coluna de ID que não serve pra nada
if 'Student_ID' in df.columns:
    df = df.drop('Student_ID', axis=1)

# Separa X (dados) e Y (target)
X = df.drop('Pass_Fail', axis=1)
Y = df['Pass_Fail']

# 3. Transformar texto em números (One-Hot Encoding)
# Pega tudo que é object (texto) e transforma em dummy variables
X = pd.get_dummies(X, drop_first=True)

# 4. Divisão Treino e Teste
X_treino, X_teste, Y_treino, Y_teste = train_test_split(X, Y, test_size=0.2, random_state=42)

# 5. Padronização (Deixar os valores na mesma escala)
scaler = StandardScaler()
X_treino_scaled = scaler.fit_transform(X_treino)
X_teste_scaled = scaler.transform(X_teste)

print(f"Dados prontos! Features: {X_treino_scaled.shape[1]}")

# Salva os arquivos npy para usar no treinamento
np.save('X_treino.npy', X_treino_scaled)
np.save('X_teste.npy', X_teste_scaled)
np.save('Y_treino.npy', Y_treino.values)
np.save('Y_teste.npy', Y_teste.values)

print("Arquivos salvos na pasta src.")