import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import matplotlib.pyplot as plt
import os

print("--- Iniciando Treinamento ---")

#Carregar os dados 
try:
    X_treino = np.load('X_treino.npy')
    X_teste = np.load('X_teste.npy')
    Y_treino = np.load('Y_treino.npy')
    Y_teste = np.load('Y_teste.npy')
except FileNotFoundError:
    print("Erro: Rode o pre_processamento.py primeiro!")
    exit()

n_features = X_treino.shape[1]

# Função para criar a rede neural
def criar_modelo(funcao_ativacao):
    model = Sequential([
        Dense(128, activation=funcao_ativacao, input_shape=(n_features,)),
        Dense(64, activation=funcao_ativacao),
        Dense(32, activation=funcao_ativacao), 
        Dense(16, activation=funcao_ativacao),
        Dense(1, activation='sigmoid') # Saída binária (0 ou 1)
    ])
    
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), 
                  loss='binary_crossentropy', 
                  metrics=['accuracy'])
    return model


for ativacao in ['relu', 'tanh']:
    print(f"\nTreinando com {ativacao}...")
    
    model = criar_modelo(ativacao)
    
    # Treina a rede 
    history = model.fit(X_treino, Y_treino, 
                        epochs=50, 
                        batch_size=32, 
                        validation_data=(X_teste, Y_teste),
                        verbose=1)
    
    # Salva resultado final
    acc = model.evaluate(X_teste, Y_teste, verbose=0)[1]
    print(f"Acurácia final ({ativacao}): {acc*100:.2f}%")
    
    # 4. Gerar e Salvar Gráficos
    # Cria a pasta docs/imagens_graficos se não existir
    if not os.path.exists('../docs/imagens_graficos'):
        os.makedirs('../docs/imagens_graficos')

    plt.figure(figsize=(10, 4))
    
    # Gráfico de Loss
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Treino')
    plt.plot(history.history['val_loss'], label='Validação')
    plt.title(f'Loss - {ativacao}')
    plt.legend()

    # Gráfico de Acurácia
    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Treino')
    plt.plot(history.history['val_accuracy'], label='Validação')
    plt.title(f'Accuracy - {ativacao}')
    plt.legend()
    
    plt.savefig(f'../docs/imagens_graficos/resultado_{ativacao}.png')
    plt.close()

print("\nTudo pronto! Gráficos salvos na pasta docs.")