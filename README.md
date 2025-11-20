# Projeto de Rede Neural - Previsão de Desempenho Estudantil

Este projeto utiliza uma Rede Neural MLP para prever se um aluno será aprovado ou reprovado.

##  Estrutura de Pastas
```text
RedeNeural-DesempenhoEstudantes/
│
├── data/
│   └── StudentPerformanceFactors.csv   # Dataset original (Kaggle)
│
├── src/
│   ├── pre_processamento.py            # Script 1: Trata os dados e gera os .npy
│   └── mlp_treinamento.py              # Script 2: Treina a rede e gera gráficos
│
├── docs/
│   ├── imagens_graficos/               # Onde os gráficos de acurácia/loss são salvos
│   └── Relatorio_Tecnico.pdf           # Relatório final do trabalho
│
├── requirements.txt                    # Lista de bibliotecas necessárias
└── README.md                           # Este arquivo
```
##  Pré-requisitos
Certifique-se de ter o Python (3.10 ou superior) instalado.
Instale as bibliotecas necessárias rodando:

```
pip install -r requirements.txt
```
ou se preferir acesse o próprio txt e instale via terminal todas as bibliotecas
ATENÇÃO (Windows + TensorFlow): A biblioteca tensorflow é robusta e pode exigir configurações adicionais no Windows:

Se ocorrer erro de instalação relacionado a limites de caracteres, pode ser necessário habilitar o suporte a "Long Paths" no Windows.

A instalação pode levar alguns minutos dependendo da conexão de internet.

##  Como Rodar (Passo a Passo)

O projeto é dividido em dois scripts que devem ser executados na ordem abaixo.

**Passo 1: Pré-processamento dos dados**
Este script lê o CSV, trata os dados e salva arquivos `.npy` prontos para a rede.
1. Abra o terminal na pasta `src`.
2. ```cd src```
3. Execute:
python pre_processamento.py

**Passo 2: Treinamento da Rede Neural**
Este script carrega os dados processados, treina a rede e gera os gráficos.
1. Ainda no terminal dentro da pasta `src`.
2. Execute:
python mlp_treinamento.py

Os gráficos de resultado serão salvos automaticamente na pasta `docs/imagens_graficos`.
