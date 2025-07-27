# Sistema de Predição de Obesidade

**Desenvolvido por:** João Paulo  
**Tech Challenge - Fase 04**

## Descrição do Projeto

Este projeto implementa um sistema completo de Machine Learning para auxiliar profissionais de saúde na predição de níveis de obesidade em pacientes. O sistema inclui pipeline de ML, aplicação web interativa e dashboard analítico.

## Estrutura do Projeto

```
├── data_pipeline.py      # Pipeline de Machine Learning
├── app.py               # Aplicação Streamlit
├── requirements.txt     # Dependências
├── Obesity.csv         # Dataset
└── README.md           # Este arquivo
```

## Funcionalidades

### 1. Pipeline de Machine Learning
- ✅ Análise exploratória dos dados
- ✅ Pré-processamento completo (Feature Engineering)
- ✅ Treinamento de múltiplos modelos
- ✅ Avaliação e seleção do melhor modelo
- ✅ Modelo com acurácia superior a 75%

### 2. Aplicação de Predição (Streamlit)
- ✅ Interface intuitiva para inserção de dados
- ✅ Predição em tempo real
- ✅ Visualização de probabilidades
- ✅ Cálculo automático de IMC

### 3. Dashboard Analítico
- ✅ Métricas principais sobre obesidade
- ✅ Visualizações interativas
- ✅ Insights para equipe médica
- ✅ Análise de correlações

## Como Executar

### Passo 1: Instalação das Dependências
```bash
pip install -r requirements.txt
```

### Passo 2: Treinar o Modelo
```bash
python data_pipeline.py
```

### Passo 3: Executar a Aplicação
```bash
streamlit run app.py
```

## Modelos Testados

1. **Logistic Regression** - Modelo linear baseline
2. **Decision Tree** - Modelo não-linear interpretável  
3. **Gradient Boosting** - Modelo ensemble (melhor performance)

## Métricas de Avaliação

- **Acurácia**: > 75% (requisito atendido)
- **Precision, Recall, F1-Score**: Balanceamento entre classes
- **Matriz de Confusão**: Análise detalhada de erros
- **ROC AUC**: Capacidade de separação das classes

## Variáveis do Dataset

### Demográficas
- **Gender**: Gênero
- **Age**: Idade
- **Height**: Altura (metros)
- **Weight**: Peso (kg)

### Hábitos Alimentares
- **family_history**: Histórico familiar de obesidade
- **FAVC**: Consumo frequente de alimentos calóricos
- **FCVC**: Frequência de consumo de vegetais
- **NCP**: Número de refeições principais
- **CAEC**: Alimentação entre refeições

### Estilo de Vida
- **SMOKE**: Fumante
- **CH2O**: Consumo diário de água
- **SCC**: Monitoramento de calorias
- **FAF**: Frequência de atividade física
- **TUE**: Tempo usando tecnologia
- **CALC**: Consumo de álcool
- **MTRANS**: Meio de transporte

### Target
- **Obesity**: Nível de obesidade (7 categorias)

## Tecnologias Utilizadas

- **Python 3.8+**
- **Scikit-learn**: Machine Learning
- **Streamlit**: Interface web
- **Plotly**: Visualizações interativas
- **Pandas/NumPy**: Manipulação de dados

## Deploy

Para deploy no Streamlit Cloud:
1. Faça upload do código para GitHub
2. Conecte o repositório ao Streamlit Cloud
3. Configure o arquivo principal como `app.py`
4. Aguarde o deploy automático

## Resultados Esperados

- **Modelo com acurácia > 75%** ✅
- **Sistema preditivo funcional** ✅  
- **Dashboard analítico completo** ✅
- **Interface limpa e profissional** ✅

## Contato

O link do vídeo apresentando o sistema foi enviado via plataforma da FIAP.



