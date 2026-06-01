# Tech Challenge 4: Predição de Níveis de Obesidade (Foco em Regras de Negócio)

Bem-vindo ao repositório do **Tech Challenge 4 (FIAP - Data Analytics)**. Este projeto tem como objetivo desenvolver um sistema inteligente e interativo capaz de prever níveis de obesidade. O foco central não é apenas a métrica de acurácia, mas a aplicação rigorosa de Ciência de Dados com visão de **Negócios e Ética (identificação de viés e Data Leakage)**.

## 🎯 Objetivo do Projeto

O desafio propõe a construção de um pipeline de Machine Learning preditivo para a área da saúde. Contudo, em análises iniciais, constatou-se que o uso irrestrito de variáveis conduzia o modelo a dois problemas fundamentais:

1. **Vazamento de Dados (Data Leakage) / Viés:** A variável "Gênero", por exemplo, indicava quase 100% de prevalência masculina ou feminina para classes específicas de obesidade severa na base de dados, fazendo com que o algoritmo apenas "decorasse" o sexo biológico ao invés de interpretar hábitos reais. O mesmo ocorreu para características sem variância estatística (ex: Hábito de Fumar).
2. **O Dilema da "Calculadora de IMC":** Treinar um modelo tendo acesso aos dados exatos de "Peso" e "Altura" faz com que o Machine Learning aja apenas como uma calculadora superdimensionada de IMC (Índice de Massa Corporal), com mais de 99% de acurácia, invalidando o uso de I.A. preditiva.

A versão final propõe **duas abordagens (Dupla Modelagem)**:
* **Modelo Biométrico:** Uma *baseline* para provar a alta assertividade utilizando IMC.
* **Modelo Comportamental / Genético (Feature Selection de Ouro):** Responde à pergunta central de negócio: *"Podemos prever o risco de obesidade de um paciente com mais de 75% de acurácia apenas analisando sua rotina, alimentação e genética familiar, sem colocá-lo na balança?"*.

## 📂 Estrutura do Repositório

* `dados/` - Diretório contendo a base de dados original (`Obesity.csv`).
* `docs/` - Documentação oficial, incluindo o dicionário de variáveis.
* `modelos/` - Arquivos de modelos exportados via `joblib/pickle` (Modelos Random Forest e Label Encoders).
* `notebooks/` - Diretório contendo as análises. O destaque é o arquivo final **`notebook_refatorado_obesidade.ipynb`**, que centraliza toda a Análise Exploratória de Dados (EDA) inicial, o estudo de Data Leakage, Feature Selection e Treinamento/Avaliação da Dupla Modelagem.
* `app.py` - O sistema interativo principal (frontend) desenvolvido utilizando **Streamlit**, que consome o modelo treinado para servir de assistente e painel analítico para a equipe médica.

## 🚀 Como Executar a Aplicação (Streamlit)

O painel final fornece duas abas: um simulador de triagem preditiva que aceita entradas do usuário e um painel de "Insights" com gráficos traduzidos que destacam os maiores riscos comportamentais e genéticos da base.

### Pré-requisitos
* Python 3.8+
* `pip install -r requirements.txt` *(certifique-se de instalar as dependências: pandas, numpy, scikit-learn, matplotlib, seaborn, streamlit, joblib)*

### Rodando o Servidor Local
Clone este repositório e, na pasta raiz, execute:
```bash
streamlit run app.py
```
Acesse no seu navegador através do endereço local (geralmente `http://localhost:8501`).

## 🧠 Conclusões do Estudo (Insights)
* **Genética é Crucial:** O Histórico Familiar de sobrepeso pula de uma taxa basal de ocorrência para quase 100% de correlação nas faixas de Obesidade Tipo II e III.
* **A Força do Comportamento:** Hábitos como sedentarismo extremo e alto consumo hipercalórico demonstram ser features essenciais (Features de Ouro) capazes de prever a Classe de Obesidade mesmo na completa ausência de informações biométricas numéricas.