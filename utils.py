import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib
import os

def carregar_e_preparar_dados(caminho_arquivo):
    """
    Carrega o dataset, traduz as colunas (se necessário) e aplica as regras
    de engenharia de features definidas na estratégia de negócio.
    """
    print(f"Carregando dados de: {caminho_arquivo}...")
    
    # Substitua pelo delimitador correto do seu CSV se necessário
    df = pd.read_csv(caminho_arquivo)
    
    # 1. Renomeando as colunas para o padrão do nosso app Streamlit
    # (Ajuste o lado esquerdo deste dicionário de acordo com os nomes originais do seu CSV)
    mapa_colunas = {
        'Age': 'Idade',
        'Weight': 'Peso',
        'Height': 'Altura',
        'family_history': 'Historico_familiar_sobrepeso',
        'FAVC': 'Consumo_alimentos_hipercaloricos',
        'FAF': 'Frequencia_semanal_atividade_fisica',
        'FCVC': 'Frequencia_consumo_vegetais',
        'NCP': 'Numero_refeicoes_diarias',
        'Obesity_level': 'Classe_peso_corporal'
    }
    
    # Renomeia apenas as colunas que existem no dicionário
    df = df.rename(columns={k: v for k, v in mapa_colunas.items() if k in df.columns})
    
    # 2. Mapeamento de variáveis categóricas para numéricas (0 e 1)
    # Assumindo que os dados originais em inglês eram 'yes'/'no'
    mapa_sim_nao = {'yes': 1, 'no': 0, 'yes ': 1, 'no ': 0}
    
    if df['Historico_familiar_sobrepeso'].dtype == 'O':
        df['Historico_familiar_sobrepeso'] = df['Historico_familiar_sobrepeso'].str.lower().map(mapa_sim_nao)
        
    if df['Consumo_alimentos_hipercaloricos'].dtype == 'O':
        df['Consumo_alimentos_hipercaloricos'] = df['Consumo_alimentos_hipercaloricos'].str.lower().map(mapa_sim_nao)

    # 3. Engenharia de Feature: Criação do IMC
    df['IMC'] = df['Peso'] / (df['Altura'] ** 2)

    # Arredondando variáveis comportamentais que podem ter vindo com decimais (ruído)
    df['Frequencia_consumo_vegetais'] = df['Frequencia_consumo_vegetais'].round().astype(int)
    df['Numero_refeicoes_diarias'] = df['Numero_refeicoes_diarias'].round().astype(int)
    df['Frequencia_semanal_atividade_fisica'] = df['Frequencia_semanal_atividade_fisica'].round().astype(int)

    # Removendo linhas com valores nulos que possam ter sido gerados
    df = df.dropna()

    return df

def treinar_e_exportar_modelos(df):
    """
    Separa as features, treina a abordagem Biométrica e a Comportamental,
    exibe as métricas de validação e exporta os modelos em .pkl.
    """
    print("\nIniciando treinamento dos modelos...")
    
    # Definindo as variáveis alvo e os conjuntos de features
    y = df['Classe_peso_corporal']
    
    features_comportamentais = [
        'Historico_familiar_sobrepeso', 
        'Consumo_alimentos_hipercaloricos', 
        'Frequencia_semanal_atividade_fisica', 
        'Idade', 
        'Frequencia_consumo_vegetais', 
        'Numero_refeicoes_diarias'
    ]
    
    features_biometricas = features_comportamentais + ['IMC']
    
    # Separando os dados (Usando as mesmas sementes para comparação justa)
    X_comp = df[features_comportamentais]
    X_bio = df[features_biometricas]
    
    X_comp_train, X_comp_test, y_train, y_test = train_test_split(X_comp, y, test_size=0.2, random_state=42, stratify=y)
    X_bio_train, X_bio_test, _, _ = train_test_split(X_bio, y, test_size=0.2, random_state=42, stratify=y)

    # ==========================================
    # MODELO 1: Biométrica (Com IMC)
    # ==========================================
    modelo_bio = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)
    modelo_bio.fit(X_bio_train, y_train)
    y_pred_bio = modelo_bio.predict(X_bio_test)
    acc_bio = accuracy_score(y_test, y_pred_bio)
    
    print("\n" + "="*50)
    print("📈 RESULTADOS: MODELO BIOMÉTRICO (Com IMC)")
    print("="*50)
    print(f"Acurácia: {acc_bio * 100:.2f}%\n")
    print(classification_report(y_test, y_pred_bio))
    
    # ==========================================
    # MODELO 2: Comportamental (Sem Peso, Altura e IMC)
    # ==========================================
    modelo_comp = RandomForestClassifier(n_estimators=150, random_state=42, max_depth=12, class_weight='balanced')
    modelo_comp.fit(X_comp_train, y_train)
    y_pred_comp = modelo_comp.predict(X_comp_test)
    acc_comp = accuracy_score(y_test, y_pred_comp)
    
    print("\n" + "="*50)
    print("📊 RESULTADOS: MODELO COMPORTAMENTAL (Sem Balança)")
    print("="*50)
    print(f"Acurácia: {acc_comp * 100:.2f}% (Meta projeto: >75%)\n")
    print(classification_report(y_test, y_pred_comp))

    # ==========================================
    # EXPORTAÇÃO (Artefatos)
    # ==========================================
    print("\nSalvando modelos (.pkl)...")
    joblib.dump(modelo_bio, 'modelo_biometrico.pkl')
    joblib.dump(modelo_comp, 'modelo_comportamental.pkl')