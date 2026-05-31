import streamlit as st
import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from utils import GroupObesityClasses, DropFeatures, OneHotEncodingNames, OrdinalFeature, MinMax, Oversample

# Configuração da página
st.set_page_config(page_title="Analisador de Obesidade", layout="wide")

# Título Principal
st.markdown("<h1 style='text-align: center;'> 🩺 Plataforma de Análise de Obesidade </h1>", unsafe_allow_html=True)
st.markdown("---")

# --- Mapeamentos ---
mapa_genero = {"Feminino": "F", "Masculino": "M"}
mapa_sim_nao = {"Sim": "yes", "Não": "no"}
mapa_caec = {"Não": "no", "Às vezes": "Sometimes", "Frequentemente": "Frequently", "Sempre": "Always"}
mapa_calc = {"Não bebe": "no", "Às vezes": "Sometimes", "Frequentemente": "Frequently", "Sempre": "Always"}
mapa_transporte = {
    "Automóvel": "Automobile", "Motocicleta": "Motorbike", "Bicicleta": "Bike",
    "Transporte Público": "Public_Transportation", "A pé": "Walking"
}
mapa_fcvc = {"Raramente": 1, "Às vezes": 2, "Sempre": 3}
mapa_ncp = {"Uma": 1, "Duas": 2, "Três": 3, "Quatro ou mais": 4}
mapa_ch2o = {"< 1 L/dia": 1, "1–2 L/dia": 2, "> 2 L/dia": 3}
mapa_faf = {"Nenhuma": 0, "1–2 x/semana": 1, "3–4 x/semana": 2, "5+ x/semana": 3}
mapa_tue = {"0–2 h/dia": 0, "3–5 h/dia": 1, "> 5 h/dia": 2}

# --- Abas do Aplicativo ---
tab_pred, tab_dash = st.tabs(["🔍 Realizar Predição", "📊 Dashboard de Dados"])

# --- TAB 1: PREDIÇÃO ---
with tab_pred:
    with st.sidebar:
        st.header("Dados Biométricos")
        input_genero_pt = st.radio("Sexo Biológico", list(mapa_genero.keys()), index=0) 
        input_idade = st.slider("Idade em anos", 14, 61, 22)
        input_altura = st.slider("Altura em metros", 1.40, 2.10, 1.65, step=0.01)
        input_peso = st.number_input("Peso em kg", 30.0, 200.0, 70.0, step=0.1)
        
        st.divider()
        st.info("Ajuste os dados e clique em Analisar Perfil.")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🧬 Histórico e Hábitos")
        input_historico_familiar_pt = st.radio("Histórico familiar de excesso de peso?", list(mapa_sim_nao.keys()), index=0, horizontal=True)
        input_favc_pt = st.radio("Consumo frequente de alimentos muito calóricos?", list(mapa_sim_nao.keys()), index=0, horizontal=True)
        
        input_fcvc_pt = st.select_slider("Frequência de consumo de vegetais", options=list(mapa_fcvc.keys()), value="Raramente")
        input_ncp_pt = st.select_slider("Número de refeições principais por dia", options=list(mapa_ncp.keys()), value="Três")
        input_caec_pt = st.select_slider("Consumo de lanches entre as refeições", options=list(mapa_caec.keys()), value="Sempre")
        input_ch2o_pt = st.select_slider("Consumo diário de água", options=list(mapa_ch2o.keys()), value="< 1 L/dia")

    with col2:
        st.subheader("🏃 Estilo de Vida e Rotina")
        input_smoke_pt = st.radio("Hábito de fumar?", list(mapa_sim_nao.keys()), index=1, horizontal=True)
        input_scc_pt = st.radio("Monitora a ingestão calórica diária?", list(mapa_sim_nao.keys()), index=1, horizontal=True)
        
        input_faf_pt = st.select_slider("Frequência semanal de atividade física", options=list(mapa_faf.keys()), value="Nenhuma")
        input_tue_pt = st.select_slider("Tempo diário usando dispositivos eletrônicos", options=list(mapa_tue.keys()), value="> 5 h/dia")
        
        input_calc_pt = st.select_slider("Consumo de bebida alcoólica", options=list(mapa_calc.keys()), value="Às vezes")
        input_meio_transporte_pt = st.selectbox("Meio de transporte habitual", list(mapa_transporte.keys()), index=3)

    st.divider()

    if st.button('🚀 ANALISAR PERFIL', use_container_width=True):
        try:
            try:
                data_saved = joblib.load('modelo_biometria_completo.joblib')
            except:
                data_saved = joblib.load('modelo_obesidade_com_altura.joblib')
                
            preprocessor = data_saved['preprocessor']
            model = data_saved['model']
            feature_names = data_saved['features']
            target_names = data_saved.get('target_names', [])
            
            novo_input = pd.DataFrame([{
                "Genero": mapa_genero[input_genero_pt], "Idade": float(input_idade), "Altura": float(input_altura), "Peso": float(input_peso),
                "Historico_familiar_sobrepeso": mapa_sim_nao[input_historico_familiar_pt], "Consumo_alimentos_hipercaloricos": mapa_sim_nao[input_favc_pt],
                "Frequencia_consumo_vegetais": float(mapa_fcvc[input_fcvc_pt]), "Numero_refeicoes_diarias": float(mapa_ncp[input_ncp_pt]),
                "Consumo_entre_refeicoes": mapa_caec[input_caec_pt], "Fumante": mapa_sim_nao[input_smoke_pt],
                "Consumo_agua_diario": float(mapa_ch2o[input_ch2o_pt]), "Monitora_ingestao_calorica": mapa_sim_nao[input_scc_pt],
                "Frequencia_semanal_atividade_fisica": float(mapa_faf[input_faf_pt]), "Tempo_diario_dispositivos_eletronicos": float(mapa_tue[input_tue_pt]),
                "Consumo_alcoolico": mapa_calc[input_calc_pt], "Meio_transporte_principal": mapa_transporte[input_meio_transporte_pt],
                "Classe_peso_corporal": "Normal_Weight"
            }])

            input_processed = preprocessor.transform(novo_input)
            X_input = input_processed.drop('Classe_peso_corporal', axis=1, errors='ignore')
            for col in feature_names:
                if col not in X_input.columns: X_input[col] = 0.0
            X_input = X_input[feature_names]
            
            raw_pred = model.predict(X_input)[0]
            probs = model.predict_proba(X_input)[0]
            predicao_texto = target_names[raw_pred] if isinstance(raw_pred, (int, np.integer)) and target_names else str(raw_pred)
            
            cor = "#FF4B4B" if "Obeso" in predicao_texto else "#4CAF50"
            st.markdown(f"<div style='text-align: center; padding: 20px; border-radius: 10px; background-color: {cor}22; border: 2px solid {cor};'><h2 style='color: {cor}; margin: 0;'>Resultado Estimado: {predicao_texto}</h2></div>", unsafe_allow_html=True)

            st.divider()
            st.subheader("🌱 Recomendações e Inteligência")
            c_rec1, c_rec2 = st.columns(2)
            with c_rec1:
                st.write("### 🚩 Plano de Ação")
                if mapa_faf[input_faf_pt] < 2: st.warning("**Exercício**: Aumentar a atividade física é essencial.")
                if mapa_ch2o[input_ch2o_pt] < 3: st.info("**Água**: Tente beber mais de 2L de água diariamente.")
                if mapa_caec[input_caec_pt] in ['Always', 'Frequently']: st.error("**Lanches**: Evite beliscar constantemente entre as refeições.")
            with c_rec2:
                st.write("### 📊 Confiança do Modelo")
                prob_df = pd.DataFrame({'Classe': target_names if target_names else model.classes_, 'Probabilidade': probs})
                prob_df = prob_df.sort_values(by='Probabilidade', ascending=False)
                fig, ax = plt.subplots(figsize=(6, 4))
                sns.barplot(x='Probabilidade', y='Classe', data=prob_df, palette='magma', ax=ax)
                st.pyplot(fig)
        except Exception as e:
            st.error(f"Erro ao processar: {e}")

# --- TAB 2: DASHBOARD ---
with tab_dash:
    st.subheader("📊 Visão Geral do Dataset de Obesidade")
    
    @st.cache_data
    def load_clean_data():
        df = pd.read_csv("dados/Obesity.csv")
        df = df.rename(columns={
            "Gender": "Gênero", "Age": "Idade", "Height": "Altura", "Weight": "Peso", 
            "family_history": "Histórico_Familiar", "FAVC": "Comida_Calórica", 
            "FCVC": "Consumo_Vegetais", "NCP": "Refeições_Diárias", 
            "CAEC": "Lanches_Intermediários", "SMOKE": "Fumante", 
            "CH2O": "Consumo_Água", "SCC": "Monitora_Calorias", 
            "FAF": "Atividade_Física", "TUE": "Uso_Eletrônicos", 
            "CALC": "Consumo_Álcool", "MTRANS": "Transporte", "Obesity": "Classe_Peso"
        })
        df['Escala_Obesidade'] = df['Classe_Peso'].map({
            'Insufficient_Weight': 0, 'Normal_Weight': 1, 'Overweight_Level_I': 2,
            'Overweight_Level_II': 3, 'Obesity_Type_I': 4, 'Obesity_Type_II': 5, 'Obesity_Type_III': 6
        })
        return df

    df_dash = load_clean_data()
    
    col_d1, col_d2 = st.columns(2)
    
    with col_d1:
        st.write("### 📍 Dispersão: Peso vs Altura")
        fig_scat, ax_scat = plt.subplots(figsize=(8, 5))
        sns.scatterplot(data=df_dash, x='Peso', y='Altura', hue='Classe_Peso', palette='viridis', alpha=0.6, ax=ax_scat)
        plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., fontsize='small')
        plt.xlabel("Peso (kg)")
        plt.ylabel("Altura (m)")
        st.pyplot(fig_scat)

    with col_d2:
        st.write("### 📈 Impacto Global na Progressão de Peso")
        corr_data = df_dash.select_dtypes(include=[np.number]).corr()['Escala_Obesidade'].drop(['Escala_Obesidade', 'Peso', 'Altura']).sort_values(ascending=False)
        fig_corr, ax_corr = plt.subplots(figsize=(8, 5))
        sns.barplot(x=corr_data.values, y=corr_data.index, palette='RdBu_r', ax=ax_corr)
        plt.xlabel("Força do Impacto (Correlação)")
        plt.ylabel("Variáveis Comportamentais")
        st.pyplot(fig_corr)

    st.divider()
    
    # --- Distribuição por Feature ---
    st.write("### 🔍 Distribuição de Hábitos por Classe de Peso")
    
    features_disponiveis = [
        'Gênero', 'Histórico_Familiar', 'Comida_Calórica', 'Consumo_Vegetais', 
        'Refeições_Diárias', 'Lanches_Intermediários', 'Fumante', 'Consumo_Água', 
        'Monitora_Calorias', 'Atividade_Física', 'Uso_Eletrônicos', 'Consumo_Álcool', 'Transporte'
    ]
    
    feature_selecionada = st.selectbox("Selecione um hábito para analisar:", features_disponiveis)
    
    df_cont = pd.crosstab(df_dash['Classe_Peso'], df_dash[feature_selecionada])
    df_perc = df_cont.div(df_cont.sum(axis=1), axis=0) * 100
    df_plot = df_perc.reset_index().melt(id_vars='Classe_Peso')
    
    ordem_classes = [
        'Insufficient_Weight', 'Normal_Weight', 'Overweight_Level_I', 
        'Overweight_Level_II', 'Obesity_Type_I', 'Obesity_Type_II', 'Obesity_Type_III'
    ]
    
    fig_feat, ax_feat = plt.subplots(figsize=(12, 6))
    sns.barplot(data=df_plot, x='Classe_Peso', y='value', hue=feature_selecionada, order=ordem_classes, palette='magma', ax=ax_feat)
    
    plt.title(f'Distribuição Percentual de {feature_selecionada.replace("_", " ")}', fontsize=14)
    plt.ylabel('Porcentagem (%)', fontsize=12)
    plt.xlabel('Classe de Peso Corporal', fontsize=12)
    plt.xticks(rotation=45)
    plt.legend(title=feature_selecionada.replace("_", " "), bbox_to_anchor=(1.05, 1), loc=2)
    
    for p in ax_feat.patches:
        if p.get_height() > 0:
            ax_feat.annotate(f'{p.get_height():.1f}%', 
                        (p.get_x() + p.get_width() / 2., p.get_height()), 
                        ha = 'center', va = 'center', 
                        xytext = (0, 9), textcoords = 'offset points', fontsize=8)
                        
    st.pyplot(fig_feat)
    
    st.divider()
    
    st.write("### 📅 Estatísticas de Idade por Classe")
    age_stats = df_dash.groupby('Classe_Peso')['Idade'].agg(['mean', 'median']).reset_index()
    age_stats['Classe_Peso'] = pd.Categorical(age_stats['Classe_Peso'], categories=ordem_classes, ordered=True)
    age_stats = age_stats.sort_values('Classe_Peso').melt(id_vars='Classe_Peso', var_name='Métrica', value_name='Anos')
    
    fig_age, ax_age = plt.subplots(figsize=(10, 5))
    sns.barplot(data=age_stats, x='Classe_Peso', y='Anos', hue='Métrica', palette='coolwarm', ax=ax_age)
    plt.xlabel("Classe de Peso Corporal")
    plt.ylabel("Idade (Anos)")
    plt.xticks(rotation=45)
    st.pyplot(fig_age)
    
    st.info("O dashboard acima utiliza os dados brutos do dataset para contextualizar a predição individual.")
