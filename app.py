import streamlit as st
import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ==========================================
# 1. Configuração da Página e UI
# ==========================================
st.set_page_config(
    page_title="HealthPredict - Triagem de Obesidade",
    page_icon="🩺",
    layout="wide"
)

# ==========================================
# 2. Carregamento dos Modelos (Em Cache)
# ==========================================
@st.cache_resource
def load_models():
    try:
        # Substitua pelos nomes reais dos arquivos gerados no seu notebook
        modelo_bio = joblib.load('modelos/modelo_biometrico.pkl')
        modelo_comp = joblib.load('modelos/modelo_comportamental.pkl')
        encoders = joblib.load('modelos/label_encoders_rf.pkl')
        return modelo_bio, modelo_comp, encoders
    except FileNotFoundError:
        return None, None, None

modelo_bio, modelo_comp, encoders = load_models()

@st.cache_data
def load_data():
    df = pd.read_csv('dados/Obesity.csv')
    df = df.rename(columns={
        "Gender": "Genero", 
        "Age": "Idade", 
        "Height": "Altura", 
        "Weight": "Peso", 
        "family_history": "Historico_familiar_sobrepeso", 
        "FAVC": "Consumo_alimentos_hipercaloricos", 
        "FCVC": "Frequencia_consumo_vegetais", 
        "NCP": "Numero_refeicoes_diarias", 
        "CAEC": "Consumo_entre_refeicoes", 
        "SMOKE": "Fumante", 
        "CH2O": "Consumo_agua_diario", 
        "SCC": "Monitora_ingestao_calorica", 
        "FAF": "Frequencia_semanal_atividade_fisica", 
        "TUE": "Tempo_diario_dispositivos_eletronicos", 
        "CALC": "Consumo_alcoolico", 
        "MTRANS": "Meio_transporte_principal",
        "Obesity": "Classe_peso_corporal"
    })
    colunas_escala = ["Frequencia_consumo_vegetais", "Numero_refeicoes_diarias", "Consumo_agua_diario", "Frequencia_semanal_atividade_fisica", "Tempo_diario_dispositivos_eletronicos"]
    for col in colunas_escala:
        df[col] = df[col].round().astype(int)
    return df

# ==========================================
# 3. Estrutura do App (Tabs)
# ==========================================
st.title("🩺 HealthPredict: Assistente de Diagnóstico e Triagem")
st.markdown("Sistema de apoio à decisão médica focado na predição de níveis de obesidade através de marcadores biométricos, comportamentais e genéticos.")

tab1, tab2 = st.tabs(["📊 Triagem de Pacientes (Predição)", "📈 Painel Analítico (Insights)"])

# ------------------------------------------
# TAB 1: PREDIÇÃO E FORMULÁRIO
# ------------------------------------------
with tab1:
    st.header("Simulação de Triagem")
    
    # Seleção da Estratégia de Modelagem
    estrategia = st.radio(
        "Selecione a abordagem preditiva:",
        ["Abordagem Comportamental (Sem Peso/Altura)", "Abordagem Biométrica (Com Peso/Altura)"],
        help="A abordagem comportamental prevê o risco baseado apenas em hábitos e genética."
    )
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    # Coleta de Dados do Paciente
    with col1:
        st.subheader("Perfil e Genética")
        idade = st.number_input("Idade", min_value=14, max_value=100, value=25)
        hist_familiar = st.selectbox("Histórico Familiar de Excesso de Peso?", ["Não", "Sim"])
        
        if "Biométrica" in estrategia:
            peso = st.number_input("Peso (kg)", min_value=30.0, max_value=250.0, value=70.0)
            altura = st.number_input("Altura (m)", min_value=1.20, max_value=2.20, value=1.70)
            
    with col2:
        st.subheader("Hábitos e Estilo de Vida")
        favc = st.selectbox("Consome alimentos hipercalóricos com frequência?", ["Não", "Sim"])
        fcvc = st.slider("Frequência de consumo de vegetais (1=Raramente, 3=Sempre)", 1, 3, 2)
        ncp = st.slider("Número de refeições principais por dia", 1, 4, 3)
        faf = st.slider("Dias na semana com atividade física (0=Nenhum, 3=Alto)", 0, 3, 1)

    # Processamento e Predição
    if st.button("Realizar Diagnóstico", type="primary"):
        if modelo_bio is None or modelo_comp is None:
            st.error("Erro: Arquivos .pkl dos modelos não encontrados. Certifique-se de exportá-los do Notebook.")
        else:
            with st.spinner('Processando os dados...'):
                # 1. Transformação dos inputs em formato numérico
                map_sim_nao = {"Não": 0, "Sim": 1}
                
                dados_entrada = {
                    'Historico_familiar_sobrepeso': map_sim_nao[hist_familiar],
                    'Consumo_alimentos_hipercaloricos': map_sim_nao[favc],
                    'Frequencia_semanal_atividade_fisica': faf,
                    'Idade': idade,
                    'Frequencia_consumo_vegetais': fcvc,
                    'Numero_refeicoes_diarias': ncp
                }
                
                # 2. Execução baseada na estratégia escolhida
                if "Biométrica" in estrategia:
                    imc = peso / (altura ** 2)
                    dados_entrada['IMC'] = imc
                    df_pred = pd.DataFrame([dados_entrada])
                    predicao_idx = modelo_bio.predict(df_pred)[0]
                    st.info(f"💡 IMC Calculado internamente: {imc:.2f}")
                else:
                    df_pred = pd.DataFrame([dados_entrada])
                    predicao_idx = modelo_comp.predict(df_pred)[0]
                
                # Decodificar predicao
                predicao = encoders['Classe_peso_corporal'].inverse_transform([predicao_idx])[0]
                
                # 3. Exibição do Resultado
                st.success(f"**Classe Prevista pelo Modelo:** {predicao}")
                
                if "Obesity" in predicao:
                    st.warning("⚠️ O paciente foi classificado em uma faixa de obesidade. Recomenda-se encaminhamento para nutricionista e avaliação cardiológica.")
                elif "Overweight" in predicao:
                    st.info("ℹ️ O paciente apresenta sobrepeso. Intervenções iniciais nos hábitos alimentares são recomendadas.")

# ------------------------------------------
# TAB 2: PAINEL ANALÍTICO
# ------------------------------------------
with tab2:
    st.header("Insights para a Equipe Médica")
    st.markdown("""
    Esta seção apresenta os principais achados da nossa análise de dados base, 
    ajudando a equipe a entender quais fatores (Features de Ouro) mais influenciam o ganho de peso.
    """)
    
    df_dados = load_data()
    df_plot_dados = df_dados.copy()
    
    # Dicionário de tradução dos labels
    map_classes = {
        'Insufficient_Weight': 'Abaixo do Peso', 
        'Normal_Weight': 'Peso Normal', 
        'Overweight_Level_I': 'Sobrepeso I', 
        'Overweight_Level_II': 'Sobrepeso II', 
        'Obesity_Type_I': 'Obesidade I', 
        'Obesity_Type_II': 'Obesidade II', 
        'Obesity_Type_III': 'Obesidade III'
    }
    df_plot_dados['Classe_peso_corporal'] = df_plot_dados['Classe_peso_corporal'].map(map_classes)
    ordem_classes = list(map_classes.values())
    
    map_hist = {'yes': 'Há histórico', 'no': 'Não há'}
    df_plot_dados['Historico_familiar_sobrepeso'] = df_plot_dados['Historico_familiar_sobrepeso'].map(map_hist)
    
    map_cons = {'yes': 'Sim', 'no': 'Não'}
    df_plot_dados['Consumo_alimentos_hipercaloricos'] = df_plot_dados['Consumo_alimentos_hipercaloricos'].map(map_cons)
    
    map_ativ = {0: 'Nenhuma', 1: '1-2x/sem', 2: '3-4x/sem', 3: '5x/sem ou mais'}
    df_plot_dados['Frequencia_semanal_atividade_fisica'] = df_plot_dados['Frequencia_semanal_atividade_fisica'].map(map_ativ)
    
    col_dash1, col_dash2 = st.columns(2)
    
    with col_dash1:
        st.subheader("1. Histórico Familiar vs Obesidade")
        fig1, ax1 = plt.subplots(figsize=(8, 5))
        df_cont1 = pd.crosstab(df_plot_dados['Classe_peso_corporal'], df_plot_dados['Historico_familiar_sobrepeso'])
        df_perc1 = df_cont1.div(df_cont1.sum(axis=1), axis=0) * 100
        df_plot1 = df_perc1.reset_index().melt(id_vars='Classe_peso_corporal')
        sns.barplot(data=df_plot1, x='Classe_peso_corporal', y='value', hue='Historico_familiar_sobrepeso', order=ordem_classes, palette='viridis', ax=ax1)
        ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45, ha='right')
        ax1.set_ylabel('Porcentagem (%)')
        ax1.set_xlabel('Classe de Peso')
        plt.tight_layout()
        st.pyplot(fig1)

        st.subheader("2. Idade por Classe de Peso")
        fig2, ax2 = plt.subplots(figsize=(8, 5))
        sns.boxplot(data=df_plot_dados, x='Classe_peso_corporal', y='Idade', order=ordem_classes, palette='magma', ax=ax2)
        ax2.set_xticklabels(ax2.get_xticklabels(), rotation=45, ha='right')
        ax2.set_xlabel('Classe de Peso')
        plt.tight_layout()
        st.pyplot(fig2)
        
    with col_dash2:
        st.subheader("3. Consumo Hipercalórico vs Obesidade")
        fig3, ax3 = plt.subplots(figsize=(8, 5))
        df_cont3 = pd.crosstab(df_plot_dados['Classe_peso_corporal'], df_plot_dados['Consumo_alimentos_hipercaloricos'])
        df_perc3 = df_cont3.div(df_cont3.sum(axis=1), axis=0) * 100
        df_plot3 = df_perc3.reset_index().melt(id_vars='Classe_peso_corporal')
        sns.barplot(data=df_plot3, x='Classe_peso_corporal', y='value', hue='Consumo_alimentos_hipercaloricos', order=ordem_classes, palette='viridis', ax=ax3)
        ax3.set_xticklabels(ax3.get_xticklabels(), rotation=45, ha='right')
        ax3.set_ylabel('Porcentagem (%)')
        ax3.set_xlabel('Classe de Peso')
        plt.tight_layout()
        st.pyplot(fig3)
        
        st.subheader("4. Atividade Física vs Obesidade")
        fig4, ax4 = plt.subplots(figsize=(8, 5))
        df_cont4 = pd.crosstab(df_plot_dados['Classe_peso_corporal'], df_plot_dados['Frequencia_semanal_atividade_fisica'])
        df_perc4 = df_cont4.div(df_cont4.sum(axis=1), axis=0) * 100
        df_plot4 = df_perc4.reset_index().melt(id_vars='Classe_peso_corporal')
        sns.barplot(data=df_plot4, x='Classe_peso_corporal', y='value', hue='Frequencia_semanal_atividade_fisica', order=ordem_classes, palette='viridis', ax=ax4)
        ax4.set_xticklabels(ax4.get_xticklabels(), rotation=45, ha='right')
        ax4.set_ylabel('Porcentagem (%)')
        ax4.set_xlabel('Classe de Peso')
        plt.tight_layout()
        st.pyplot(fig4)
    
    st.markdown("---")
    st.write("**Nota Técnica:** O modelo preditivo atual omite intencionalmente variáveis enviesadas no dataset original (como Gênero e Hábitos de Fumo) para garantir uma predição médica imparcial e focada nos hábitos de vida.")