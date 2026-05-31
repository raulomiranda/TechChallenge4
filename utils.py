import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler

try:
    from imblearn.over_sampling import SMOTE
    _SMOTE_AVAILABLE = True
except ImportError:
    _SMOTE_AVAILABLE = False

class GroupObesityClasses(BaseEstimator, TransformerMixin):
    def fit(self, df, y=None): return self
    def transform(self, df):
        df = df.copy()
        if 'Classe_peso_corporal' not in df.columns: return df
        def agrupar(classe):
            if not isinstance(classe, str): return classe
            if classe in ["Insufficient_Weight", "Normal_Weight"]:
                return "Peso Normal"
            elif classe in ["Overweight_Level_I", "Overweight_Level_II"]:
                return "Sobrepeso"
            elif classe == "Obesity_Type_I": return "Obeso tipo 1"
            elif classe == "Obesity_Type_II": return "Obeso tipo 2"
            elif classe == "Obesity_Type_III": return "Obeso tipo 3"
            return classe
        df['Classe_peso_corporal'] = df['Classe_peso_corporal'].apply(agrupar)
        return df

class DropFeatures(BaseEstimator, TransformerMixin):
    def __init__(self, drop=['Peso', 'Altura']): self.drop = drop
    def fit(self, df, y=None): return self
    def transform(self, df):
        df = df.copy()
        cols = [f for f in self.drop if f in df.columns]
        if cols: return df.drop(cols, axis=1)
        return df

class OneHotEncodingNames(BaseEstimator, TransformerMixin):
    def __init__(self, cols=['Meio_transporte_principal']): 
        self.cols = cols
        self.enc = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    def fit(self, df, y=None): 
        self.enc.fit(df[self.cols])
        return self
    def transform(self, df):
        df = df.copy()
        oh = pd.DataFrame(self.enc.transform(df[self.cols]), columns=self.enc.get_feature_names_out(self.cols), index=df.index)
        df_rest = df.drop(self.cols, axis=1)
        return pd.concat([oh, df_rest], axis=1)

class OrdinalFeature(BaseEstimator, TransformerMixin):
    def fit(self, df, y=None): return self
    def transform(self, df):
        df = df.copy()
        map_genero = {"F": 1, "M": 0}
        map_sim_nao = {"yes": 1, "no": 0}
        map_frequencia = {"no": 0, "Sometimes": 1, "Frequently": 2, "Always": 3}
        if 'Genero' in df.columns and df['Genero'].dtype == 'object':
            df['Genero'] = df['Genero'].map(map_genero)
        colunas_binarias = ['Historico_familiar_sobrepeso', 'Consumo_alimentos_hipercaloricos', 'Fumante', 'Monitora_ingestao_calorica']
        for col in colunas_binarias:
            if col in df.columns and df[col].dtype == 'object':
                df[col] = df[col].map(map_sim_nao)
        for col in ['Consumo_entre_refeicoes', 'Consumo_alcoolico']:
            if col in df.columns and df[col].dtype == 'object':
                df[col] = df[col].map(map_frequencia)
        return df.fillna(0)

class MinMax(BaseEstimator, TransformerMixin):
    def __init__(self, cols=None):
        self.cols = cols or ['Idade', 'Frequencia_consumo_vegetais', 'Numero_refeicoes_diarias', 'Consumo_agua_diario', 'Frequencia_semanal_atividade_fisica', 'Tempo_diario_dispositivos_eletronicos']
        self.scaler = MinMaxScaler()
    def fit(self, df, y=None): 
        cols_to_fit = [c for c in self.cols if c in df.columns]
        if cols_to_fit: self.scaler.fit(df[cols_to_fit])
        return self
    def transform(self, df):
        df = df.copy()
        cols_to_transform = [c for c in self.cols if c in df.columns]
        if cols_to_transform:
            try:
                df[cols_to_transform] = self.scaler.transform(df[cols_to_transform])
            except: pass
        return df

class Oversample(BaseEstimator, TransformerMixin):
    def __init__(self, seed=42): self.seed = seed
    def fit(self, df, y=None): return self
    def transform(self, df):
        # Só aplica SMOTE se a coluna alvo estiver presente (treinamento)
        if 'Classe_peso_corporal' in df.columns and _SMOTE_AVAILABLE:
            sm = SMOTE(sampling_strategy='not majority', random_state=self.seed)
            X_res, y_res = sm.fit_resample(df.drop('Classe_peso_corporal', axis=1), df['Classe_peso_corporal'])
            return pd.concat([X_res.reset_index(drop=True), y_res.reset_index(drop=True)], axis=1)
        return df
