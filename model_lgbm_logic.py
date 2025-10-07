import lightgbm as lgb
import pandas as pd
import numpy as np
from typing import List

# Define as features, na ordem correta
FEATURE_NAMES = [
    'COD_UF_EMIT', 'TIP_FIN_NFE', 'CEST_COMPLETO', 'COD_CST', 'NCM_COMPLETO', 'CFOP_COMPLETO', 'EMIT_CNAE_COMPLETO', 'EMIT_CRT', 
    'EMIT_IND_SN', 'DEST_CNAE_COMPLETO', 'DEST_SITUACAO', 'DEST_IND_SN', 'DEST_POSSUI_IE', 'EMIT_CNAE_DIVISAO', 'EMIT_CNAE_GRUPO', 
    'EMIT_CNAE_CLASSE', 'DEST_CNAE_DIVISAO', 'DEST_CNAE_GRUPO', 'DEST_CNAE_CLASSE', 'NCM_CAPITULO', 'NCM_POSICAO', 'NCM_SUBPOSICAO', 
    'CFOP_NATUREZA', 'CFOP_OPERACAO', 'POSSUI_CEST', 'CEST_SEGMENTO', 'CEST_ITEM']

# Features que devem ser tratadas como Categóricas (para corrigir o erro)
CATEGORICAL_FEATURES = [
    'COD_UF_EMIT', 'TIP_FIN_NFE', 'CEST_COMPLETO', 'COD_CST', 'NCM_COMPLETO', 'CFOP_COMPLETO', 'EMIT_CNAE_COMPLETO', 'EMIT_CRT', 
    'EMIT_IND_SN', 'DEST_CNAE_COMPLETO', 'DEST_SITUACAO', 'DEST_IND_SN', 'DEST_POSSUI_IE', 'EMIT_CNAE_DIVISAO', 'EMIT_CNAE_GRUPO', 
    'EMIT_CNAE_CLASSE', 'DEST_CNAE_DIVISAO', 'DEST_CNAE_GRUPO', 'DEST_CNAE_CLASSE', 'NCM_CAPITULO', 'NCM_POSICAO', 'NCM_SUBPOSICAO', 
    'CFOP_NATUREZA', 'CFOP_OPERACAO', 'POSSUI_CEST', 'CEST_SEGMENTO', 'CEST_ITEM'
]


# TODO: ajustar antes de executar
MODEL_FILE_PATH = "modelo_lgbm 3.txt"
DATA_FILE_PATH = "base_para_teste 4.csv"

# Mapeamento de exemplo para as classes (Y)
CLASSIFICATION_MAP = {
    0: "0 -> ICMS ANT",
    1: "1 -> ICMS ANTEF",
    2: "2 -> ICMS DIFAL",
    3: "3 -> ICMS ST",
    4: "4 -> ICMS STDIF"
}


def predict_model(data_df: pd.DataFrame) -> List[str]:
    """Carrega o modelo LightGBM e realiza a classificação dos dados."""
    
    try:
        bst = lgb.Booster(model_file=MODEL_FILE_PATH)
    except lgb.basic.LightGBMError as e:
        print(f"ERRO: Não foi possível carregar o modelo. Erro: {e}")
        return ["ERRO_MODELO"] * len(data_df)
    
    # Seleção de features e cópia
    dados_para_predicao = data_df[FEATURE_NAMES].copy()

    # Correção: Conversão para tipo 'category'
    for col in CATEGORICAL_FEATURES:
        if col in dados_para_predicao.columns:
            # Converte para string primeiro para tratar NaN e tipos mistos consistentemente
            dados_para_predicao[col] = dados_para_predicao[col].astype(str).astype('category')
    
    try:
        print("DEBUG: Dados para predição (após conversão):")
        print(dados_para_predicao.dtypes)
        print(dados_para_predicao.head())
        probabilidades = bst.predict(dados_para_predicao)
        # argmax retorna o índice (0 a 4) da classe com maior probabilidade
        if probabilidades.ndim == 1:
            # fallback binário, mas esperado multiclass
            classificacao_y_indices = (probabilidades >= 0.5).astype(int)
        else:
            classificacao_y_indices = np.argmax(probabilidades, axis=1)

        print(f"Probabilidades: {probabilidades}")
        classificacao_y_indices = np.argmax(probabilidades, axis=1)
        
        # Mapeamento do índice para o nome do imposto
        classificacao_y_nomes = [CLASSIFICATION_MAP.get(idx, f"Classe_{idx}") for idx in classificacao_y_indices]
        return classificacao_y_nomes
    except Exception as e:
        print(f"ERRO: Falha na predição. {e}")
        return ["ERRO_PREDICAO"] * len(data_df)

def load_data() -> pd.DataFrame:
    """Carrega o arquivo CSV e garante que a coluna SEQ_NFE é um inteiro."""
    try:
        # Tenta carregar o arquivo CSV
        df = pd.read_csv(DATA_FILE_PATH, sep=';')
        
        # Se o cabeçalho estiver incorreto (baseado no snippet), tenta corrigir
        if df.columns[0] != 'SEQ_NFE' and df.shape[0] > 0:
            df = pd.read_csv(DATA_FILE_PATH, sep=';', header=None, skiprows=1)
            header = pd.read_csv(DATA_FILE_PATH, sep=';', nrows=1).columns.tolist()
            df.columns = header

        # ==========================================================
        # LINHAS DE DEBUGGING A ADICIONAR
        print("-" * 50)
        print("DEBUG: Colunas do DataFrame após o carregamento:", df.columns.tolist())
        if 'SEQ_NFE' not in df.columns:
             # Isso é crucial se a coluna estiver com nome errado, como 'SEQ_NFE '
             print("ERRO DE COLUNA: SEQ_NFE não encontrado. Colunas encontradas:", df.columns.tolist())
             print("DEBUG: Tentando renomear a coluna mais próxima de SEQ_NFE.")
             # Tenta renomear a primeira coluna se o nome estiver sujo (e.g., espaço)
             if df.columns[0].strip() == 'SEQ_NFE':
                 df.rename(columns={df.columns[0]: 'SEQ_NFE'}, inplace=True)
        print("-" * 50)
        # ==========================================================

        # Garantir que SEQ_NFE é um tipo numérico (int) para a busca.
        if 'SEQ_NFE' in df.columns:
            df['SEQ_NFE'] = pd.to_numeric(df['SEQ_NFE'], errors='coerce').fillna(-1).astype(int)
        
        # Filtrar apenas as colunas necessárias para evitar erros de leitura
        required_cols = ['SEQ_NFE'] + FEATURE_NAMES
        df = df[[col for col in required_cols if col in df.columns]]
        df = df[df['SEQ_NFE'] != -1] # Remove linhas com erro na conversão de SEQ_NFE
        
        return df
    
    except FileNotFoundError:
        print(f"ERRO FATAL: Arquivo de dados não encontrado: '{DATA_FILE_PATH}'.")
        return pd.DataFrame()
    except Exception as e:
        print(f"ERRO FATAL: Falha ao carregar ou processar os dados: {e}")
        return pd.DataFrame()