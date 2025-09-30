import pandas as pd
from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from typing import List, Union, Optional

# Importar lógica do modelo
import model_lgbm_logic as model_logic

app = FastAPI()
templates = Jinja2Templates(directory="templates")

# Variável global para armazenar os dados carregados
data_store: pd.DataFrame = None

# Pydantic Model para as linhas de dados (X) enviadas para a predição
class FeatureRow(BaseModel):
    # O Pydantic irá tentar converter o tipo de dado recebido (útil para o modelo)
    SEQ_NFE: int 
    COD_UF_EMIT: Optional[str]
    TIP_FIN_NFE: Union[int, str]
    COD_CEST: Optional[int]
    COD_CST: Union[int, str]
    COD_NCM: Union[int, str]
    COD_CFOP: Union[int, str]
    EMIT_COD_CNAE: Union[int, str, None]
    EMIT_CRT: Union[int, str]
    EMIT_IND_SN: Union[str, None]
    DEST_CNAE_PRINC: Union[int, str, None]
    DEST_POSSUI_IE: Union[int, str]
    DEST_SIMPLES: Union[str, None]
    # 'key' é adicionado apenas no frontend para mapear o retorno
    key: str 

class PredictionRequest(BaseModel):
    data: List[FeatureRow]


@app.on_event("startup")
async def startup_event():
    """Carrega os dados da planilha no startup do servidor."""
    global data_store
    print("Iniciando o carregamento dos dados...")
    data_store = model_logic.load_data()
    print(f"Total de linhas carregadas: {len(data_store)}.")


@app.get("/", response_class=HTMLResponse)
async def home_page(request: Request):
    """Tela inicial de pesquisa."""
    return templates.TemplateResponse("search_form.html", {"request": request})


@app.post("/search", response_class=HTMLResponse)
async def search_nfe(request: Request, seq_nfe: str = Form(...)):
    """Busca o SEQ_NFE no DataFrame e exibe a tela de edição."""
    
    debug_line = 0

    debug_line += 1
    print(f"[DEBUG {debug_line}] Iniciando busca por SEQ_NFE = {seq_nfe}")
    if data_store is None or data_store.empty:
        return templates.TemplateResponse(
            "error_page.html", 
            {"request": request, "message": "Dados não carregados. Verifique o console do servidor."}, 
            status_code=500
        )
        
    debug_line += 1
    print(f"[DEBUG {debug_line}] Iniciando busca por SEQ_NFE = {seq_nfe}")
    try:
        seq_nfe_int = int(seq_nfe.strip())
    except ValueError:
        return templates.TemplateResponse(
            "search_form.html", 
            {"request": request, "error": f"O valor '{seq_nfe}' não é um número válido para SEQ_NFE."}, 
            status_code=400
        )

    results_df = data_store[data_store['SEQ_NFE'] == seq_nfe_int].copy()
    
    debug_line += 1
    print(f"[DEBUG {debug_line}] Iniciando busca por SEQ_NFE = {seq_nfe}")
    if results_df.empty:
        return templates.TemplateResponse(
            "search_form.html", 
            {"request": request, "error": f"Nenhuma nota encontrada com SEQ_NFE = {seq_nfe_int}."}, 
            status_code=404
        )

    debug_line += 1
    print(f"[DEBUG {debug_line}] Iniciando busca por SEQ_NFE = {seq_nfe}")
    # Prepara os dados para o template
    results_df['Tipo Imposto'] = "" # Adiciona a coluna vazia para o resultado
    display_columns = ['SEQ_NFE'] + model_logic.FEATURE_NAMES + ['Tipo Imposto']
    data_for_template = results_df[display_columns].to_dict('records')
    
    # print(data_for_template)    
    return templates.TemplateResponse(
        "editable_features.html", 
        {
            "request": request, 
            "data": data_for_template, 
            "features": model_logic.FEATURE_NAMES,
            "seq_nfe": seq_nfe
        }
    )


@app.post("/predict")
async def predict_rows(prediction_request: PredictionRequest):
    """Recebe os dados editados, executa o modelo e retorna o JSON com as predições."""
    
    if not prediction_request.data:
        return {"error": "Nenhum dado enviado para predição."}

    # 1. Converter Pydantic models para DataFrame
    data_list = [row.dict(exclude={'key'}) for row in prediction_request.data]
    prediction_df = pd.DataFrame(data_list)
    
    # 2. Executar a predição
    classificacao_y = model_logic.predict_model(prediction_df)
    
    # 3. Mapear predições de volta para o identificador (key) do frontend
    results = {}
    for i, row_data in enumerate(prediction_request.data):
        # A chave 'key' é o identificador único criado no frontend
        row_key = row_data.key 
        results[row_key] = classificacao_y[i]
        
    return {"predictions": results} # Retorna JSON para o JavaScript