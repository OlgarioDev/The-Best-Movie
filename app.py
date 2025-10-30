import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from rapidfuzz import process, fuzz 
import spacy as sp
from PIL import Image 
import requests 
from io import BytesIO
import plotly.express as px
from sklearn.linear_model import LinearRegression as LR
import numpy as np
import datetime


# Configuração inicial do Streamlit
st.set_page_config(page_title="The Best Movie", layout="wide")


# Centraliza o texto e aplica o tema dark (complementando o config.toml)
st.markdown(
    """
    <style>
    /* Aplica o Fundo Escuro */
    .stApp {
        background-color: #0E1117; 
        color: #FFFFFF; 
    }
    /* Centralização de Títulos e Textos */
    h1, h2, h3, .stMarkdown {
        text-align: center;
    }
    /* Centraliza a caixa de busca e botão */
    .stTextInput {
        width: 60%; 
        margin: 0 auto; 
    }
    .stDataFrame {
        margin: 0 auto; 
    }
    /* Outros estilos do seu tema */
    :root {
        --primary-color: #E94560; /* Neon Pink/Brilhante */
        --secondary-background-color: #262730; 
    }
    
    /* Ajusta o padding superior para alinhar os botões com o campo de busca */
        
    .search_col, .button_col, .clear_top_col {
        padding-top: 29px; 
    }
    </style>
    """,
    unsafe_allow_html=True
)

def scroll_to_top():
    
    components.html(
        # ... (código do JavaScript/HTML) ...
        """
        <script>
        
        </script>
        """,
        height=0,
        width=0,
    )

# ... (restante das suas funções, como load_spacy_model) ...




def clear_search_input():
    
    """Função de callback para limpar o campo de pesquisa e rolar para o topo."""
    # A ordem agora é: 1. Rola; 2. Limpa o estado.
    
    scroll_to_top() 
    st.session_state.search_query_input = "" # Esta linha força a re-execução do Streamlit




# API para buscar posters no TMDb

TMDB_IMAGE_BASE_URL = "https://image.tmdb.org/t/p/w500" 
TMDB_API_KEY = "e7237188da61192e1afb5736111b16db" 



# Inicialização de sessão para o campo de busca
if 'search_query_input' not in st.session_state:
    st.session_state['search_query_input'] = "Titanic" # padrão inicial



@st.cache_resource
def load_spacy_model():
    
    """Carrega o modelo spaCy de forma eficiente."""
    try:
        return sp.load("en_core_web_sm")
    except OSError:
        st.error("O modelo 'en_core_web_sm' do spaCy não foi encontrado. Execute: `python -m spacy download en_core_web_sm`")
        return None


# Funções de Machine Learning


@st.cache_resource
def train_linear_model(df):
    """Treina um modelo de Regressão Linear."""
    df_clean = df.dropna(subset=['Watch Time', 'Movie Rating'])
    X = df_clean['Watch Time'].values.reshape(-1, 1)
    Y = df_clean['Movie Rating'].values
    model = LR()
    model.fit(X, Y)
    return model

def predict_rating(model, runtime):
    """Faz a predição da nota para uma dada duração."""
    prediction = model.predict(np.array([[runtime]]))
    return prediction[0]




# --- FUNÇÕES DE CARREGAMENTO E PRÉ-PROCESSAMENTO (PANDAS/SPACY) ---


@st.cache_data
def load_data():
    """Carrega o DataFrame dos filmes."""
    try:
        df = pd.read_csv('dataset/cleaned_top_1000_imdb_movies.csv')
    except FileNotFoundError:
        try:
            # Cenário alternativo: CASO não encontre o CSV limpo, tenta carregar o CSV original e limpar.
            df = pd.read_csv('top_1000_imdb_movies.csv')
            df = df.drop(columns=['Unnamed: 0'], errors='ignore')
            df = df.rename(columns={'Title': 'Movie Name', 'Rating': 'Movie Rating'})
            df['Year of Release'] = df['Year of Release'].astype(str).str.extract(r'(\d{4})', expand=False).astype(int)
           
        except FileNotFoundError:
           
             return pd.DataFrame()
    return df

# Objeto nlp e DataFrame
nlp = load_spacy_model()
df_movies = load_data()


def preprocess_text(text):
    
    """Lematiza, remove stop words e pontuação usando spaCy."""
    
    if not nlp or pd.isna(text): 
        return str(text).lower()
    doc = nlp(str(text).lower())
    tokens = [
        token.lemma_ 
        for token in doc 
        if token.is_alpha and not token.is_stop
    ]
    return " ".join(tokens)

@st.cache_data
def get_processed_names(df):
    
    """Gera a lista de nomes de filmes pré-processados (cacheada)."""
    
    return df['Movie Name'].apply(preprocess_text).tolist()


@st.cache_data
def search_movies(query, df, limit=10, threshold=70):
    
    """Realiza a busca aproximada (Fuzzy Search) dos filmes."""
    
    if df.empty or not query:
        return pd.DataFrame()

    processed_query = preprocess_text(query)
    processed_movie_names = get_processed_names(df)
    
    search_list = processed_movie_names if processed_query else df['Movie Name'].tolist()
    search_query = processed_query if processed_query else query

    
    results = process.extract(
        search_query, 
        search_list, 
        limit=limit, 
        scorer=fuzz.WRatio #Usado WRatio para melhor precisão na similaridade da pesquisa.
    )

    # Aplica o filtro de threshold manualmente
    results = [(name, score, index) for name, score, index in results if score >= threshold]

    if not results:
        return pd.DataFrame()

    matched_indices = [index for _, score, index in results]
    results_df = df.iloc[matched_indices].copy()
    score_map = {df.iloc[idx]['Movie Name']: score for _, score, idx in results}
    results_df['Similaridade (%)'] = results_df['Movie Name'].apply(lambda x: score_map.get(x, 0))
    
    results_df = results_df.sort_values(
        by=['Similaridade (%)', 'Movie Rating'], 
        ascending=[False, False]
    ).reset_index(drop=True)

    return results_df

# Adicionar esta função de busca por imagem: Import Requests

@st.cache_data(show_spinner=False)
def search_movie_tmdb(movie_name):
    
    
    if not TMDB_API_KEY or TMDB_API_KEY == "SUA_CHAVE_AQUI":
        return None 

    url = "https://api.themoviedb.org/3/search/movie"
    params = {
        'api_key': TMDB_API_KEY,
        'query': movie_name,
        'language': 'en-US' # Use en-US para resultados mais consistentes na busca
    }

    try:
        response = requests.get(url, params=params, timeout=5)
        response.raise_for_status() 
        data = response.json()
        
        # Pega o poster_path do primeiro resultado
        
        if data.get('results') and data['results'][0].get('poster_path'):
            return data['results'][0]['poster_path']
        if data.get('results') and data['results'][0].get('poster_path'):
            return data['results'][0]['poster_path']
        return None
    except requests.exceptions.RequestException as e:

        st.warning(f"Erro ao buscar o poster no TMDb: {e}")
        pass
        
    return None

@st.cache_data(show_spinner=False)
def fetch_poster_image(movie_name):
    """
    Busca o poster real do filme no TMDb.
    """
    poster_path = search_movie_tmdb(movie_name)
    
    if poster_path:
        full_url = TMDB_IMAGE_BASE_URL + poster_path
        
        try:
            # Faz a requisição para a imagem
            image_response = requests.get(full_url, timeout=5)
            image_response.raise_for_status()
            
            # Converte o conteúdo binário em objeto Image do PIL
            return Image.open(BytesIO(image_response.content))
            
        except requests.exceptions.RequestException:
            # Em caso de falha no download da imagem, retorna o placeholder
            pass 
            
    # Placeholder se a chave não estiver configurada ou a busca falhar
    return Image.new('RGB', (100, 150), color='#E94560')

# --- INTERFACE STREAMLIT ---

st.title("IMDB MOVIE SEARCH🎬🔍")
st.markdown("🍿Este é um mecanismo de busca para encontrar os melhores filmes do **IMDB** usando **Fuzzy Search** e **spaCy** para busca e precisão.")

# --- BOTÕES E CAMPO DE BUSCA (ÁREA SUPERIOR) ---
search_col, button_col, clear_top_col = st.columns([4, 0.5, 0.5])



with search_col:
    search = st.text_input(
        label="🔍 Pesquisar:",
        key='search_query_input',
        label_visibility="collapsed",
        placeholder="Digite o nome do filme...",
        max_chars=100
        
    )

    
with button_col:
    # Ajuste vertical para alinhar com o campo de texto
    st.markdown("<div style='margin-top: 5px;'></div>", unsafe_allow_html=True) 
    search_button = st.button("🔍Buscar")
    

with clear_top_col:
    st.markdown("<div style='margin-top: 5px;'></div>", unsafe_allow_html=True) 
    st.button("🗑️ Limpar", on_click=clear_search_input, key='clear_button_top') 
    
# --------------------------------------------------

# Parâmetros de busca
with st.expander("🛠️ Configurações Avançadas"):
    col1, col2 = st.columns(2)
    with col1:
        result_limit = st.slider("Máximo de resultados", min_value=1, max_value=20, value=10)
    with col2:
        similarity_threshold = st.slider(
            "Similaridade Mínima (%)", 
            min_value=50, 
            max_value=100, 
            value=70,
            help="Resultados com pontuação de similaridade abaixo deste valor serão descartados."
        )

# --- Execução da Busca ---

search_term = search.strip()
results_df = pd.DataFrame()

# A busca só é executada se o botão 'Buscar! e enter' for clicado E houver texto
if (search_button or (search_term != "Titanic" and not 'initial_run' in st.session_state)) and search_term:
    st.session_state['initial_run'] = False # Marca que não é a primeira execução
    
    with st.spinner(f"A procurar filmes semelhantes a '{search_term}'..."):
        results_df = search_movies(
            search_term, 
            df_movies, 
            limit=result_limit,
            threshold=similarity_threshold
        )
    
    if not results_df.empty:
        st.success(f"Encontrados {len(results_df)} resultados:")
        
        # --- 1. MELHOR RESULTADO EM DESTAQUE (com PIL Placeholder) ---
        st.subheader("Melhor Correspondência:")
        col_img, col_data = st.columns([1, 4])
        
        primeiro_filme = results_df.iloc[0]
        poster_img = fetch_poster_image(primeiro_filme['Movie Name'])

        with col_img:
            st.image(poster_img, caption=primeiro_filme['Movie Name'])
            
        with col_data:
             st.markdown(f"**Filme:** {primeiro_filme['Movie Name']} ({primeiro_filme['Year of Release']})")
             st.markdown(f"**Nota IMDb:** {primeiro_filme['Movie Rating']} ⭐")
             st.markdown(f"**Similaridade:** {int(primeiro_filme['Similaridade (%)'])}%")
             st.markdown(f"**Duração:** {primeiro_filme['Watch Time']} min")
             st.markdown(f"**Descrição:** {primeiro_filme['Description'][:200]}...")
             
        st.markdown("---")


        # ---  TABELA DE RESULTADOS ---
        
        st.subheader("Tabela de Resultados:")
        display_cols = ['Similaridade (%)', 'Movie Name', 'Year of Release', 'Movie Rating', 'Watch Time']
        
        st.dataframe(
            results_df[display_cols].style.format({'Similaridade (%)': "{:.0f}"}),
            use_container_width=True,
        )
        
        st.markdown("---")

        # ---  VISUALIZAÇÃO COM PLOTLY ---
        
        st.subheader("Visualização dos Resultados (Movie Rating)")
        fig = px.bar(results_df, x='Movie Name', y='Movie Rating', color='Similaridade (%)', 
                     color_continuous_scale=px.colors.sequential.Plasma,
                     hover_data=['Year of Release', 'Watch Time'],
                     labels={'Movie Name': 'Filme', 'Movie Rating': 'Nota IMDb'},
                     title="Comparação de Notas dos Filmes Encontrados")
        fig.update_layout(xaxis={'categoryorder':'total descending'}, height=500)
        st.plotly_chart(fig, use_container_width=True)
        
    else:
        st.warning(f"Nenhum filme encontrado com o nome '{search_term}'.")

# --- BOTÃO LIMPAR NA BASE ---

st.markdown("---")
col_left, col_center, col_right = st.columns([1.5, 2, 1.5])

with col_center:
    scroll_to_top()
    st.button("⟲ Iniciar Nova Busca", on_click=clear_search_input, key='clear_button_bottom')
st.markdown("---")

# --- ZONA DE APRENDIZAGEM ---

st.subheader("🕛Predição de Nota por Duração")
st.markdown("Use a **Regressão Linear** treinada em todo o conjunto de dados para estimar a nota do filme.")

linear_model = train_linear_model(df_movies)

runtime_input = st.slider(
    "Selecione a Duração (em minutos) do filme para prever a nota:",
    min_value=df_movies['Watch Time'].min(),
    max_value=df_movies['Watch Time'].max(),
    value=120, 
    step=5
)

predicted_rating = predict_rating(linear_model, runtime_input)

st.info(f"Para um filme com **{runtime_input} minutos** de duração, a nota média prevista é de **{predicted_rating:.2f}** ⭐.")


# --- FOOTER ----


current_year = datetime.date.today().year

st.markdown(
    f"""
    <div style="text-align: center; color: #AAAAAA; font-size: 0.9em;">
        © {current_year} The Best Movie. Todos os direitos reservados.
    </div>
    <div style="text-align: center; margin-top: 5px; font-weight: bold;">
        <a href="https://www.linkedin.com/in/olg%C3%a1rio-catanha-107566252/" style="color: #E94560; text-decoration: none;">
            Desenvolvido com Streamlit e RapidFuzz por <span text-decoration: style="font-size: 1.2em;">Olgario Catanha</span>
        </a>
        <br>
        <span style="font-size: 0.75em; color: #888888;">
            Dados fornecidos por IMDb e The Movie Database (TMDb).
        </span>
    </div>
    """, 
    unsafe_allow_html=True
)