Biblioteca	Funcionalidade Principal

`Pandas` leitura, limpeza e manipulação dos dados
`Streamlit`	Interface web completa com tema dark e texto centralizado.
`RapidFuzz`	Busca aproximada (fuzzy search) de alta performance.
`spaCy`	Pré-processamento de texto (lematização e stop word removal) para aumentar a precisão da busca.
`Plotly` Express	Visualização de dados interativa dos resultados.
`Scikit-learn` (LR)	Modelo de Machine Learning para predição da nota do filme.
`PIL (Pillow)`	Exibição de placeholders de imagem, pronto para carregar os pôsteres reais.
`Requests` a par do Pil ela trabalhou com a API, ou seja, foo buscar imagens de filmes e carregar no projecto.
`Datetime`, para colocar a data actual no footer, toda vez que os anos forem passando, vai actualizar de forma automática.
- `requests`  – para fazer chamadas à API
- `api`       – "https://image.tmdb.org/t/p/w500" 


Título: The Best Movie

Este é um projeto em Python, com mecanismo de busca para encontrar os melhores filmes do IMDB usando Fuzzy Search e spaCy para busca e precisão..

## Estrutura do Projeto
- `index.ipynb`             - Pasta para importar e limpar os dados
- `app.py`                  - Pasta principal do projecto, interface da aplicação feita com Streamlit
- `requirements.txt`        - Lista de dependências do projeto

- `.streamlit`              - Pasta para estilização
- `config.toml`             - Ficheiro com a para dar cores ao projecto




## Como correr o projeto

1. Abrir um terminal
2. Vai até à pasta do projeto
```bash 
cd final_project
```
3. Instalar as dependências
```bash 
pip install -r requirements.txt
```
4. Executar a aplicação: 
```bash 
streamlit run app.py
```

A aplicação será automaticamente aberta no browser

## Dependências

As principais bibliotecas usadas são:

Todas as dependências estão listadas em `requirements.txt`

caso tenha dificuldades, baixe o .venv, cujo a sua função principal é o isolamento de dependências


Usei o Wratio porque é o que torna a aplicação uma verdadeira ferramenta de busca aproximada (fuzzy search). Ele é qye ajuda a pesquisa inteligente, buscando semelhanças aproximadas.
# The-Best-Movie
