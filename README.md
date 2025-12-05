# Conceitos Machine Learning

## PRÁTICA 1 - Laboratórios com Google Colab

Você está em um ambiente Google Colab e vai trabalhar com três laboratórios. Antes de treinar qualquer modelo de Machine Learning, precisamos aprender a importar os dados, fazer uma análise exploratória detalhada e transformar os dados para que a base esteja ideal para o modelo.

## OBJETIVO DA ATIVIDADE

Aplicar os conceitos fundamentais de preparação de dados para Machine Learning utilizando o Google Colab, explorando as etapas de importação, análise e transformação de variáveis, com foco na construção de um pipeline de dados robusto e reprodutível.
Objetivos específicos:

- Importar dados no ambiente Google Colab

- Explorar o conjunto de dados com o apoio de bibliotecas como Pandas, Seaborn e Matplotlib.

- Aplicar técnicas de codificação de variáveis categóricas, a fim de preparar os dados para a etapa de modelagem preditiva.

## DESAFIO PRÁTICO

Nesta atividade, você irá aplicar e preparar um conjunto de dados para uso posterior em modelos de Machine Learning.
Você utilizará o ambiente Google Colab e deverá seguir as instruções dos laboratórios para reproduzir as ações descritas e registrar os resultados e análises com clareza.


Os seus notebooks devem, no mínimo, conter:

1. Etapa 1
a. Carregamento dos dados

- Realizar o upload do dataset diretamente no Colab (usando files.upload()) ou acessá-lo pelo Google Drive.

- Ler o arquivo em um DataFrame utilizando pandas.read_csv().

b. Verificação inicial do DataFrame

- Apresentar o número de linhas e colunas (.shape) e exibir os primeiros registros (.head()).

c. Registro de observações e decisões

- Em células Markdown, descrever os insights extraídos, análises importantes e qualquer decisão de tratamento aplicada aos dados.

2. Etapa 2

a. Carregamento dos dados

- Upload do dataset no Colab ou leitura a partir do Google Drive.

- Leitura em um DataFrame com pandas.read_csv().

b. Verificação inicial do DataFrame

- Apresentar o número de linhas e colunas (.shape) e exibir os primeiros registros (.head()).

c. Registro de observações e decisões

- Em células Markdown, descrever os insights extraídos e qualquer decisão de tratamento.

d. Classificação de variáveis e tipos de dados


- Exibir informações do DataFrame com .info().

- Classificar as colunas em numéricas e categóricas com base nos tipos (.select_dtypes() ou análise manual).

e. Identificação de valores ausentes

- Utilizar .isnull().sum() para verificar a quantidade de valores nulos por coluna.

- Indicar se alguma ação é necessária (remoção, imputação, etc.).

f. Estatísticas descritivas básicas

- Aplicar .describe() nas variáveis numéricas para calcular média, desvio padrão, mínimo, máximo, etc.

g. Visualização dos dados

- Criar gráficos exploratórios com seaborn ou matplotlib.

3. Etapa 3

a. Carregamento dos dados

- Upload do dataset no Colab ou leitura a partir do Google Drive.

- Leitura em um DataFrame com pandas.read_csv().

b. Verificação inicial do DataFrame

- Apresentar o número de linhas e colunas (.shape) e exibir os primeiros registros (.head()).

c. Registro de observações e decisões

- Em células Markdown, descrever os insights extraídos e qualquer decisão de tratamento aplicada aos dados.

d. Estatísticas descritivas básicas

- Aplicar .describe() nas variáveis numéricas para calcular média, desvio padrão, mínimo, máximo, etc.

e. Visualização dos dados

- Criar gráficos exploratórios com seaborn ou matplotlib.

f. Codificação de variáveis categóricas

- Aplicar Label Encoding ou One-Hot Encoding em colunas categóricas.

- Explicar em Markdown qual método foi escolhido e por quê.

|           Etapa          |                        Ações mínimas requeridas                   |               Funções/Ferramentas chave         |
| ------------------------ | ----------------------------------------------------------------- |------------------------------------------------ |
|  Carregamento seguro     |   Ler todos os .zip para um único DataFrame.                     | pd.read_csv, compression="zip", dtype=           |
|                          | Garantir que separador, decimal e codificaça o estejam corretos.  |                                                 |
| Visão geral da estrutura | Dimensa o (shape), Amostra (head, tail), Tipos & nulos (info).    | df.shape, df.head(), df.info(), df.isna().sum() | 
| Tratamento de valores ausentes |  - Quantificar % de nulos por coluna.                                                      |                  |
|                                 |   - Definir estrate gia: descartar, imputar (me dia/mediana/moda) ou criar flag de nulo.  |                  |
|                                 |   - Documentar sua decisão                                                                | isna, fillna, dropna, SimpleImputer |
|Conversa o e limpeza de tipos    |Converter datas para datetime.                                                             | pd.to_datetime, astype, str.strip, replace |
|                                 | Garantir que atrasos estejam em formato numerico.                                        |                                            |
|                                 | Corrigir categorias inconsistentes (ex.: “AA ” vs “AA”).                                  |                                            |
| Feature engineering inicial     | Criar coluna bina ria “Segurança alta”.                                                    |                                            |
|                                 | Extrair nível ordinal da aceitaçãoo do carro (class).                                    |
|                                 | Calcular escore de capacidade total = persons + lug_boot. |                       |        |
| Filtragem e segmentação | Exemplos: voos de um aeroporto especí fico, somente atrasos > 0, temporadas alta/baixa. | query, boolean indexing|
|                          | Usar filtros para ana lises específicas. | |
| Estatísticas descritivas e outliers | describe() em nume ricas. Identificar quartis e usar regra IQR (1,5×IQR) para flag de outlier. |   |
|                                     | Calcular skewness e kurtosis se necessário. |     | describe, quantile, clip, skew, kurtosis | 
| Visualizaço es exploratórias | Distribuiça o das avaliações dos carros: countplot e boxplot da varia vel class_nivel. | seaborn (histplot, boxplot, heatmap, barplot), plt.savefig |
|                              | Heatmap de correlaço es entre varia veis nume ricas (ex.: doors, persons, maint_nivel). Barplot das combinaço es buying × safety com maior proporça o de carros “vgood”. Salvar figuras (.png) para inserir no Word. |  |
|                              |                                                                                 |                                                    |
| Documentaça o de insights    | Em celulas Markdown, registrar anomalias, colunas irrelevantes, hipóteses (ex.: clima, horário de pico). Lista de colunas candidatas a remoça o ou transformaça o posterior. | Markdown |
| Salvar dataset limpo         | Exportar df_clean para CSV ou Parquet para reutilizar no Step 3. | to_csv, to_parquet |
|                              | Versa o com e sem outliers, se aplicável. |           |

# DICAS

Para facilitar sua pesquisa e aprendizado durante a pra tica 1, fique de olho em algumas dicas:

1. Busque informaço es sobre os seguintes to picos nos capí tulos anteriores, em tutoriais, ví deos ou documentaça o.
2. Variáveis em Python: Revisite como float, int e object sa o exibidos em pandas.Estruturas condicionais (if, elif, else): Busque exemplos que mostrem como tomar deciso es no co digo, executando comandos diferentes conforme as respostas do usua rio.
3. df.describe() & df.value_counts(): Geram estatí sticas nume ricas e freque ncias catego ricas.
4. Visualizações básicas: sns.histplot, sns.boxplot, sns.heatmap
5. Tratamento de nulos: df.isna().sum(), fillna ou dropna com critério justificado.
6. Funções reutiliza veis: Crie def carregar_dados() ou def plot_distrib(col)
