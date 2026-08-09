# generate_worked_examples.py

Programa em Python que gera automaticamente uma base de **worked examples**
(exemplos resolvidos) para apoiar a construção de uma sequência didática
baseada em conceitos, usando a API do Google Gemini e verbos da
**Taxonomia de Bloom Revisada**.

## Como funciona

1. **Etapa 1** — a IA gera `K` conceitos fundamentais associados a um tema
   (por padrão, "Programação em Python"). Chamada feita uma única vez.
2. **Etapa 2** — o programa percorre cada um dos `K` conceitos.
3. **Etapa 3** — para cada worked example, um verbo da Taxonomia de Bloom é
   sorteado aleatoriamente a partir de `bloom_verbs.txt`.
4. **Etapa 4** — para cada conceito, são feitas `N` chamadas independentes à
   IA, cada uma gerando um worked example que exercita o conceito usando o
   verbo sorteado.

A ideia pedagógica: ao gerar vários worked examples sobre um conceito, é
esperado que a IA mencione naturalmente outros conceitos relacionados. Essas
menções podem ser usadas depois para montar uma matriz de coocorrência entre
conceitos e inferir uma sequência pedagógica a partir das relações
observadas — em vez de definir essa ordem manualmente.

## Requisitos

- Python 3.9 ou superior
- Uma chave de API do Google Gemini ([aistudio.google.com](https://aistudio.google.com/app/apikey))

## Instalação (VS Code)

1. Extraia este projeto e abra a pasta no VS Code.
2. Abra um terminal (`Terminal > New Terminal`) e crie um ambiente virtual (recomendado):

   ```bash
   python -m venv venv
   # Windows:
   venv\Scripts\activate
   # macOS/Linux:
   source venv/bin/activate
   ```

3. Instale as dependências:

   ```bash
   pip install -r requirements.txt
   ```

4. Configure sua chave de API: copie `.env.example` para `.env` e substitua
   pelo valor da sua chave:

   ```bash
   cp .env.example .env   # Windows: copy .env.example .env
   ```

   Edite o arquivo `.env` gerado e coloque sua chave em `GEMINI_API_KEY`.
   O arquivo `.env` já está no `.gitignore` e nunca deve ser compartilhado.

## Uso

```bash
python generate_worked_examples.py --C <numero_conceitos> --N <numero_exemplos>
```

Exemplo (gera 5 conceitos, com 3 worked examples cada = 15 arquivos):

```bash
python generate_worked_examples.py --C 5 --N 3
```

### Testar sem gastar créditos (`--dry-run`)

Antes de rodar de verdade, você pode validar toda a estrutura de
pastas/arquivos sem chamar a IA e **sem precisar de `GEMINI_API_KEY`**:

```bash
python generate_worked_examples.py --C 3 --N 2 --dry-run
```

Isso cria os mesmos arquivos e pastas, mas com um conteúdo de exemplo no
lugar do texto gerado pela IA — útil para conferir a instalação antes de
uma execução real.

### Parâmetros

| Parâmetro   | Obrigatório | Padrão                    | Descrição                                                        |
|-------------|:-----------:|----------------------------|-------------------------------------------------------------------|
| `--C`       | sim         | —                           | Quantidade de conceitos (K) a gerar                                |
| `--N`       | sim         | —                           | Quantidade de worked examples por conceito                        |
| `--tema`    | não         | `Programação em Python`    | Tema/domínio de conteúdo                                          |
| `--model`   | não         | `gemini-2.5-flash`           | Modelo Gemini usado nas chamadas                                  |
| `--seed`    | não         | aleatório                  | Semente para tornar reprodutível o sorteio de verbos               |
| `--dry-run` | não         | desativado                 | Roda sem chamar a IA de verdade (ver seção acima)                 |

## Estrutura de arquivos gerados

```text
worked_examples/
├── conceitos.json                   # Lista dos K conceitos gerados (JSON)
├── conceitos.txt                    # Lista dos K conceitos gerados (1 por linha)
├── worked_examples.csv              # Tabela com id do exemplo, verbo bloom e conceito usado
├── worked_example_<conceito>_1.txt   # Conteúdo do worked example gerado
├── worked_example_<conceito>_2.txt
└── ...
```

`<conceito>` é o nome do conceito convertido para `snake_case` (minúsculas,
acentos normalizados, espaços viram `_`, caracteres especiais são
removidos). A pasta `worked_examples/` é criada automaticamente se não
existir.

A tabela `worked_examples.csv` possui os seguintes cabeçalhos:
- `id`: Identificador único do exemplo (ex.: `worked_example_busca_vetorial_1`).
- `verbo_bloom`: Verbo da Taxonomia de Bloom sorteado para o exemplo.
- `conceito`: Nome do conceito trabalhado.

## Sobre o `bloom_verbs.txt`

Este arquivo foi gerado a partir do **anexo em PDF fornecido** (*Anexo II –
Exemplos de verbos da Taxonomia de Bloom Revisada*), não de uma lista
genérica. Os verbos foram extraídos e organizados pelas 6 categorias
cognitivas originais do documento (Lembrar, Entender, Aplicar, Analisar,
Avaliar, Criar), separadas por linhas de comentário (`# Categoria`).

- Linhas em branco e linhas iniciadas com `#` são ignoradas no sorteio —
  servem apenas como cabeçalho/organização.
- O conteúdo completo do arquivo é anexado ao contexto de cada chamada de
  geração de worked example, para que a IA compreenda o nível cognitivo do
  verbo sorteado (conforme pedido na especificação original).
- Você pode editar `bloom_verbs.txt` livremente (adicionar, remover ou
  reorganizar verbos); o programa relê o arquivo a cada execução.

## Tratamento de erros

O programa foi feito para lidar com os cenários abaixo sem travar de forma
inesperada:

- **`bloom_verbs.txt` ausente ou vazio** → mensagem de erro clara e
  encerramento (código de saída 1) antes de qualquer chamada à IA.
- **`GEMINI_API_KEY` não configurada** → mensagem de erro explicando como
  configurar (ou sugerindo `--dry-run`).
- **JSON inválido na geração dos conceitos** → o programa detecta, avisa e
  tenta novamente (até 3 tentativas) com uma instrução de correção.
- **Falhas de comunicação com a API** (erro de conexão, limite de
  requisições, erro 5xx do servidor) → novas tentativas automáticas com
  espera progressiva (backoff exponencial). Erros do lado do cliente (ex.:
  chave inválida) não são repetidos.
- **Falha definitiva em um worked example específico** → o programa avisa,
  pula esse arquivo e continua com os demais, em vez de interromper a
  execução inteira.

## Testes automatizados

O projeto inclui uma suíte de testes que valida toda a lógica do programa
**sem precisar de uma chave de API real** (as chamadas à IA são simuladas
com `unittest.mock`, e o fluxo completo também é testado via `--dry-run`).
Isso cobre: geração de slugs, leitura e deduplicação de `bloom_verbs.txt`,
validação de JSON, nomes/conteúdo dos arquivos salvos, novas tentativas em
falhas de rede/rate limit, e o fluxo de ponta a ponta em `main()`.

Para rodar os testes:

```bash
python -m unittest discover -s tests -v
```

## Estrutura do projeto

```text
.
├── generate_worked_examples.py   # programa principal
├── bloom_verbs.txt                # verbos da Taxonomia de Bloom (extraídos do PDF fornecido)
├── requirements.txt                # dependências (google-genai, python-dotenv)
├── .env.example                    # modelo para configurar sua chave de API
├── .gitignore
├── README.md
└── tests/
    └── test_generate_worked_examples.py
```

## Próximos passos sugeridos (fora do escopo deste programa)

Com os worked examples gerados, o passo seguinte é ler os arquivos em
`worked_examples/`, identificar menções a outros conceitos dentro de cada
texto e montar uma matriz de coocorrência entre conceitos — que pode então
ser usada para propor uma sequência pedagógica baseada nas relações
observadas, em vez de uma ordem definida manualmente.
