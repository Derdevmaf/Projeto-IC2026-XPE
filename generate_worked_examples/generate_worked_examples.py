#!/usr/bin/env python3
"""
generate_worked_examples.py

Gera automaticamente uma base de "worked examples" (exemplos resolvidos)
para apoiar a construcao de uma sequencia didatica baseada em conceitos,
utilizando um modelo de IA (Claude, via API da Anthropic).

Fluxo geral:
    1) A IA gera K conceitos fundamentais associados a um tema (--C).
    2) Para cada conceito, sao gerados N worked examples (--N), cada um
       exercitando um verbo sorteado aleatoriamente da Taxonomia de
       Bloom Revisada (arquivo bloom_verbs.txt).
    3) Cada worked example e salvo em um arquivo de texto individual,
       dentro da pasta worked_examples/.

Uso:
    python generate_worked_examples.py --C <numero_conceitos> --N <numero_exemplos>

Exemplos:
    python generate_worked_examples.py --C 5 --N 3
    python generate_worked_examples.py --C 3 --N 2 --dry-run
    python generate_worked_examples.py --C 4 --N 2 --tema "Estruturas de Dados" --model claude-haiku-4-5-20251001

Requisitos:
    - Python 3.9+
    - Variavel de ambiente ANTHROPIC_API_KEY definida (ou um arquivo .env
      com essa variavel), exceto ao usar --dry-run.
    - Dependencias listadas em requirements.txt (pip install -r requirements.txt)
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import time
import unicodedata
from dataclasses import dataclass
from pathlib import Path

# python-dotenv e opcional: se instalado, carrega variaveis de um arquivo
# .env automaticamente. Se nao estiver instalado, seguimos usando apenas
# variaveis de ambiente definidas normalmente no sistema/terminal.
try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

try:
    from google import genai
    from google.genai import errors
except ImportError:
    print(
        "ERRO: a biblioteca 'google-genai' nao esta instalada.\n"
        "Instale as dependencias com: pip install -r requirements.txt",
        file=sys.stderr,
    )
    sys.exit(1)


# ---------------------------------------------------------------------------
# Configuracao e constantes
# ---------------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent
BLOOM_VERBS_PATH = SCRIPT_DIR / "bloom_verbs.txt"
OUTPUT_DIR = SCRIPT_DIR / "worked_examples"

TEMA_PADRAO = "Programação em Python"
MODELO_PADRAO = "gemini-2.5-flash"

MAX_TENTATIVAS_IA = 3          # numero de tentativas por chamada a IA
ESPERA_BASE_SEGUNDOS = 2       # base do backoff exponencial (2s, 4s, 8s, ...)
MAX_TOKENS_CONCEITOS = 1024
MAX_TOKENS_WORKED_EXAMPLE = 1500
AVISO_MUITAS_CHAMADAS = 60     # a partir de quantas chamadas exibir um aviso


@dataclass
class VerboBloom:
    """Representa um verbo (ou expressao) da Taxonomia de Bloom Revisada."""

    verbo: str
    categoria: str | None = None  # ex.: "Lembrar", "Analisar", "Criar"...


# ---------------------------------------------------------------------------
# Etapa 0 - leitura dos parametros da linha de comando
# ---------------------------------------------------------------------------


def ler_argumentos(argv: list[str] | None = None) -> argparse.Namespace:
    """Le e valida os parametros da linha de comando."""
    parser = argparse.ArgumentParser(
        prog="generate_worked_examples.py",
        description=(
            "Gera worked examples com IA para apoiar a construcao de uma "
            "sequencia didatica baseada em conceitos."
        ),
        epilog="Exemplo: python generate_worked_examples.py --C 5 --N 3",
    )
    parser.add_argument(
        "--C",
        type=int,
        required=True,
        help="Quantidade de conceitos (K) a serem gerados pela IA.",
    )
    parser.add_argument(
        "--N",
        type=int,
        required=True,
        help="Quantidade de worked examples a gerar para cada conceito.",
    )
    parser.add_argument(
        "--tema",
        type=str,
        default=TEMA_PADRAO,
        help=f"Tema/dominio de conteudo a ser utilizado (padrao: '{TEMA_PADRAO}').",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=MODELO_PADRAO,
        help=f"Modelo Gemini a ser utilizado nas chamadas de IA (padrao: '{MODELO_PADRAO}').",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Semente aleatoria opcional, para tornar reprodutivel o sorteio dos verbos.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Executa todo o fluxo (conceitos, verbos, arquivos) sem chamar a API de "
            "IA de verdade. Util para testar a estrutura de pastas/arquivos e a "
            "instalacao sem gastar creditos e sem precisar de GEMINI_API_KEY."
        ),
    )

    args = parser.parse_args(argv)

    if args.C <= 0:
        parser.error("--C deve ser um numero inteiro positivo.")
    if args.N <= 0:
        parser.error("--N deve ser um numero inteiro positivo.")

    return args


# ---------------------------------------------------------------------------
# Etapa 3 - leitura de bloom_verbs.txt e selecao aleatoria do verbo
# ---------------------------------------------------------------------------


def carregar_verbos_bloom(filepath: Path) -> list[VerboBloom]:
    """
    Le o arquivo bloom_verbs.txt e retorna a lista de verbos disponiveis.

    Regras de leitura:
      - linhas em branco sao ignoradas;
      - linhas iniciadas com '#' sao tratadas como comentarios/cabecalhos de
        categoria (ex.: "# Analisar") e nao sao usadas como verbos, mas
        atualizam a categoria associada aos verbos seguintes;
      - verbos duplicados (o mesmo verbo pode aparecer em mais de uma
        categoria no documento de origem) sao mantidos uma unica vez, para
        que o sorteio aleatorio nao fique enviesado.

    Levanta:
        FileNotFoundError: se o arquivo nao existir.
        ValueError: se o arquivo existir mas nao contiver nenhum verbo valido.
    """
    if not filepath.exists():
        raise FileNotFoundError(
            f"Arquivo de verbos nao encontrado: '{filepath}'.\n"
            "Crie um arquivo 'bloom_verbs.txt' (um verbo da Taxonomia de "
            "Bloom por linha) na mesma pasta do script."
        )

    categoria_atual: str | None = None
    ja_vistos: set[str] = set()
    verbos: list[VerboBloom] = []

    with filepath.open("r", encoding="utf-8") as f:
        for linha_bruta in f:
            linha = linha_bruta.strip()

            if not linha:
                continue

            if linha.startswith("#"):
                cabecalho = linha.lstrip("#").strip()
                if cabecalho:
                    categoria_atual = cabecalho
                continue

            chave = linha.rstrip(".").lower()
            if chave in ja_vistos:
                continue

            ja_vistos.add(chave)
            verbos.append(VerboBloom(verbo=linha, categoria=categoria_atual))

    if not verbos:
        raise ValueError(
            f"O arquivo '{filepath}' esta vazio ou nao contem nenhum verbo valido."
        )

    return verbos


def selecionar_verbo_aleatorio(verbos: list[VerboBloom]) -> VerboBloom:
    """Seleciona aleatoriamente um verbo da lista carregada de bloom_verbs.txt."""
    return random.choice(verbos)


# ---------------------------------------------------------------------------
# Utilitario - conversao de conceito para snake_case (nome de arquivo)
# ---------------------------------------------------------------------------


def gerar_slug(texto: str) -> str:
    """
    Converte um texto (ex.: nome de conceito) para snake_case, adequado
    para uso em nomes de arquivo:
      - minusculas;
      - espacos substituidos por '_';
      - remocao de caracteres especiais (mantendo apenas letras, numeros
        e '_'). Acentos sao normalizados para a forma sem acento antes da
        remocao, para preservar a legibilidade
        (ex.: "Funcao Lambda" -> "funcao_lambda").
    """
    texto = texto.strip().lower()
    texto = texto.replace(" ", "_")

    # normaliza acentuacao (NFKD) e remove as marcas de acento
    texto = unicodedata.normalize("NFKD", texto)
    texto = "".join(ch for ch in texto if not unicodedata.combining(ch))

    # remove qualquer caractere que nao seja [a-z0-9_]
    texto = re.sub(r"[^a-z0-9_]", "", texto)

    # colapsa multiplos '_' seguidos e remove '_' nas bordas
    texto = re.sub(r"_+", "_", texto).strip("_")

    return texto or "conceito"


# ---------------------------------------------------------------------------
# Comunicacao com a IA (Anthropic / Claude API)
# ---------------------------------------------------------------------------


def obter_cliente_ia() -> genai.Client:
    """Cria o cliente da API Gemini, validando a presenca da chave de API."""
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print(
            "ERRO: a variavel de ambiente GEMINI_API_KEY nao foi encontrada.\n"
            "Defina sua chave de API (copie '.env.example' para '.env' e "
            "preencha com sua chave, ou exporte a variavel no terminal) antes "
            "de executar o programa. Alternativamente, use --dry-run para "
            "testar o programa sem chamar a IA de verdade.",
            file=sys.stderr,
        )
        sys.exit(1)

    return genai.Client(api_key=api_key)


def chamar_ia(
    client: genai.Client,
    model: str,
    user_prompt: str,
    system_prompt: str | None = None,
    max_tokens: int = MAX_TOKENS_WORKED_EXAMPLE,
    max_tentativas: int = MAX_TENTATIVAS_IA,
) -> str:
    """
    Realiza uma chamada a API do Gemini, com tratamento de falhas de
    comunicacao (erros de rede, limite de requisicoes, erros do servidor)
    por meio de novas tentativas com espera progressiva (backoff exponencial).

    Retorna o texto da resposta.

    Levanta:
        RuntimeError: se a chamada falhar de forma definitiva (erro do lado
        do cliente, ex.: chave invalida) ou apos esgotar as tentativas.
    """
    ultimo_erro: Exception | None = None

    for tentativa in range(1, max_tentativas + 1):
        try:
            config_kwargs: dict = {
                "max_output_tokens": max_tokens,
            }
            if system_prompt:
                config_kwargs["system_instruction"] = system_prompt

            resposta = client.models.generate_content(
                model=model,
                contents=user_prompt,
                config=genai.types.GenerateContentConfig(**config_kwargs)
            )
            return resposta.text.strip() if resposta.text else ""

        except errors.APIError as e:
            ultimo_erro = e
            if e.code == 429:
                _aguardar_nova_tentativa(tentativa, max_tentativas, "limite de requisicoes atingido")
            elif e.code is not None and 500 <= e.code < 600:
                _aguardar_nova_tentativa(
                    tentativa, max_tentativas, f"erro no servidor da IA (HTTP {e.code})"
                )
            else:
                # Erro do lado do cliente (ex.: chave invalida, requisicao
                # malformada) - novas tentativas nao vao ajudar.
                raise RuntimeError(
                    f"Erro da API ao chamar o modelo de IA (HTTP {e.code}): {e.message}"
                ) from e
        except Exception as e:
            # Qualquer outro erro de conexao ou inesperado.
            ultimo_erro = e
            _aguardar_nova_tentativa(tentativa, max_tentativas, f"erro inesperado: {e}")

    raise RuntimeError(
        f"Falha ao comunicar com a IA apos {max_tentativas} tentativa(s). "
        f"Ultimo erro: {ultimo_erro}"
    )


def _aguardar_nova_tentativa(tentativa: int, max_tentativas: int, motivo: str) -> None:
    """Registra um aviso e aguarda (backoff exponencial) antes da proxima tentativa."""
    if tentativa >= max_tentativas:
        return
    espera = ESPERA_BASE_SEGUNDOS ** tentativa
    print(
        f"    [aviso] {motivo} (tentativa {tentativa}/{max_tentativas}). "
        f"Nova tentativa em {espera}s..."
    )
    time.sleep(espera)


# ---------------------------------------------------------------------------
# Validacao de JSON
# ---------------------------------------------------------------------------


def extrair_json(texto: str) -> str:
    """Remove cercas de markdown (```json ... ```), se presentes, de uma resposta."""
    texto = texto.strip()
    if texto.startswith("```"):
        texto = re.sub(r"^```[a-zA-Z]*\s*\n?", "", texto)
        texto = re.sub(r"\n?```\s*$", "", texto)
        texto = texto.strip()
    return texto


def validar_json(texto: str) -> object | None:
    """
    Tenta fazer o parse de uma string como JSON (apos remover eventuais
    cercas de markdown). Retorna o objeto Python correspondente em caso de
    sucesso, ou None caso o texto nao seja um JSON valido.
    """
    candidato = extrair_json(texto)
    try:
        return json.loads(candidato)
    except json.JSONDecodeError:
        return None


# ---------------------------------------------------------------------------
# Etapa 1 - geracao dos conceitos
# ---------------------------------------------------------------------------


def gerar_conceitos(
    client: genai.Client | None,
    model: str,
    tema: str,
    k: int,
    dry_run: bool = False,
) -> list[str]:
    """
    Realiza uma unica chamada a IA solicitando K conceitos fundamentais
    associados ao tema informado. A resposta e validada como JSON (lista
    de strings); em caso de falha de validacao, uma nova tentativa e feita
    com uma instrucao de correcao, ate MAX_TENTATIVAS_IA vezes.
    """
    if dry_run:
        return [f"Conceito de exemplo {i}" for i in range(1, k + 1)]

    assert client is not None  # garantido pelo fluxo de main() quando nao e dry_run

    system_prompt = (
        "Voce e um especialista em design instrucional. Responda SEMPRE e "
        "SOMENTE com um JSON valido, sem nenhum texto adicional antes ou "
        "depois, e sem cercas de markdown (```)."
    )
    user_prompt = (
        f'Gere {k} conceitos fundamentais associados ao conteudo "{tema}".\n\n'
        f"Responda EXCLUSIVAMENTE em formato JSON, como uma lista de "
        f"exatamente {k} strings, no seguinte formato:\n"
        f'["Conceito 1", "Conceito 2", ...]\n\n'
        "Nao inclua nenhum texto antes ou depois do JSON."
    )

    for tentativa in range(1, MAX_TENTATIVAS_IA + 1):
        texto = chamar_ia(
            client, model, user_prompt, system_prompt=system_prompt, max_tokens=MAX_TOKENS_CONCEITOS
        )
        dados = validar_json(texto)

        if (
            isinstance(dados, list)
            and len(dados) > 0
            and all(isinstance(item, str) and item.strip() for item in dados)
        ):
            conceitos = [c.strip() for c in dados]
            if len(conceitos) != k:
                print(
                    f"  [aviso] a IA retornou {len(conceitos)} conceito(s) em vez de {k}. "
                    "Prosseguindo com os conceitos recebidos."
                )
            return conceitos

        print(
            f"  [aviso] a resposta da IA nao e um JSON valido de conceitos "
            f"(tentativa {tentativa}/{MAX_TENTATIVAS_IA})."
        )
        user_prompt = (
            "A resposta anterior nao estava em um formato JSON valido. Gere "
            f'novamente {k} conceitos fundamentais associados ao conteudo "{tema}", '
            "respondendo EXCLUSIVAMENTE com uma lista JSON de strings, por exemplo: "
            '["Conceito 1", "Conceito 2", ...]. Nao inclua nenhum texto adicional.'
        )

    raise RuntimeError(
        f"Nao foi possivel obter uma lista de conceitos valida da IA apos "
        f"{MAX_TENTATIVAS_IA} tentativas."
    )


# ---------------------------------------------------------------------------
# Etapa 4 - geracao dos worked examples
# ---------------------------------------------------------------------------


def gerar_worked_example(
    client: genai.Client | None,
    model: str,
    tema: str,
    conceito: str,
    verbo: VerboBloom,
    contexto_bloom: str,
    dry_run: bool = False,
) -> str:
    """
    Gera um worked example (exemplo resolvido, passo a passo) que exercita
    um conceito especifico usando um verbo sorteado da Taxonomia de Bloom.
    O conteudo de bloom_verbs.txt e anexado ao contexto da chamada, para
    que a IA utilize o verbo de forma pedagogicamente correta.
    """
    categoria_txt = f' (categoria "{verbo.categoria}")' if verbo.categoria else ""

    if dry_run:
        return (
            "[MODO --dry-run: conteudo simulado, nenhuma chamada real a IA foi feita]\n\n"
            "Este arquivo representa o worked example que seria gerado pela IA para:\n"
            f"  - Tema: {tema}\n"
            f"  - Conceito: {conceito}\n"
            f"  - Verbo da Taxonomia de Bloom: {verbo.verbo}{categoria_txt}\n\n"
            "Execute novamente sem a flag --dry-run (com GEMINI_API_KEY "
            "configurada) para gerar o worked example real."
        )

    assert client is not None

    system_prompt = (
        "Voce e um especialista em design instrucional e na Taxonomia de "
        "Bloom Revisada, elaborando material didatico. Abaixo esta o arquivo "
        "de referencia com os verbos da Taxonomia de Bloom, organizados por "
        "categoria cognitiva - use-o como contexto para compreender o nivel "
        "cognitivo correto do verbo solicitado em cada exercicio:\n\n"
        f"{contexto_bloom}"
    )
    user_prompt = (
        f'Gere um worked example (exemplo resolvido, explicado passo a passo) que '
        f'exercite o conceito "{conceito}", dentro do tema "{tema}", utilizando o '
        f'verbo "{verbo.verbo}"{categoria_txt} da Taxonomia de Bloom.\n\n'
        "O worked example deve conter:\n"
        "1. Um enunciado claro do problema/tarefa a ser resolvida;\n"
        "2. A resolucao comentada, passo a passo, demonstrando o raciocinio;\n"
        "3. Uma breve conclusao ou observacao final.\n\n"
        f'Sinta-se livre para mencionar naturalmente outros conceitos relacionados a '
        f'"{conceito}" sempre que isso ajudar a explicar o raciocinio - essas relacoes '
        "serao usadas posteriormente para mapear dependencias entre conceitos.\n\n"
        "Responda apenas com o conteudo do worked example, em texto corrido, sem "
        "comentarios meta sobre a tarefa em si."
    )

    return chamar_ia(
        client, model, user_prompt, system_prompt=system_prompt, max_tokens=MAX_TOKENS_WORKED_EXAMPLE
    )


# ---------------------------------------------------------------------------
# Criacao de diretorios e escrita de arquivos
# ---------------------------------------------------------------------------


def criar_diretorio_saida(diretorio: Path) -> None:
    """Cria o diretorio de saida (e pais necessarios), caso ainda nao exista."""
    diretorio.mkdir(parents=True, exist_ok=True)


def _caminho_para_exibicao(caminho: Path) -> str:
    """Formata um caminho para exibicao no console (relativo a SCRIPT_DIR quando possivel)."""
    try:
        return str(caminho.relative_to(SCRIPT_DIR))
    except ValueError:
        return str(caminho)


def salvar_worked_example(diretorio: Path, conceito_slug: str, indice: int, conteudo: str) -> Path:
    """
    Salva um worked example em um arquivo de texto, seguindo o padrao:
        worked_examples/worked_example_<conceito>_<indice>.txt
    """
    nome_arquivo = f"worked_example_{conceito_slug}_{indice}.txt"
    caminho = diretorio / nome_arquivo
    caminho.write_text(conteudo.strip() + "\n", encoding="utf-8")
    return caminho


# ---------------------------------------------------------------------------
# Orquestracao principal
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    args = ler_argumentos(argv)

    if args.seed is not None:
        random.seed(args.seed)

    print("=" * 60)
    print("generate_worked_examples.py")
    print("=" * 60)
    print(f"Tema.......................: {args.tema}")
    print(f"Conceitos a gerar (--C)....: {args.C}")
    print(f"Worked examples/conceito (--N): {args.N}")
    print(f"Modelo.....................: {args.model}")
    if args.dry_run:
        print("Modo.......................: DRY-RUN (sem chamadas reais a IA)")
    print("-" * 60)

    estimativa_chamadas = args.C * args.N + 1
    if not args.dry_run and estimativa_chamadas > AVISO_MUITAS_CHAMADAS:
        print(
            f"[aviso] esta execucao fara aproximadamente {estimativa_chamadas} "
            "chamadas a API, o que pode levar tempo e ter custo associado.\n"
        )

    # -- verbos da Taxonomia de Bloom -------------------------------------
    try:
        verbos = carregar_verbos_bloom(BLOOM_VERBS_PATH)
    except (FileNotFoundError, ValueError) as e:
        print(f"ERRO: {e}", file=sys.stderr)
        return 1

    print(f"Carregados {len(verbos)} verbos de '{BLOOM_VERBS_PATH.name}'.")
    contexto_bloom = BLOOM_VERBS_PATH.read_text(encoding="utf-8")

    # -- cliente de IA ------------------------------------------------------
    client = None if args.dry_run else obter_cliente_ia()

    # -- Etapa 1: geracao dos conceitos -------------------------------------
    print("\nEtapa 1: gerando conceitos com a IA...")
    try:
        conceitos = gerar_conceitos(client, args.model, args.tema, args.C, dry_run=args.dry_run)
    except RuntimeError as e:
        print(f"ERRO: {e}", file=sys.stderr)
        return 1

    print(f"Conceitos gerados ({len(conceitos)}):")
    for c in conceitos:
        print(f"  - {c}")

    # -- Etapas 2-4: para cada conceito, gerar N worked examples ------------
    criar_diretorio_saida(OUTPUT_DIR)

    total_gerado = 0
    total_falhas = 0

    print(f"\nEtapas 2-4: gerando {args.N} worked example(s) por conceito...")
    for conceito in conceitos:
        slug = gerar_slug(conceito)
        print(f"\nConceito: {conceito}  (slug: {slug})")

        for indice in range(1, args.N + 1):
            verbo = selecionar_verbo_aleatorio(verbos)
            print(f"  [{indice}/{args.N}] verbo sorteado: '{verbo.verbo}'...", end=" ", flush=True)

            try:
                conteudo = gerar_worked_example(
                    client, args.model, args.tema, conceito, verbo, contexto_bloom, dry_run=args.dry_run
                )
                caminho = salvar_worked_example(OUTPUT_DIR, slug, indice, conteudo)
                print(f"OK -> {_caminho_para_exibicao(caminho)}")
                total_gerado += 1
            except RuntimeError as e:
                print(f"FALHOU ({e})")
                total_falhas += 1

    # -- resumo final ---------------------------------------------------------
    print("\n" + "=" * 60)
    print(f"Concluido. {total_gerado} worked example(s) gerado(s) com sucesso.")
    if total_falhas:
        print(f"{total_falhas} chamada(s) falharam e foram ignoradas (veja os avisos acima).")
    print(f"Arquivos salvos em: {OUTPUT_DIR}")
    if not args.dry_run and total_gerado > 0:
        print(
            "\nProximo passo sugerido: use os worked examples gerados para montar "
            "uma matriz de coocorrencia entre conceitos e inferir uma sequencia "
            "pedagogica baseada nas relacoes observadas."
        )

    return 0 if total_falhas == 0 or total_gerado > 0 else 1


if __name__ == "__main__":
    sys.exit(main())
