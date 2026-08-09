#!/usr/bin/env python3
"""
generate_worked_examples.py

Gera automaticamente uma base de "worked examples" (exemplos resolvidos)
para apoiar a construcao de uma sequencia didatica baseada em conceitos,
utilizando um modelo de IA (Groq, OpenRouter ou Gemini).

Fluxo geral:
    1) A IA gera K conceitos fundamentais associados a um tema (--C).
    2) Para cada conceito, sao gerados N worked examples (--N) em UMA UNICA
       chamada de API por conceito (JSON em lote), cada um exercitando um
       verbo sorteado da Taxonomia de Bloom Revisada (bloom_verbs.txt).
    3) Cada worked example e salvo em um arquivo de texto individual,
       dentro da pasta worked_examples/.

Uso:
    python generate_worked_examples.py --C <num_conceitos> --N <num_exemplos>

Exemplos:
    python generate_worked_examples.py --C 15 --N 3
    python generate_worked_examples.py --C 15 --N 3 --provider groq
    python generate_worked_examples.py --C 5 --N 3 --provider openrouter
    python generate_worked_examples.py --C 3 --N 2 --dry-run
    python generate_worked_examples.py --C 4 --N 2 --tema "Estruturas de Dados"

Provedores suportados (--provider):
    groq       — Groq (recomendado; ~14400 req/dia gratis). Requer GROQ_API_KEY.
    openrouter — OpenRouter (~200 req/dia por modelo gratis). Requer OPENROUTER_API_KEY.
    gemini     — Google Gemini (limite de 20 req/dia no plano gratis). Requer GEMINI_API_KEY.

Requisitos:
    - Python 3.9+
    - Chave de API do provedor escolhido no arquivo .env (copie .env.example).
    - Dependencias listadas em requirements.txt (pip install -r requirements.txt)
"""

from __future__ import annotations

import argparse
import csv
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

# openai SDK — usado pelo Groq e pelo OpenRouter (interface OpenAI-compat)
try:
    import openai as openai_lib
except ImportError:
    openai_lib = None  # type: ignore

# google-genai SDK — usado apenas quando --provider gemini
try:
    from google import genai
    from google.genai import errors as genai_errors
except ImportError:
    genai = None  # type: ignore
    genai_errors = None  # type: ignore


# ---------------------------------------------------------------------------
# Configuracao e constantes
# ---------------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent
BLOOM_VERBS_PATH = SCRIPT_DIR / "bloom_verbs.txt"
OUTPUT_DIR = SCRIPT_DIR / "worked_examples"

TEMA_PADRAO = "Programação em Python"
PROVEDOR_PADRAO = "groq"

# Modelos padrao por provedor
MODELOS_PADRAO: dict[str, str] = {
    "groq":       "llama-3.3-70b-versatile",
    "openrouter": "openrouter/auto",
    "gemini":     "gemini-2.5-flash-lite",
}

PROVEDORES_VALIDOS = tuple(MODELOS_PADRAO.keys())

# URLs base dos provedores OpenAI-compat
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
GROQ_BASE_URL       = "https://api.groq.com/openai/v1"

MAX_TENTATIVAS_IA = 5          # tentativas por chamada a IA
ESPERA_BASE_SEGUNDOS = 2       # base do backoff exponencial (2s, 4s, 8s, ...)
MAX_TOKENS_CONCEITOS = 1024
MAX_TOKENS_WORKED_EXAMPLE = 1500
AVISO_MUITAS_CHAMADAS = 60     # a partir de quantas chamadas exibir um aviso


class QuotaDiariaExcedidaError(Exception):
    """Excecao lancada quando a cota diaria de requisicoes e atingida."""

    pass


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
        epilog=(
            "Exemplos:\n"
            "  python generate_worked_examples.py --C 15 --N 3\n"
            "  python generate_worked_examples.py --C 15 --N 3 --provider groq\n"
            "  python generate_worked_examples.py --C 5 --N 3 --provider openrouter\n"
            "  python generate_worked_examples.py --C 3 --N 2 --dry-run"
        ),
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
        "--provider",
        type=str,
        default=PROVEDOR_PADRAO,
        choices=list(PROVEDORES_VALIDOS),
        help=(
            f"Provedor de IA a usar (padrao: '{PROVEDOR_PADRAO}'). "
            "groq=Groq (~14400 req/dia gratis); "
            "openrouter=OpenRouter (~200 req/dia gratis); "
            "gemini=Google Gemini (20 req/dia no plano gratis)."
        ),
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help=(
            "Modelo a ser utilizado. Se omitido, usa o modelo padrao do provedor: "
            + ", ".join(f"{p}='{m}'" for p, m in MODELOS_PADRAO.items()) + "."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Semente aleatoria opcional, para tornar reprodutivel o sorteio dos verbos.",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0.0,
        help=(
            "Tempo de espera fixo em segundos entre chamadas sucessivas a API para evitar "
            "exceder a cota de requisicoes por minuto."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Executa todo o fluxo (conceitos, verbos, arquivos) sem chamar a API de "
            "IA de verdade. Util para testar a estrutura de pastas/arquivos sem "
            "precisar de chave de API."
        ),
    )

    args = parser.parse_args(argv)

    if args.C <= 0:
        parser.error("--C deve ser um numero inteiro positivo.")
    if args.N <= 0:
        parser.error("--N deve ser um numero inteiro positivo.")

    # Aplica modelo padrao do provedor se --model nao foi fornecido
    if args.model is None:
        args.model = MODELOS_PADRAO[args.provider]

    return args


# ---------------------------------------------------------------------------
# Etapa 3 - leitura de bloom_verbs.txt e selecao aleatoria do verbo
# ---------------------------------------------------------------------------


def carregar_verbos_bloom(filepath: Path) -> list[VerboBloom]:
    """
    Le o arquivo bloom_verbs.txt e retorna a lista de verbos disponiveis.
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


def obter_cliente_ia(provider: str) -> object:
    """
    Cria e retorna o cliente de IA adequado para o provedor informado.

    Retorna:
      - Para 'groq' e 'openrouter': instancia de openai.OpenAI configurada
        com a base_url e api_key corretas.
      - Para 'gemini': instancia de google.genai.Client.
    """
    if provider == "groq":
        if openai_lib is None:
            print(
                "ERRO: a biblioteca 'openai' nao esta instalada.\n"
                "Instale as dependencias com: pip install -r requirements.txt",
                file=sys.stderr,
            )
            sys.exit(1)
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            print(
                "ERRO: a variavel de ambiente GROQ_API_KEY nao foi encontrada.\n"
                "Defina sua chave em '.env' (copie '.env.example') ou exporte no terminal.\n"
                "Gere uma chave gratuita em: https://console.groq.com/keys\n"
                "Use --dry-run para testar sem chave de API.",
                file=sys.stderr,
            )
            sys.exit(1)
        return openai_lib.OpenAI(api_key=api_key, base_url=GROQ_BASE_URL)

    elif provider == "openrouter":
        if openai_lib is None:
            print(
                "ERRO: a biblioteca 'openai' nao esta instalada.\n"
                "Instale as dependencias com: pip install -r requirements.txt",
                file=sys.stderr,
            )
            sys.exit(1)
        api_key = os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            print(
                "ERRO: a variavel de ambiente OPENROUTER_API_KEY nao foi encontrada.\n"
                "Defina sua chave em '.env' (copie '.env.example') ou exporte no terminal.\n"
                "Gere uma chave gratuita em: https://openrouter.ai/keys\n"
                "Use --dry-run para testar sem chave de API.",
                file=sys.stderr,
            )
            sys.exit(1)
        return openai_lib.OpenAI(api_key=api_key, base_url=OPENROUTER_BASE_URL)

    else:  # gemini
        if genai is None:
            print(
                "ERRO: a biblioteca 'google-genai' nao esta instalada.\n"
                "Instale as dependencias com: pip install -r requirements.txt",
                file=sys.stderr,
            )
            sys.exit(1)
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            print(
                "ERRO: a variavel de ambiente GEMINI_API_KEY nao foi encontrada.\n"
                "Defina sua chave em '.env' (copie '.env.example') ou exporte no terminal.\n"
                "Use --dry-run para testar sem chave de API.",
                file=sys.stderr,
            )
            sys.exit(1)
        return genai.Client(api_key=api_key)


def obter_retry_delay(e: Exception, tentativa: int) -> float:
    """
    Extrai o tempo de espera (retry delay) recomendado pela API Gemini a partir
    da mensagem ou detalhes do erro. Caso nao encontre um tempo especifico,
    retorna um tempo de espera seguro (minimo 10s, dobrando a cada tentativa,
    maximo 60s) para nao estourar a cota de requisicoes por minuto (ex.: 10 RPM).
    """
    mensagem = str(e)

    # 1. Procura por padroes como "retry after 12.5s", "retry in 10s", "retry_delay: 15"
    match = re.search(
        r"retry\s*(?:after|in|delay[:\s]+)?\s*([0-9]+(?:\.[0-9]+)?)\s*s",
        mensagem,
        re.IGNORECASE,
    )
    if match:
        try:
            return float(match.group(1)) + 1.0  # +1s de margem de seguranca
        except ValueError:
            pass

    # 2. Procura por "wait X seconds" ou "X seconds"
    match_sec = re.search(
        r"([0-9]+(?:\.[0-9]+)?)\s*(?:seconds|segundos)",
        mensagem,
        re.IGNORECASE,
    )
    if match_sec:
        try:
            return float(match_sec.group(1)) + 1.0
        except ValueError:
            pass

    # 3. Fallback seguro para limite de 10 RPM: 10s, 20s, 40s, 60s...
    return float(min(60, 10 * (2 ** (tentativa - 1))))


def _chamar_openai_compat(
    client: "openai_lib.OpenAI",
    model: str,
    user_prompt: str,
    system_prompt: str | None,
    max_tokens: int,
) -> str:
    """Realiza uma chamada via OpenAI Chat Completions API (Groq / OpenRouter)."""
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_prompt})

    resposta = client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=max_tokens,
    )
    conteudo = resposta.choices[0].message.content
    return conteudo.strip() if conteudo else ""


def _chamar_gemini(
    client: "genai.Client",
    model: str,
    user_prompt: str,
    system_prompt: str | None,
    max_tokens: int,
) -> str:
    """Realiza uma chamada via SDK nativo do Google Gemini."""
    config_kwargs: dict = {"max_output_tokens": max_tokens}
    if system_prompt:
        config_kwargs["system_instruction"] = system_prompt

    resposta = client.models.generate_content(
        model=model,
        contents=user_prompt,
        config=genai.types.GenerateContentConfig(**config_kwargs),
    )
    return resposta.text.strip() if resposta.text else ""


def _e_erro_cota_diaria(msg: str) -> bool:
    """Retorna True se a mensagem de erro indica esgotamento de cota diaria (RPD)."""
    indicadores = [
        "GenerateRequestsPerDay",
        "PerDay",
        "daily",
        "day quota",
        "rate_limit_exceeded",  # Groq usa este para cota diaria
    ]
    msg_upper = msg.upper()
    return any(ind.upper() in msg_upper for ind in indicadores)


def _e_erro_rate_limit(msg: str, status_code: int | None) -> bool:
    """Retorna True se o erro e de limite de requisicoes por minuto (RPM / 429)."""
    if status_code == 429:
        return True
    msg_upper = msg.upper()
    return "429" in msg or "RESOURCE_EXHAUSTED" in msg_upper or "RATELIMIT" in msg_upper or "RATE_LIMIT" in msg_upper


def chamar_ia(
    client: object,
    model: str,
    user_prompt: str,
    system_prompt: str | None = None,
    max_tokens: int = MAX_TOKENS_WORKED_EXAMPLE,
    max_tentativas: int = MAX_TENTATIVAS_IA,
    delay_entre_chamadas: float = 0.0,
    provider: str = "groq",
) -> str:
    """
    Realiza uma chamada a API de IA (Groq, OpenRouter ou Gemini), com
    tratamento de falhas por meio de novas tentativas com espera inteligente.

    Retorna o texto da resposta.

    Levanta:
        QuotaDiariaExcedidaError: se a cota diaria (RPD) for excedida.
        RuntimeError: se a chamada falhar definitivamente apos esgotar tentativas.
    """
    ultimo_erro: Exception | None = None
    usa_gemini = (provider == "gemini")

    for tentativa in range(1, max_tentativas + 1):
        try:
            if usa_gemini:
                texto = _chamar_gemini(client, model, user_prompt, system_prompt, max_tokens)  # type: ignore[arg-type]
            else:
                texto = _chamar_openai_compat(client, model, user_prompt, system_prompt, max_tokens)  # type: ignore[arg-type]

            if delay_entre_chamadas > 0:
                time.sleep(delay_entre_chamadas)
            return texto

        except QuotaDiariaExcedidaError:
            raise

        except Exception as e:
            ultimo_erro = e
            msg_erro = str(e)
            status_code: int | None = getattr(e, "status_code", None) or getattr(e, "code", None)

            if _e_erro_cota_diaria(msg_erro):
                print(
                    "\n    [ERRO CRITICO - COTA DIARIA EXCEDIDA] "
                    "A cota diaria de requisicoes da API foi atingida. "
                    "Encerrando para evitar retentativas desnecessarias."
                )
                raise QuotaDiariaExcedidaError(
                    "Cota diaria de requisicoes atingida."
                ) from e

            elif _e_erro_rate_limit(msg_erro, status_code):
                _aguardar_rate_limit(e, tentativa, max_tentativas)

            elif status_code is not None and 500 <= status_code < 600:
                _aguardar_nova_tentativa(
                    tentativa, max_tentativas, f"erro no servidor da IA (HTTP {status_code})"
                )

            # Gemini-especifico: APIError com code nao-5xx e nao-429 => erro permanente
            elif genai_errors is not None and isinstance(e, genai_errors.APIError):
                raise RuntimeError(
                    f"Erro da API Gemini (HTTP {status_code}): {getattr(e, 'message', str(e))}"
                ) from e

            else:
                _aguardar_nova_tentativa(tentativa, max_tentativas, f"erro inesperado: {e}")

    raise RuntimeError(
        f"Falha ao comunicar com a IA apos {max_tentativas} tentativa(s). "
        f"Ultimo erro: {ultimo_erro}"
    )


def _aguardar_rate_limit(e: Exception, tentativa: int, max_tentativas: int) -> None:
    """Aguarda o tempo adequado antes de tentar novamente apos Rate Limit (429 / RPM)."""
    if tentativa >= max_tentativas:
        return
    espera = obter_retry_delay(e, tentativa)
    print(
        f"    [Rate Limit] Limite de requisicoes por minuto (RPM) atingido. "
        f"Aguardando {espera:.1f}s antes de tentar novamente (tentativa {tentativa}/{max_tentativas})..."
    )
    time.sleep(espera)


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
    client: object | None,
    model: str,
    tema: str,
    k: int,
    dry_run: bool = False,
    delay: float = 0.0,
    provider: str = "groq",
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
            client,
            model,
            user_prompt,
            system_prompt=system_prompt,
            max_tokens=MAX_TOKENS_CONCEITOS,
            delay_entre_chamadas=delay,
            provider=provider,
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


def gerar_worked_examples_do_conceito(
    client: object | None,
    model: str,
    tema: str,
    conceito: str,
    verbos: list[VerboBloom],
    contexto_bloom: str,
    dry_run: bool = False,
    delay: float = 0.0,
    provider: str = "groq",
) -> list[dict[str, str]]:
    """
    Gera N worked examples para um único conceito em UMA ÚNICA CHAMADA à IA.
    Retorna uma lista de dicionarios: [{"indice": int, "verbo": str, "conteudo": str}, ...].
    """
    n = len(verbos)
    if dry_run:
        exemplos = []
        for idx, verbo in enumerate(verbos, start=1):
            categoria_txt = f' (categoria "{verbo.categoria}")' if verbo.categoria else ""
            conteudo = (
                "[MODO --dry-run: conteudo simulado, nenhuma chamada real a IA foi feita]\n\n"
                "Este arquivo representa o worked example que seria gerado pela IA para:\n"
                f"  - Tema: {tema}\n"
                f"  - Conceito: {conceito}\n"
                f"  - Verbo da Taxonomia de Bloom: {verbo.verbo}{categoria_txt}\n\n"
                "Execute novamente sem a flag --dry-run (com chave de API "
                "configurada) para gerar o worked example real."
            )
            exemplos.append({
                "indice": idx,
                "verbo": verbo.verbo,
                "conteudo": conteudo,
            })
        return exemplos

    assert client is not None

    verbos_formatados = "\n".join(
        f"{idx}. Verbo \"{v.verbo}\"" + (f' (categoria "{v.categoria}")' if v.categoria else "")
        for idx, v in enumerate(verbos, start=1)
    )

    system_prompt = (
        "Voce e um especialista em design instrucional e na Taxonomia de "
        "Bloom Revisada, elaborando material didatico. Responda SEMPRE e "
        "SOMENTE com um JSON valido (uma lista de objetos), sem nenhum texto "
        "adicional antes ou depois, e sem cercas de markdown (```).\n\n"
        "Abaixo esta o arquivo de referencia com os verbos da Taxonomia de Bloom, "
        "organizados por categoria cognitiva:\n\n"
        f"{contexto_bloom}"
    )

    user_prompt = (
        f'Gere EXATAMENTE {n} worked examples (exemplos resolvidos, explicados passo a passo) '
        f'que exercitem o conceito "{conceito}", dentro do tema "{tema}".\n\n'
        f'Cada worked example deve utilizar EXCLUSIVAMENTE o verbo da Taxonomia de Bloom correspondente abaixo:\n'
        f'{verbos_formatados}\n\n'
        'Cada worked example deve conter:\n'
        '1. Um enunciado claro do problema/tarefa a ser resolvida;\n'
        '2. A resolucao comentada, passo a passo, demonstrando o raciocinio;\n'
        '3. Uma breve conclusao ou observacao final.\n\n'
        f'Sinta-se livre para mencionar naturalmente outros conceitos relacionados a '
        f'"{conceito}" sempre que isso ajudar a explicar o raciocinio.\n\n'
        'Responda EXCLUSIVAMENTE em formato JSON contendo uma lista de objetos, no formato:\n'
        '[\n'
        '  {\n'
        '    "indice": 1,\n'
        '    "verbo": "<verbo exato utilizado>",\n'
        '    "conteudo": "<texto completo do worked example 1>"\n'
        '  },\n'
        '  ...\n'
        ']\n\n'
        'Nao inclua nenhum texto fora do JSON.'
    )

    max_tokens_lote = min(8192, MAX_TOKENS_WORKED_EXAMPLE * n)

    for tentativa in range(1, MAX_TENTATIVAS_IA + 1):
        texto = chamar_ia(
            client,
            model,
            user_prompt,
            system_prompt=system_prompt,
            max_tokens=max_tokens_lote,
            delay_entre_chamadas=delay,
            provider=provider,
        )
        dados = validar_json(texto)

        if isinstance(dados, list) and len(dados) > 0:
            exemplos_validos = []
            for idx, item in enumerate(dados, start=1):
                if isinstance(item, dict):
                    v_str = str(item.get("verbo", verbos[idx - 1].verbo if idx <= len(verbos) else "")).strip()
                    c_str = str(item.get("conteudo", "")).strip()
                    if c_str:
                        exemplos_validos.append({
                            "indice": idx,
                            "verbo": v_str or verbos[idx - 1].verbo,
                            "conteudo": c_str,
                        })

            if len(exemplos_validos) == n:
                return exemplos_validos
            elif len(exemplos_validos) > 0:
                print(
                    f"  [aviso] a IA retornou {len(exemplos_validos)} exemplo(s) em vez de {n}. "
                    "Prosseguindo com os exemplos recebidos."
                )
                return exemplos_validos

        print(
            f"  [aviso] a resposta da IA nao e um JSON valido de worked examples "
            f"(tentativa {tentativa}/{MAX_TENTATIVAS_IA})."
        )
        user_prompt = (
            "A resposta anterior nao estava em um formato JSON valido. "
            f'Gere novamente a lista de {n} worked examples para o conceito "{conceito}", '
            "respondendo EXCLUSIVAMENTE com uma lista JSON no formato "
            '[{"indice": 1, "verbo": "...", "conteudo": "..."}, ...]. Nao inclua texto adicional.'
        )

    raise RuntimeError(
        f"Nao foi possivel obter os worked examples em formato JSON valido da IA para o conceito '{conceito}' "
        f"apos {MAX_TENTATIVAS_IA} tentativas."
    )


# Mantida para compatibilidade
def gerar_worked_example(
    client: object | None,
    model: str,
    tema: str,
    conceito: str,
    verbo: VerboBloom,
    contexto_bloom: str,
    dry_run: bool = False,
    delay: float = 0.0,
    provider: str = "groq",
) -> str:
    res = gerar_worked_examples_do_conceito(
        client, model, tema, conceito, [verbo], contexto_bloom,
        dry_run=dry_run, delay=delay, provider=provider,
    )
    return res[0]["conteudo"] if res else ""


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


def salvar_conceitos(diretorio: Path, conceitos: list[str]) -> tuple[Path, Path]:
    """
    Salva a lista de K conceitos gerados pela IA em dois formatos:
      - conceitos.json: lista de strings em formato JSON
      - conceitos.txt: um conceito por linha em texto simples
    """
    caminho_json = diretorio / "conceitos.json"
    caminho_txt = diretorio / "conceitos.txt"

    caminho_json.write_text(
        json.dumps(conceitos, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    caminho_txt.write_text(
        "\n".join(conceitos) + "\n", encoding="utf-8"
    )
    return caminho_json, caminho_txt


def salvar_tabela_worked_examples(diretorio: Path, registros: list[dict[str, str]]) -> Path:
    """
    Salva a tabela worked_examples.csv contendo os metadados de cada exemplo gerado:
      - id: identificador unico do exemplo (ex.: worked_example_busca_vetorial_1)
      - verbo_bloom: verbo da Taxonomia de Bloom utilizado
      - conceito: conceito trabalhado no exemplo
    """
    caminho_csv = diretorio / "worked_examples.csv"
    cabeçalhos = ["id", "verbo_bloom", "conceito"]

    with caminho_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=cabeçalhos)
        writer.writeheader()
        writer.writerows(registros)

    return caminho_csv


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
    print(f"Provedor...................: {args.provider}")
    print(f"Tema.......................: {args.tema}")
    print(f"Conceitos a gerar (--C)....: {args.C}")
    print(f"Worked examples/conceito (--N): {args.N}")
    print(f"Modelo.....................: {args.model}")
    if args.delay > 0:
        print(f"Delay entre chamadas.......: {args.delay}s")
    if args.dry_run:
        print("Modo.......................: DRY-RUN (sem chamadas reais a IA)")
    print("-" * 60)

    # Com geracao por lote (1 chamada por conceito), a estimativa de chamadas e:
    # 1 chamada (para K conceitos) + K chamadas (1 por conceito) = K + 1 chamadas total.
    estimativa_chamadas = args.C + 1
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
    client = None if args.dry_run else obter_cliente_ia(args.provider)

    registros_csv: list[dict[str, str]] = []
    total_gerado = 0
    total_falhas = 0

    try:
        # -- Etapa 1: geracao dos conceitos -------------------------------------
        print("\nEtapa 1: gerando conceitos com a IA...")
        conceitos = gerar_conceitos(
            client, args.model, args.tema, args.C,
            dry_run=args.dry_run, delay=args.delay, provider=args.provider,
        )

        print(f"Conceitos gerados ({len(conceitos)}):")
        for c in conceitos:
            print(f"  - {c}")

        # -- Etapas 2-4: para cada conceito, gerar N worked examples em 1 chamada ----
        criar_diretorio_saida(OUTPUT_DIR)

        caminho_json, caminho_txt = salvar_conceitos(OUTPUT_DIR, conceitos)
        print(f"\nLista de conceitos salva em:")
        print(f"  - {_caminho_para_exibicao(caminho_json)}")
        print(f"  - {_caminho_para_exibicao(caminho_txt)}")

        print(f"\nEtapas 2-4: gerando {args.N} worked example(s) por conceito (1 chamada por conceito)...")
        for conceito in conceitos:
            slug = gerar_slug(conceito)
            print(f"\nConceito: {conceito}  (slug: {slug})")

            verbos_sorteados = [selecionar_verbo_aleatorio(verbos) for _ in range(args.N)]
            v_nomes = ", ".join(f"'{v.verbo}'" for v in verbos_sorteados)
            print(f"  [1 chamada para {args.N} exemplos] verbos sorteados: [{v_nomes}]...", end=" ", flush=True)

            try:
                exemplos_gerados = gerar_worked_examples_do_conceito(
                    client,
                    args.model,
                    args.tema,
                    conceito,
                    verbos_sorteados,
                    contexto_bloom,
                    dry_run=args.dry_run,
                    delay=args.delay,
                    provider=args.provider,
                )
                print("OK")
                for item in exemplos_gerados:
                    idx = item["indice"]
                    v_usado = item["verbo"]
                    conteudo = item["conteudo"]

                    caminho = salvar_worked_example(OUTPUT_DIR, slug, idx, conteudo)
                    exemplo_id = f"worked_example_{slug}_{idx}"
                    registros_csv.append({
                        "id": exemplo_id,
                        "verbo_bloom": v_usado,
                        "conceito": conceito,
                    })
                    print(f"    - Exemplo {idx}/{args.N} (verbo: '{v_usado}') -> {_caminho_para_exibicao(caminho)}")
                    total_gerado += 1
            except RuntimeError as e:
                print(f"FALHOU ({e})")
                total_falhas += 1

    except QuotaDiariaExcedidaError as e:
        print(f"\nERRO DE COTA: {e}")
        print("A execução foi interrompida graciosamente devido ao esgotamento da cota diária de requisições.")
        if registros_csv:
            caminho_csv = salvar_tabela_worked_examples(OUTPUT_DIR, registros_csv)
            print(f"Metadados dos exemplos salvos até o momento foram salvos em: {_caminho_para_exibicao(caminho_csv)}")
        return 1

    if registros_csv:
        caminho_csv = salvar_tabela_worked_examples(OUTPUT_DIR, registros_csv)
        print(f"\nTabela de metadados salva em: {_caminho_para_exibicao(caminho_csv)}")

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
