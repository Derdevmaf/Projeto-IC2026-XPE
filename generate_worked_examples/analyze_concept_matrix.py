#!/usr/bin/env python3
"""
analyze_concept_matrix.py
--------------------------
Gera uma matriz de co-ocorrência de conceitos nos exercícios gerados.

Cada linha = conceito "dono" dos exercícios (ex: Funções)
Cada coluna = conceito mencionado nos textos dos exercícios (ex: Listas)
Valor = quantidade de vezes que o conceito da coluna aparece
        nos exercícios do conceito da linha.

Saída:
  - concept_matrix.csv       → matriz numérica bruta
  - concept_matrix.html      → visualização interativa com heatmap
  - dependency_report.md     → relatório de dependências inferidas
"""

import json
import os
import re
import csv
import unicodedata
from pathlib import Path
from collections import defaultdict

# ─── Caminhos ────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent
EXAMPLES_DIR = BASE_DIR / "worked_examples"
CONCEITOS_JSON = EXAMPLES_DIR / "conceitos.json"
OUTPUT_CSV = EXAMPLES_DIR / "concept_matrix.csv"
OUTPUT_HTML = EXAMPLES_DIR / "concept_matrix.html"
OUTPUT_REPORT = EXAMPLES_DIR / "dependency_report.md"


# ─── Utilitários ─────────────────────────────────────────────────────────────

def normalize(text: str) -> str:
    """Normaliza texto: minúsculas, sem acentos, sem pontuação extra."""
    text = text.lower()
    text = unicodedata.normalize("NFKD", text)
    text = "".join(c for c in text if not unicodedata.combining(c))
    return text


def concept_to_slug(concept: str) -> str:
    """Converte conceito para slug de arquivo (mesmo padrão do gerador)."""
    slug = normalize(concept)
    slug = re.sub(r"[^a-z0-9]+", "_", slug)
    slug = slug.strip("_")
    return slug


def load_concepts() -> list:
    with open(CONCEITOS_JSON, encoding="utf-8") as f:
        return json.load(f)


def load_exercises_for_concept(concept: str) -> list:
    """Lê todos os arquivos worked_example_<slug>_*.txt para um conceito."""
    slug = concept_to_slug(concept)
    pattern = f"worked_example_{slug}_*.txt"
    files = sorted(EXAMPLES_DIR.glob(pattern))
    texts = []
    for fp in files:
        content = fp.read_text(encoding="utf-8", errors="ignore").strip()
        if content:
            texts.append(content)
    return texts


def count_mentions(text: str, target_concept: str) -> int:
    """
    Conta quantas vezes o target_concept (e suas variações normalizadas)
    aparecem no texto.
    Usa busca por substring normalizada para ser robusto a acentuação.
    """
    norm_text = normalize(text)
    terms = build_search_terms(target_concept)
    total = 0
    for term in terms:
        norm_term = normalize(term)
        if not norm_term:
            continue
        count = len(re.findall(re.escape(norm_term), norm_text))
        total += count
    return total


def build_search_terms(concept: str) -> list:
    """
    Gera variações de busca para um conceito.
    Ex: "Laços de Repetição" → ["laços de repetição", "loop", "loops", ...]
    """
    extras_map = {
        "Variáveis": ["variavel", "variáveis", "variavel", "variable"],
        "Tipos de Dados": ["tipo de dado", "tipos de dados", "tipo dado", "int", "float", "string", "bool"],
        "Estruturas de Controle": ["estrutura de controle", "if", "else", "elif", "condicional", "condicionais"],
        "Laços de Repetição": ["laco de repeticao", "laço de repetição", "loop", "loops", "for", "while", "iteracao", "iteração"],
        "Funções": ["funcao", "função", "funções", "funcoes", "def ", "parametro", "parâmetro", "retorno"],
        "Módulos": ["modulo", "módulo", "módulos", "modulos", "import", "biblioteca"],
        "Tratamento de Erros": ["tratamento de erro", "try", "except", "raise", "exception", "erro", "erros"],
        "Manipulação de Arquivos": ["manipulacao de arquivo", "arquivo", "arquivos", "open(", "leitura", "escrita", "file"],
        "Estruturas de Dados": ["estrutura de dado", "estrutura de dados", "pilha", "fila", "arvore", "grafo"],
        "Listas": ["lista", "listas", "append", "list"],
        "Tuplas": ["tupla", "tuplas", "tuple"],
        "Dicionários": ["dicionario", "dicionários", "dicionários", "dict", "chave", "valor", "key", "value"],
        "Programação Orientada a Objetos": ["poo", "oop", "classe", "classes", "objeto", "objetos", "heranca", "herança", "polimorfismo", "encapsulamento"],
        "Bibliotecas e Frameworks": ["biblioteca", "bibliotecas", "framework", "frameworks", "numpy", "pandas", "flask", "django"],
        "Desenvolvimento de Aplicações": ["desenvolvimento de aplicacao", "aplicacao", "aplicações", "aplicativo", "app", "sistema"],
    }
    terms = [concept]
    if concept in extras_map:
        terms.extend(extras_map[concept])
    return terms


# ─── Construção da Matriz ────────────────────────────────────────────────────

def build_matrix(concepts: list) -> dict:
    """
    Retorna matrix[conceito_linha][conceito_coluna] = contagem de menções.
    """
    matrix = {c: {other: 0 for other in concepts} for c in concepts}

    for concept in concepts:
        exercises = load_exercises_for_concept(concept)
        if not exercises:
            print(f"  WARNING: Nenhum exercício encontrado para: {concept}")
            continue

        combined_text = " ".join(exercises)
        print(f"  {concept}: {len(exercises)} exercício(s) carregado(s)")

        for other_concept in concepts:
            matrix[concept][other_concept] = count_mentions(combined_text, other_concept)

    return matrix


# ─── Exportação CSV ──────────────────────────────────────────────────────────

def save_csv(matrix: dict, concepts: list):
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Conceito (linha \\ coluna)"] + concepts)
        for row_concept in concepts:
            row = [row_concept] + [matrix[row_concept][col] for col in concepts]
            writer.writerow(row)
    print(f"\n  CSV salvo: {OUTPUT_CSV}")


# ─── Inferência de Dependências ───────────────────────────────────────────────

def infer_dependencies(matrix: dict, concepts: list, threshold: int = 1) -> list:
    """
    Retorna lista de dependências inferidas ordenadas por contagem descendente.
    Se exercícios do conceito A mencionam conceito B >= threshold vezes,
    então B é provável pré-requisito de A.
    """
    deps = []
    for row_concept in concepts:
        for col_concept in concepts:
            if row_concept == col_concept:
                continue
            count = matrix[row_concept][col_concept]
            if count >= threshold:
                deps.append({
                    "concept": row_concept,
                    "prerequisite": col_concept,
                    "count": count,
                })
    deps.sort(key=lambda d: d["count"], reverse=True)
    return deps


# ─── Exportação HTML (Heatmap Interativo) ────────────────────────────────────

def save_html(matrix: dict, concepts: list):
    data_rows = []
    all_values = []
    for row_concept in concepts:
        row = [matrix[row_concept][col] for col in concepts]
        data_rows.append(row)
        all_values.extend(row)

    max_val = max((v for v in all_values if v > 0), default=1)

    concepts_json = json.dumps(concepts, ensure_ascii=False)
    matrix_json = json.dumps(data_rows)

    dependencies = infer_dependencies(matrix, concepts)
    dep_list_html = ""
    for dep in dependencies:
        dep_list_html += (
            f'<div class="dep-card">'
            f'<span class="dep-arrow">&#8594;</span>'
            f'<span class="dep-text">'
            f'Ensinar <strong>{dep["prerequisite"]}</strong> '
            f'antes de <strong>{dep["concept"]}</strong> '
            f'<span class="dep-count">({dep["count"]} men&#231;&#245;es)</span>'
            f'</span></div>\n'
        )

    html = (
        "<!DOCTYPE html>\n"
        '<html lang="pt-BR">\n'
        "<head>\n"
        '  <meta charset="UTF-8">\n'
        '  <meta name="viewport" content="width=device-width, initial-scale=1.0">\n'
        "  <title>Matriz de Co-ocorr&#234;ncia de Conceitos</title>\n"
        '  <link rel="preconnect" href="https://fonts.googleapis.com">\n'
        '  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">\n'
        "  <style>\n"
        "    :root {\n"
        "      --bg: #0f1117;\n"
        "      --surface: #1a1d27;\n"
        "      --surface2: #22263a;\n"
        "      --border: #2e3250;\n"
        "      --accent: #6366f1;\n"
        "      --accent2: #a78bfa;\n"
        "      --text: #e2e8f0;\n"
        "      --text-muted: #94a3b8;\n"
        "    }\n"
        "    * { box-sizing: border-box; margin: 0; padding: 0; }\n"
        "    body { font-family: 'Inter', sans-serif; background: var(--bg); color: var(--text); min-height: 100vh; padding: 2rem; }\n"
        "    .header { text-align: center; margin-bottom: 3rem; }\n"
        "    .header h1 { font-size: 2rem; font-weight: 700; background: linear-gradient(135deg, var(--accent), var(--accent2)); -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text; margin-bottom: 0.5rem; }\n"
        "    .header p { color: var(--text-muted); font-size: 0.95rem; max-width: 600px; margin: 0 auto; }\n"
        "    .legend { display: flex; align-items: center; justify-content: center; gap: 1rem; margin-bottom: 2rem; flex-wrap: wrap; }\n"
        "    .legend-label { font-size: 0.8rem; color: var(--text-muted); }\n"
        "    .legend-gradient { width: 180px; height: 14px; border-radius: 7px; background: linear-gradient(to right, #151821, #1e3a5f, #2563eb, #7c3aed, #f59e0b); border: 1px solid var(--border); }\n"
        "    .matrix-container { overflow-x: auto; border-radius: 16px; border: 1px solid var(--border); background: var(--surface); padding: 1.5rem; margin-bottom: 3rem; box-shadow: 0 8px 32px rgba(0,0,0,0.4); }\n"
        "    table { border-collapse: collapse; width: max-content; min-width: 100%; }\n"
        "    th, td { padding: 0; text-align: center; }\n"
        "    .corner { padding: 0.5rem; color: var(--text-muted); font-size: 0.7rem; text-align: right; vertical-align: bottom; min-width: 160px; }\n"
        "    .col-header { vertical-align: bottom; padding-bottom: 0.5rem; }\n"
        "    .col-header-inner { writing-mode: vertical-rl; transform: rotate(180deg); white-space: nowrap; font-size: 0.72rem; font-weight: 500; color: var(--text-muted); padding: 0.5rem 0.4rem; transition: color 0.2s; }\n"
        "    .col-header-inner.highlighted { color: var(--accent2); }\n"
        "    .row-header { text-align: right; padding-right: 0.75rem; font-size: 0.75rem; font-weight: 500; color: var(--text-muted); white-space: nowrap; min-width: 160px; transition: color 0.2s; }\n"
        "    .row-header.highlighted { color: var(--accent2); }\n"
        "    .cell { width: 44px; height: 44px; font-size: 0.8rem; font-weight: 600; cursor: pointer; transition: transform 0.15s, box-shadow 0.15s, filter 0.15s; border: 1px solid transparent; }\n"
        "    .cell:hover { transform: scale(1.15); z-index: 10; box-shadow: 0 4px 20px rgba(0,0,0,0.5); filter: brightness(1.3); border-color: rgba(255,255,255,0.3); }\n"
        "    .cell.diagonal { opacity: 0.4; }\n"
        "    .tooltip { display: none; position: fixed; background: #1e2235; border: 1px solid var(--border); border-radius: 10px; padding: 0.75rem 1rem; font-size: 0.8rem; z-index: 1000; pointer-events: none; max-width: 280px; box-shadow: 0 8px 24px rgba(0,0,0,0.5); }\n"
        "    .tooltip strong { color: var(--accent2); }\n"
        "    .tooltip .count { font-size: 1.1rem; font-weight: 700; color: var(--text); }\n"
        "    .section-title { font-size: 1.2rem; font-weight: 600; color: var(--text); margin-bottom: 1.25rem; display: flex; align-items: center; gap: 0.5rem; }\n"
        "    .section-title::before { content: ''; display: inline-block; width: 4px; height: 1.2em; background: linear-gradient(to bottom, var(--accent), var(--accent2)); border-radius: 2px; }\n"
        "    .dep-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(320px, 1fr)); gap: 0.75rem; margin-bottom: 3rem; }\n"
        "    .dep-card { background: var(--surface); border: 1px solid var(--border); border-radius: 10px; padding: 0.85rem 1.1rem; display: flex; align-items: center; gap: 0.75rem; transition: border-color 0.2s, transform 0.15s; }\n"
        "    .dep-card:hover { border-color: var(--accent); transform: translateY(-1px); }\n"
        "    .dep-arrow { font-size: 1.2rem; color: var(--accent); }\n"
        "    .dep-text { font-size: 0.85rem; color: var(--text-muted); line-height: 1.4; }\n"
        "    .dep-text strong { color: var(--text); }\n"
        "    .dep-count { font-size: 0.75rem; background: var(--surface2); border-radius: 999px; padding: 0.1em 0.5em; margin-left: 0.25rem; color: var(--accent2); }\n"
        "    .stats-bar { display: flex; gap: 1rem; flex-wrap: wrap; margin-bottom: 2.5rem; }\n"
        "    .stat-card { background: var(--surface); border: 1px solid var(--border); border-radius: 12px; padding: 1rem 1.5rem; flex: 1; min-width: 140px; text-align: center; }\n"
        "    .stat-value { font-size: 1.8rem; font-weight: 700; background: linear-gradient(135deg, var(--accent), var(--accent2)); -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text; }\n"
        "    .stat-label { font-size: 0.75rem; color: var(--text-muted); margin-top: 0.2rem; }\n"
        "    footer { text-align: center; color: var(--text-muted); font-size: 0.75rem; margin-top: 2rem; padding-top: 1rem; border-top: 1px solid var(--border); }\n"
        "  </style>\n"
        "</head>\n"
        "<body>\n"
        '  <div class="header">\n'
        "    <h1>&#129504; Matriz de Co-ocorr&#234;ncia de Conceitos</h1>\n"
        "    <p>Cada c&#233;lula mostra quantas vezes o conceito da coluna &#233; mencionado nos exerc&#237;cios do conceito da linha. Passe o mouse para ver detalhes.</p>\n"
        "  </div>\n"
        '  <div id="stats-bar" class="stats-bar"></div>\n'
        '  <div class="legend">\n'
        '    <span class="legend-label">Poucas men&#231;&#245;es</span>\n'
        '    <div class="legend-gradient"></div>\n'
        '    <span class="legend-label">Muitas men&#231;&#245;es</span>\n'
        "  </div>\n"
        '  <div class="matrix-container"><table id="matrix-table"></table></div>\n'
        '  <div class="section-title">&#128204; Depend&#234;ncias Inferidas (pr&#233;-requisitos)</div>\n'
        '  <div class="dep-grid">\n'
        + dep_list_html
        + "  </div>\n"
        '  <footer>Gerado automaticamente por analyze_concept_matrix.py &bull; Projeto IC 2026 XPE</footer>\n'
        '  <div class="tooltip" id="tooltip"></div>\n'
        "  <script>\n"
        f"    const concepts = {concepts_json};\n"
        f"    const matrix = {matrix_json};\n"
        f"    const maxVal = {max_val};\n"
        "\n"
        "    let totalMentions = 0, strongDeps = 0;\n"
        "    concepts.forEach((r, i) => {\n"
        "      concepts.forEach((c, j) => {\n"
        "        if (i !== j) { totalMentions += matrix[i][j]; if (matrix[i][j] >= 2) strongDeps++; }\n"
        "      });\n"
        "    });\n"
        "\n"
        "    const statsBar = document.getElementById('stats-bar');\n"
        "    [\n"
        "      [concepts.length, 'Conceitos'],\n"
        "      [totalMentions, 'Men\\u00e7\\u00f5es totais'],\n"
        "      [strongDeps, 'Depend\\u00eancias fortes (\\u22652)'],\n"
        "      [Math.round(totalMentions / (concepts.length * (concepts.length - 1) || 1) * 10) / 10, 'M\\u00e9dia men\\u00e7\\u00f5es/par']\n"
        "    ].forEach(([val, label]) => {\n"
        "      statsBar.innerHTML += `<div class=\"stat-card\"><div class=\"stat-value\">${val}</div><div class=\"stat-label\">${label}</div></div>`;\n"
        "    });\n"
        "\n"
        "    function heatColor(val, isdiag) {\n"
        "      if (isdiag) return 'rgba(100,100,120,0.2)';\n"
        "      if (val === 0) return '#151821';\n"
        "      const t = Math.min(val / maxVal, 1);\n"
        "      const colors = [[30,58,95],[37,99,235],[124,58,237],[245,158,11]];\n"
        "      const seg = Math.floor(t * 3);\n"
        "      const local = (t * 3) - seg;\n"
        "      const c0 = colors[Math.min(seg, 2)];\n"
        "      const c1 = colors[Math.min(seg + 1, 3)];\n"
        "      const r = Math.round(c0[0] + (c1[0]-c0[0]) * local);\n"
        "      const g = Math.round(c0[1] + (c1[1]-c0[1]) * local);\n"
        "      const b = Math.round(c0[2] + (c1[2]-c0[2]) * local);\n"
        "      return `rgb(${r},${g},${b})`;\n"
        "    }\n"
        "\n"
        "    function textColor(val, isdiag) {\n"
        "      if (isdiag || val === 0) return '#3d4460';\n"
        "      return Math.min(val / maxVal, 1) > 0.4 ? '#fff' : '#c8d4e8';\n"
        "    }\n"
        "\n"
        "    const table = document.getElementById('matrix-table');\n"
        "    const thead = document.createElement('thead');\n"
        "    const headerRow = document.createElement('tr');\n"
        "    headerRow.innerHTML = '<th class=\"corner\">linha &rarr; conceito<br>coluna &rarr; mencionado</th>';\n"
        "    concepts.forEach((c, j) => {\n"
        "      headerRow.innerHTML += `<th class=\"col-header\" id=\"col-${j}\"><div class=\"col-header-inner\">${c}</div></th>`;\n"
        "    });\n"
        "    thead.appendChild(headerRow);\n"
        "    table.appendChild(thead);\n"
        "\n"
        "    const tbody = document.createElement('tbody');\n"
        "    concepts.forEach((rowC, i) => {\n"
        "      const tr = document.createElement('tr');\n"
        "      tr.innerHTML = `<td class=\"row-header\" id=\"row-${i}\">${rowC}</td>`;\n"
        "      concepts.forEach((colC, j) => {\n"
        "        const val = matrix[i][j];\n"
        "        const isdiag = (i === j);\n"
        "        const td = document.createElement('td');\n"
        "        td.className = 'cell' + (isdiag ? ' diagonal' : '');\n"
        "        td.style.background = heatColor(val, isdiag);\n"
        "        td.style.color = textColor(val, isdiag);\n"
        "        td.textContent = (val === 0 && !isdiag) ? '\\u00b7' : val;\n"
        "        td.dataset.row = i; td.dataset.col = j; td.dataset.val = val;\n"
        "        tr.appendChild(td);\n"
        "      });\n"
        "      tbody.appendChild(tr);\n"
        "    });\n"
        "    table.appendChild(tbody);\n"
        "\n"
        "    const tooltip = document.getElementById('tooltip');\n"
        "    document.querySelectorAll('.cell').forEach(cell => {\n"
        "      cell.addEventListener('mouseenter', (e) => {\n"
        "        const i = +cell.dataset.row, j = +cell.dataset.col, val = +cell.dataset.val;\n"
        "        document.querySelectorAll('.row-header').forEach(el => el.classList.remove('highlighted'));\n"
        "        document.querySelectorAll('.col-header-inner').forEach(el => el.classList.remove('highlighted'));\n"
        "        document.getElementById('row-' + i)?.classList.add('highlighted');\n"
        "        document.querySelector('#col-' + j + ' .col-header-inner')?.classList.add('highlighted');\n"
        "        const dep = (i !== j && val > 0) ? `<br><em style=\"color:#a78bfa\">&ldquo;${concepts[j]}&rdquo; pode ser pr&eacute;-requisito de &ldquo;${concepts[i]}&rdquo;</em>` : '';\n"
        "        tooltip.innerHTML = `<div><strong>Exerc&iacute;cios de:</strong> ${concepts[i]}</div><div><strong>Menciona:</strong> ${concepts[j]}</div><div class=\"count\" style=\"margin-top:4px\">${val} men&ccedil;&atilde;o(&#245;es)</div>${dep}`;\n"
        "        tooltip.style.display = 'block';\n"
        "      });\n"
        "      cell.addEventListener('mousemove', (e) => {\n"
        "        const x = e.clientX + 14, y = e.clientY - 10;\n"
        "        tooltip.style.left = (x + 280 > window.innerWidth ? e.clientX - 290 : x) + 'px';\n"
        "        tooltip.style.top = (y + 140 > window.innerHeight ? e.clientY - 150 : y) + 'px';\n"
        "      });\n"
        "      cell.addEventListener('mouseleave', () => {\n"
        "        tooltip.style.display = 'none';\n"
        "        document.querySelectorAll('.row-header').forEach(el => el.classList.remove('highlighted'));\n"
        "        document.querySelectorAll('.col-header-inner').forEach(el => el.classList.remove('highlighted'));\n"
        "      });\n"
        "    });\n"
        "  </script>\n"
        "</body>\n"
        "</html>\n"
    )

    with open(OUTPUT_HTML, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"  HTML salvo: {OUTPUT_HTML}")


# ─── Relatório Markdown ────────────────────────────────────────────────────────

def save_report(matrix: dict, concepts: list):
    deps = infer_dependencies(matrix, concepts, threshold=1)

    lines = [
        "# Relatório de Dependências entre Conceitos\n",
        "_Gerado automaticamente por `analyze_concept_matrix.py`_\n",
        "",
        "## Como ler",
        "- Cada dependência indica que os exercícios de **Conceito A** mencionam **Conceito B**.",
        "- Isso sugere que **B deve ser ensinado antes de A**.",
        "",
        "## Tabela completa de menções (>= 1)\n",
        "| Exercícios de (linha) | Menciona (coluna) | Menções |",
        "|---|---|:---:|",
    ]

    for dep in deps:
        lines.append(f"| {dep['concept']} | {dep['prerequisite']} | {dep['count']} |")

    lines += [
        "",
        "## Dependências fortes (menções >= 2)\n",
        "| Conceito | Pré-requisito sugerido | Menções |",
        "|---|---|:---:|",
    ]
    for dep in deps:
        if dep["count"] >= 2:
            lines.append(f"| {dep['concept']} | {dep['prerequisite']} | {dep['count']} |")

    lines += [
        "",
        "## Sugestão de ordem de ensino",
        "",
        "_Conceitos com menos menções recebidas de outros exercícios tendem a ser mais fundamentais._",
        "",
    ]

    in_degree = defaultdict(int)
    for dep in deps:
        if dep["count"] >= 1:
            in_degree[dep["concept"]] += dep["count"]

    ordered = sorted(concepts, key=lambda c: in_degree.get(c, 0))
    for rank, concept in enumerate(ordered, 1):
        lines.append(f"{rank}. **{concept}** (total de menções recebidas: {in_degree.get(concept, 0)})")

    with open(OUTPUT_REPORT, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"  Relatório salvo: {OUTPUT_REPORT}")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    print("Carregando conceitos...")
    concepts = load_concepts()
    print(f"   {len(concepts)} conceitos encontrados.\n")

    print("Construindo matriz de co-ocorrência...")
    matrix = build_matrix(concepts)

    print("\nSalvando resultados...")
    save_csv(matrix, concepts)
    save_html(matrix, concepts)
    save_report(matrix, concepts)

    print("\nConcluido! Abra concept_matrix.html para visualizar.")
    print(f"   Diretório: {EXAMPLES_DIR}\n")

    # Imprimir matriz no terminal
    print("\nMatriz de co-ocorrência:\n")
    header = f"{'':35s}" + "".join(f"{c[:10]:>12}" for c in concepts)
    print(header)
    print("-" * len(header))
    for row_concept in concepts:
        row_str = f"{row_concept[:35]:35s}"
        for col_concept in concepts:
            val = matrix[row_concept][col_concept]
            display = "·" if (val == 0 and row_concept != col_concept) else str(val)
            row_str += f"{display:>12}"
        print(row_str)


if __name__ == "__main__":
    main()
