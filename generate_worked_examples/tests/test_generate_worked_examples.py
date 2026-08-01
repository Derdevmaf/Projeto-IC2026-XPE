"""
Testes automatizados para generate_worked_examples.py

Execute a partir da pasta do projeto com:
    python -m unittest discover -s tests -v

A maior parte dos testes NAO faz chamadas reais a IA: as chamadas sao
substituidas por "mocks" (unittest.mock) ou o modo --dry-run e usado.
Isso permite validar toda a logica do programa (parsing de argumentos,
sorteio de verbos, geracao de nomes de arquivo, escrita de arquivos,
tratamento de JSON invalido, novas tentativas em falhas de rede etc.)
sem precisar de uma chave GEMINI_API_KEY nem gastar creditos.
"""

from __future__ import annotations

import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from google import genai
from google.genai import errors

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import generate_worked_examples as gwe  # noqa: E402


# ---------------------------------------------------------------------------
# Auxiliares para simular respostas e erros da API
# ---------------------------------------------------------------------------


def _fake_status_error(status_code: int, message: str = "erro simulado"):
    e = MagicMock(spec=errors.APIError)
    e.code = status_code
    e.message = message
    return e


def _fake_text_response(text: str) -> MagicMock:
    resposta = MagicMock()
    resposta.text = text
    return resposta


# ---------------------------------------------------------------------------
# gerar_slug
# ---------------------------------------------------------------------------


class TestGerarSlug(unittest.TestCase):
    def test_minusculas_e_espacos(self):
        self.assertEqual(gwe.gerar_slug("Tokens"), "tokens")
        self.assertEqual(gwe.gerar_slug("Busca Vetorial"), "busca_vetorial")

    def test_acentos_sao_normalizados(self):
        self.assertEqual(
            gwe.gerar_slug("Programação Orientada a Objetos"), "programacao_orientada_a_objetos"
        )
        self.assertEqual(gwe.gerar_slug("Funções Lambda"), "funcoes_lambda")

    def test_caracteres_especiais_sao_removidos(self):
        self.assertEqual(gwe.gerar_slug("List Comprehension!"), "list_comprehension")
        self.assertEqual(gwe.gerar_slug("F-strings & Formatação"), "fstrings_formatacao")

    def test_espacos_multiplos_e_bordas(self):
        self.assertEqual(gwe.gerar_slug("  Decoradores  "), "decoradores")
        self.assertEqual(gwe.gerar_slug("A   B"), "a_b")

    def test_texto_sem_caracteres_validos_usa_fallback(self):
        self.assertEqual(gwe.gerar_slug("!!!"), "conceito")


# ---------------------------------------------------------------------------
# carregar_verbos_bloom
# ---------------------------------------------------------------------------


class TestCarregarVerbosBloom(unittest.TestCase):
    def setUp(self):
        self.tmpdir = Path(tempfile.mkdtemp())

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_arquivo_inexistente_levanta_erro(self):
        with self.assertRaises(FileNotFoundError):
            gwe.carregar_verbos_bloom(self.tmpdir / "nao_existe.txt")

    def test_arquivo_vazio_levanta_erro(self):
        caminho = self.tmpdir / "vazio.txt"
        caminho.write_text("", encoding="utf-8")
        with self.assertRaises(ValueError):
            gwe.carregar_verbos_bloom(caminho)

    def test_arquivo_so_com_comentarios_levanta_erro(self):
        caminho = self.tmpdir / "so_comentarios.txt"
        caminho.write_text("# Lembrar\n# Entender\n", encoding="utf-8")
        with self.assertRaises(ValueError):
            gwe.carregar_verbos_bloom(caminho)

    def test_leitura_basica_e_categorias(self):
        caminho = self.tmpdir / "verbos.txt"
        caminho.write_text("# Lembrar\nListar\nNomear\n\n# Criar\nCompor\n", encoding="utf-8")
        verbos = gwe.carregar_verbos_bloom(caminho)
        self.assertEqual(len(verbos), 3)
        self.assertEqual(verbos[0].verbo, "Listar")
        self.assertEqual(verbos[0].categoria, "Lembrar")
        self.assertEqual(verbos[2].verbo, "Compor")
        self.assertEqual(verbos[2].categoria, "Criar")

    def test_duplicatas_entre_categorias_sao_removidas(self):
        caminho = self.tmpdir / "dup.txt"
        caminho.write_text("# Lembrar\nCitar\n# Entender\nCitar\nDescrever\n", encoding="utf-8")
        verbos = gwe.carregar_verbos_bloom(caminho)
        nomes = [v.verbo for v in verbos]
        self.assertEqual(nomes.count("Citar"), 1)
        self.assertIn("Descrever", nomes)

    def test_arquivo_real_do_projeto_carrega_sem_erros(self):
        verbos = gwe.carregar_verbos_bloom(gwe.BLOOM_VERBS_PATH)
        self.assertGreater(len(verbos), 50)
        categorias = {v.categoria for v in verbos}
        for esperado in ["Lembrar", "Entender", "Aplicar", "Analisar", "Avaliar", "Criar"]:
            self.assertIn(esperado, categorias)


class TestSelecionarVerboAleatorio(unittest.TestCase):
    def test_retorna_item_da_lista(self):
        verbos = [gwe.VerboBloom("A"), gwe.VerboBloom("B"), gwe.VerboBloom("C")]
        for _ in range(20):
            self.assertIn(gwe.selecionar_verbo_aleatorio(verbos), verbos)


# ---------------------------------------------------------------------------
# validar_json / extrair_json
# ---------------------------------------------------------------------------


class TestValidarJson(unittest.TestCase):
    def test_json_valido_simples(self):
        self.assertEqual(gwe.validar_json('["a", "b"]'), ["a", "b"])

    def test_json_com_cercas_markdown(self):
        self.assertEqual(gwe.validar_json('```json\n["a", "b", "c"]\n```'), ["a", "b", "c"])

    def test_json_com_cercas_sem_linguagem(self):
        self.assertEqual(gwe.validar_json('```\n["x"]\n```'), ["x"])

    def test_json_invalido_retorna_none(self):
        self.assertIsNone(gwe.validar_json("isto nao e json"))
        self.assertIsNone(gwe.validar_json("[1, 2,"))


# ---------------------------------------------------------------------------
# salvar_worked_example
# ---------------------------------------------------------------------------


class TestSalvarWorkedExample(unittest.TestCase):
    def setUp(self):
        self.tmpdir = Path(tempfile.mkdtemp())

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_nome_e_conteudo_do_arquivo(self):
        caminho = gwe.salvar_worked_example(self.tmpdir, "busca_vetorial", 2, "conteudo de teste")
        self.assertEqual(caminho.name, "worked_example_busca_vetorial_2.txt")
        self.assertTrue(caminho.exists())
        self.assertEqual(caminho.read_text(encoding="utf-8").strip(), "conteudo de teste")


# ---------------------------------------------------------------------------
# chamar_ia (tratamento de falhas de comunicacao)
# ---------------------------------------------------------------------------


class TestChamarIa(unittest.TestCase):
    def test_sucesso_na_primeira_tentativa(self):
        client = MagicMock()
        client.models.generate_content.return_value = _fake_text_response("ola mundo")
        resultado = gwe.chamar_ia(client, "gemini-2.5-flash", "diga ola")
        self.assertEqual(resultado, "ola mundo")
        self.assertEqual(client.models.generate_content.call_count, 1)

    @patch("generate_worked_examples.time.sleep", return_value=None)
    def test_retentativa_apos_rate_limit_e_sucesso(self, _mock_sleep):
        client = MagicMock()
        client.models.generate_content.side_effect = [
            _fake_status_error(429, "rate limited"),
            _fake_text_response("agora funcionou"),
        ]
        resultado = gwe.chamar_ia(client, "gemini-2.5-flash", "prompt")
        self.assertEqual(resultado, "agora funcionou")
        self.assertEqual(client.models.generate_content.call_count, 2)

    @patch("generate_worked_examples.time.sleep", return_value=None)
    def test_erro_5xx_esgota_tentativas(self, _mock_sleep):
        client = MagicMock()
        client.models.generate_content.side_effect = _fake_status_error(500, "server error")
        with self.assertRaises(RuntimeError):
            gwe.chamar_ia(client, "gemini-2.5-flash", "prompt", max_tentativas=2)
        self.assertEqual(client.models.generate_content.call_count, 2)

    def test_erro_4xx_nao_tenta_de_novo(self):
        client = MagicMock()
        client.models.generate_content.side_effect = _fake_status_error(400, "bad request")
        with self.assertRaises(RuntimeError):
            gwe.chamar_ia(client, "gemini-2.5-flash", "prompt", max_tentativas=3)
        self.assertEqual(client.models.generate_content.call_count, 1)

    @patch("generate_worked_examples.time.sleep", return_value=None)
    def test_erro_de_conexao_esgota_tentativas(self, _mock_sleep):
        client = MagicMock()
        client.models.generate_content.side_effect = Exception("connection error")
        with self.assertRaises(RuntimeError):
            gwe.chamar_ia(client, "gemini-2.5-flash", "prompt", max_tentativas=2)
        self.assertEqual(client.models.generate_content.call_count, 2)


# ---------------------------------------------------------------------------
# gerar_conceitos (Etapa 1)
# ---------------------------------------------------------------------------


class TestGerarConceitos(unittest.TestCase):
    def test_dry_run_gera_conceitos_placeholder(self):
        conceitos = gwe.gerar_conceitos(None, "modelo", "tema", 4, dry_run=True)
        self.assertEqual(len(conceitos), 4)

    def test_resposta_valida_de_primeira(self):
        client = MagicMock()
        client.models.generate_content.return_value = _fake_text_response('["Variaveis", "Listas", "Funcoes"]')
        conceitos = gwe.gerar_conceitos(client, "modelo", "Python", 3)
        self.assertEqual(conceitos, ["Variaveis", "Listas", "Funcoes"])

    def test_json_invalido_depois_valido(self):
        client = MagicMock()
        client.models.generate_content.side_effect = [
            _fake_text_response("isso nao e json"),
            _fake_text_response('["A", "B"]'),
        ]
        conceitos = gwe.gerar_conceitos(client, "modelo", "Python", 2)
        self.assertEqual(conceitos, ["A", "B"])
        self.assertEqual(client.models.generate_content.call_count, 2)

    def test_json_invalido_persistente_levanta_erro(self):
        client = MagicMock()
        client.models.generate_content.return_value = _fake_text_response("nunca vira json")
        with self.assertRaises(RuntimeError):
            gwe.gerar_conceitos(client, "modelo", "Python", 2)


# ---------------------------------------------------------------------------
# gerar_worked_example (Etapa 4)
# ---------------------------------------------------------------------------


class TestGerarWorkedExample(unittest.TestCase):
    def test_dry_run(self):
        verbo = gwe.VerboBloom("Analisar sintaticamente", "Analisar")
        texto = gwe.gerar_worked_example(None, "modelo", "Python", "Listas", verbo, "contexto", dry_run=True)
        self.assertIn("Listas", texto)
        self.assertIn("Analisar sintaticamente", texto)

    def test_chamada_real_mockada_inclui_contexto_bloom(self):
        client = MagicMock()
        client.models.generate_content.return_value = _fake_text_response("worked example de teste")
        verbo = gwe.VerboBloom("Aplicar", "Aplicar")
        texto = gwe.gerar_worked_example(client, "modelo", "Python", "Listas", verbo, "contexto bloom xyz")
        self.assertEqual(texto, "worked example de teste")
        _, kwargs = client.models.generate_content.call_args
        self.assertEqual(kwargs["config"].system_instruction, "Voce e um especialista em design instrucional e na Taxonomia de Bloom Revisada, elaborando material didatico. Abaixo esta o arquivo de referencia com os verbos da Taxonomia de Bloom, organizados por categoria cognitiva - use-o como contexto para compreender o nivel cognitivo correto do verbo solicitado em cada exercicio:\n\ncontexto bloom xyz")


# ---------------------------------------------------------------------------
# main() - fluxo completo
# ---------------------------------------------------------------------------


class TestMainDryRun(unittest.TestCase):
    """Fluxo completo em --dry-run: nao requer GEMINI_API_KEY nem rede."""

    def setUp(self):
        self.tmpdir = Path(tempfile.mkdtemp())
        self._orig_output_dir = gwe.OUTPUT_DIR
        gwe.OUTPUT_DIR = self.tmpdir / "worked_examples"

    def tearDown(self):
        gwe.OUTPUT_DIR = self._orig_output_dir
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_gera_arquivos_esperados(self):
        codigo = gwe.main(["--C", "2", "--N", "3", "--dry-run", "--seed", "42"])
        self.assertEqual(codigo, 0)
        arquivos = sorted(p.name for p in gwe.OUTPUT_DIR.glob("*.txt"))
        self.assertEqual(len(arquivos), 6)  # 2 conceitos x 3 exemplos
        for nome in arquivos:
            self.assertTrue(nome.startswith("worked_example_conceito_de_exemplo_"))

    def test_C_ou_N_invalidos_sao_rejeitados(self):
        with self.assertRaises(SystemExit):
            gwe.main(["--C", "0", "--N", "2", "--dry-run"])


class TestMainComIaMockada(unittest.TestCase):
    """Fluxo completo (sem --dry-run) com o cliente mockado."""

    def setUp(self):
        self.tmpdir = Path(tempfile.mkdtemp())
        self._orig_output_dir = gwe.OUTPUT_DIR
        gwe.OUTPUT_DIR = self.tmpdir / "worked_examples"

    def tearDown(self):
        gwe.OUTPUT_DIR = self._orig_output_dir
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    @patch.dict("os.environ", {"GEMINI_API_KEY": "chave-falsa-para-teste"})
    @patch("generate_worked_examples.genai.Client")
    def test_fluxo_completo_com_client_mockado(self, mock_genai_cls):
        mock_client = MagicMock()
        respostas = [_fake_text_response('["Variaveis", "Listas"]')] + [
            _fake_text_response(f"worked example {i}") for i in range(1, 5)
        ]
        mock_client.models.generate_content.side_effect = respostas
        mock_genai_cls.return_value = mock_client

        codigo = gwe.main(["--C", "2", "--N", "2", "--seed", "1"])

        self.assertEqual(codigo, 0)
        arquivos = sorted(p.name for p in gwe.OUTPUT_DIR.glob("*.txt"))
        self.assertEqual(len(arquivos), 4)
        self.assertTrue(any("variaveis" in n for n in arquivos))
        self.assertTrue(any("listas" in n for n in arquivos))

    def test_sem_api_key_encerra_com_erro(self):
        with patch.dict("os.environ", {}, clear=True):
            with self.assertRaises(SystemExit):
                gwe.main(["--C", "1", "--N", "1"])


if __name__ == "__main__":
    unittest.main()
