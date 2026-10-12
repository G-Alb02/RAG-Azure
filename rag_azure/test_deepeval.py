import os
import json
import datetime
from dotenv import load_dotenv
from openai import AzureOpenAI
from deepeval.test_case import LLMTestCase, LLMTestCaseParams
from deepeval.metrics import AnswerRelevancyMetric, FaithfulnessMetric, GEval
from deepeval.dataset import EvaluationDataset


# === 1. Carica variabili da .env ===
load_dotenv()

AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT")
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION")

# === 2. Inizializza client Azure OpenAI ===
client = AzureOpenAI(
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    api_key=AZURE_OPENAI_API_KEY,
    api_version=AZURE_OPENAI_API_VERSION,
)

# === 3. Funzione helper per generare risposta dal modello Azure ===


def call_chatbot(question: str, retrieval_context: list[str]) -> str:
    """Genera una risposta usando il modello Azure OpenAI"""
    context_text = "\n".join(retrieval_context)
    prompt = (
        f"Context:\n{context_text}\n\n"
        f"Question: {question}\n"
        "Answer clearly and accurately, using only the provided context."
    )

    try:
        response = client.chat.completions.create(
            model=AZURE_OPENAI_DEPLOYMENT,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=500,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"[ERROR] Azure OpenAI call failed: {e}")
        return ""


# === 4. Carica i test case dal file JSON ===
with open("test_cases.json", "r", encoding="utf-8") as f:
    raw_cases = json.load(f)

test_cases = []
for item in raw_cases:
    # Pulisce eventuali campi estranei (per evitare bug con multi-turn)
    item_clean = {
        "input": item.get("input", ""),
        "retrieval_context": item.get("retrieval_context", []),
        "actual_output": item.get("actual_output", ""),
        "expected_output": item.get("expected_output", None)
    }

    # Genera la risposta se manca
    if not item_clean["actual_output"]:
        print(
            f"[INFO] Genero risposta per input: {item_clean['input'][:60]}...")
        item_clean["actual_output"] = call_chatbot(
            item_clean["input"], item_clean["retrieval_context"]
        )

    # Crea test case single-turn
    test_case = LLMTestCase(
        input=item_clean["input"],
        actual_output=item_clean["actual_output"],
        retrieval_context=item_clean["retrieval_context"],
        expected_output=item_clean["expected_output"]
    )
    test_cases.append(test_case)

# === 5. Definisci metriche ===
answer_relevancy = AnswerRelevancyMetric(threshold=0.6, model="gpt-4o")
faithfulness = FaithfulnessMetric(threshold=0.7, model="gpt-4o")
correctness = GEval(
    name="Correctness",
    model="gpt-4o",
    evaluation_params=[
        LLMTestCaseParams.ACTUAL_OUTPUT,
        LLMTestCaseParams.EXPECTED_OUTPUT
    ],
    criteria="Check if the actual output matches the expected output accurately."
)

# === 6. Esegui valutazione singola per ogni test case ===
results = []

for tc in test_cases:
    print(f"[INFO] Valuto test case: {tc.input[:60]}...")

    try:
        ar_score = answer_relevancy.measure(tc)
        f_score = faithfulness.measure(tc)
        c_score = correctness.measure(tc)

        results.append({
            "input": tc.input,
            "actual_output": tc.actual_output,
            "answer_relevancy_score": ar_score,
            "faithfulness_score": f_score,
            "correctness_score": c_score
        })
    except Exception as e:
        print(
            f"[ERROR] Valutazione fallita per test case: {tc.input[:60]} – {e}")

# === 7. Salva risultati ===
output_file = f"evaluation_results_{datetime.date.today()}.json"
with open(output_file, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

print(f"\n✅ Valutazione completata. Risultati salvati in: {output_file}")

# === 8. Stampa riepilogo sintetico ===
for r in results:
    print(f"\n--- Test Case ---")
    print(f"Input: {r['input']}")
    print(f"Answer Relevancy: {r['answer_relevancy_score']:.3f}")
    print(f"Faithfulness: {r['faithfulness_score']:.3f}")
    print(f"Correctness: {r['correctness_score']:.3f}")
