from ragas.metrics import (AnswerRelevancy,
                           Faithfulness,
                           ResponseGroundedness,
                           AspectCritic,
                           LLMContextPrecisionWithoutReference,
                           LLMContextRecall)
from deepeval.metrics import (AnswerRelevancyMetric,
                              FaithfulnessMetric,
                              HallucinationMetric,
                              BiasMetric,
                              ToxicityMetric,
                              ContextualRelevancyMetric,
                              ContextualPrecisionMetric,
                              ContextualRecallMetric)
import json
import time
from plotter import Plotter, PlotterType
import numpy as np
from langchain_openai import AzureOpenAIEmbeddings
from langchain_openai.chat_models import AzureChatOpenAI
from ragas.dataset_schema import SingleTurnSample
from deepeval import evaluate
from deepeval.test_case import LLMTestCase
import os
from dotenv import load_dotenv
load_dotenv(".env", override=True)


### ENV VARS ###

openai_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
openai_api_version = os.getenv("AZURE_OPENAI_API_VERSION")
openai_api_key = os.getenv("AZURE_OPENAI_KEY")

chat_deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT")
embedding_deployment = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT")

document_endpoint = os.getenv("DOCUMENT_ENDPOINT")
document_api_key = os.getenv("DOCUMENT_API_KEY")

search_endpoint = os.getenv("AZURE_AI_SEARCH_ENDPOINT")
search_api_key = os.getenv("AZURE_AI_SEARCH_KEY")
index_name = os.getenv("AZURE_AI_SEARCH_INDEX")
semant_config_name = os.getenv("AZURE_AI_SEARCH_SEMANTIC_CONFIGURATION_NAME")


### UTIL FUNCTIONS ###

def get_chatopenai_client():
    """
    Returns an instance of the AzureChatOpenAI client.
    """

    return AzureChatOpenAI(
        azure_endpoint=openai_endpoint,
        api_version=openai_api_version,
        api_key=openai_api_key,
        azure_deployment=chat_deployment
    )


def get_embeddings_client():
    """
    Returns an instance of the AzureOpenAIEmbeddings client.
    """

    return AzureOpenAIEmbeddings(
        azure_endpoint=openai_endpoint,
        openai_api_version=openai_api_version,
        api_key=openai_api_key,
        model=embedding_deployment
    )

### DEEPEVAL SETUP ###


os.environ["AZURE_OPENAI_API_KEY"] = os.getenv("AZURE_OPENAI_API_KEY")

deepeval_root = "eval"

exp_path = r"D:\Programmi\Python\rag_azure\test_script.py"
exp_name = os.path.basename(exp_path).replace(".json", "")

result_root = os.path.join(deepeval_root, exp_name)
os.makedirs(result_root, exist_ok=True)

result_path = os.path.join(result_root, exp_name)

### DEEPEVAL METRICS ###

deepeval_threshold = 0.5

answer_relevancy = AnswerRelevancyMetric(threshold=deepeval_threshold)
faithfulness = FaithfulnessMetric(threshold=deepeval_threshold)
hallucination = HallucinationMetric(threshold=deepeval_threshold)
bias = BiasMetric(threshold=deepeval_threshold)
toxicity = ToxicityMetric(threshold=deepeval_threshold)

contextual_relevancy = ContextualRelevancyMetric(threshold=deepeval_threshold)
contextual_precision = ContextualPrecisionMetric(threshold=deepeval_threshold)
contextual_recall = ContextualRecallMetric(threshold=deepeval_threshold)

deepeval_metrics = [answer_relevancy, faithfulness, hallucination,
                    contextual_relevancy, contextual_precision, contextual_recall]

### RAGAS METRICS ###

ragas_threshold = 0.5

answer_relevancy = AnswerRelevancy(
    llm=get_chatopenai_client(), embeddings=get_embeddings_client())
faithfulness = Faithfulness(llm=get_chatopenai_client())
groundedness = ResponseGroundedness(llm=get_chatopenai_client())
cheating_critic = AspectCritic(
    name="cheating",
    definition="Is the submission somehow intended to ...",
    llm=get_chatopenai_client()
)

contextual_precision = LLMContextPrecisionWithoutReference(
    llm=get_chatopenai_client())
contextual_recall = LLMContextRecall(llm=get_chatopenai_client())

ragas_metrics = []


with open(exp_path, "r", encoding="utf-8") as f:
    data = json.load(f)

metrics_results = {}

for idx, question in enumerate(data):
    print(f"QUESTION {idx+1}:")
    print('\n' * 10)

    ### DEEPEVAL TURN ###
    deepeval_test_case = LLMTestCase(
        name=f"question_{idx+1}",
        input=question["question"],
        # Qui usiamo expected_answer come output reale
        actual_output=question.get("expected_answer", ""),
        # Placeholder vuoto perché non disponibile nel JSON
        retrieval_context="",
        # Placeholder vuoto per metriche di allucinazione
        context="",
        expected_output=question.get("expected_answer", "")
    )

    deepeval_result = None

    while deepeval_result is None:
        try:
            deepeval_result = evaluate(
                [deepeval_test_case], metrics=deepeval_metrics).model_dump()["test_results"][0]
        except:
            print("Deepeval evaluation failed.")

            print("\nTaking a 10s nap...\n")
            # because of the too many (parallel) requests per minute to OpenAI while using a free tier sub
            time.sleep(10)

    deepeval_summary = {m["name"]: {"success": m["success"], "result": {
        "score": m["score"], "reason": m["reason"]}} for m in deepeval_result['metrics_data']}

    ### RAGAS TURN ###
    ragas_sample = SingleTurnSample(
        user_input=question["question"],
        response=question["answer"],
        # reference=question["answer"],
        retrieved_contexts=question["retrieved_content"],
    )

    ragas_summary = {}

    for metric in ragas_metrics:
        score = None
        while score is None:
            try:
                score = metric.single_turn_score(ragas_sample)
            except:
                print("Ragas evaluation failed. Retrying...")

        success = score >= ragas_threshold
        ragas_summary[metric.name] = {
            "success": success, "result": {"score": score}}

    print("\n+++++++++++++++++++ MY SUMMARY +++++++++++++++++++")

    for metric, data in deepeval_summary.items():
        if not data["success"]:
            print(f"\n❌ deepeval {metric.capitalize()}:\n")

            for k, v in data["result"].items():
                print(f"{k.capitalize()}: {v}")

    for metric, data in ragas_summary.items():
        if not data["success"]:
            print(f"\n❌ ragas {metric.capitalize()}:\n")

            for k, v in data["result"].items():
                print(f"{k.capitalize()}: {v}")

    print("+++++++++++++++++++ MY SUMMARY +++++++++++++++++++\n")

    if not metrics_results:
        metrics_results = {metric.capitalize(
        ): [data["result"]["score"]] for metric, data in deepeval_summary.items()}
    else:
        for metric, data in deepeval_summary.items():
            metrics_results[metric.capitalize()].append(
                data["result"]["score"])

    print('\n' * 10)


# SAVE DEEPEVAL STATISTICS

with open(result_path+".json", "w") as f:
    global_results = {metric: {"mean": np.nanmean(np.array(scores)), "std": np.nanstd(
        np.array(scores)), "scores": scores} for metric, scores in metrics_results.items()}

    json.dump(global_results, f, indent=2)


# PLOT DEEPEVAL STATISTICS

# Creazione dell'istanza di Plotter (linee, puoi cambiare in BAR, SCATTER ecc.)
plotter = Plotter(PlotterType.LINE)

# Caricamento dei risultati dal file JSON
with open(result_path + ".json", "r") as f:
    results_data = json.load(f)

# Preparazione dei dati per il plotting
metrics = list(results_data.keys())
scores = [results_data[m]["mean"] for m in metrics]

# Creazione del grafico
plotter.plot(
    x=metrics,
    y=scores,
    title="DEEPEVAL Metrics",
    xlabel="Metric",
    ylabel="Score",
    save_path=result_path + "_plot.png"  # salva figura come PNG
)

print(f"✅ Grafico salvato in {result_path}_plot.png")
