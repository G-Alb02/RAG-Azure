import json
import matplotlib.pyplot as plt
import numpy as np


def plot_metrics_from_json(file_path: str):
    """
    Legge un file JSON contenente i risultati DeepEval e genera un grafico
    dei punteggi delle metriche (Answer Relevancy, Faithfulness, Correctness).

    Parametri:
        file_path (str): percorso del file JSON contenente i risultati.
    """

    # Carica i dati dal file JSON
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            results = json.load(f)
    except Exception as e:
        print(f"❌ Errore durante la lettura del file '{file_path}': {e}")
        return

    # Verifica formato corretto
    if not isinstance(results, list) or len(results) == 0:
        print("⚠️ Il file JSON non contiene una lista valida di risultati.")
        return

    # Estrai metriche dai risultati
    labels = [f"Test {i+1}" for i in range(len(results))]
    relevancy_scores = [r.get("answer_relevancy_score", 0) for r in results]
    faithfulness_scores = [r.get("faithfulness_score", 0) for r in results]
    correctness_scores = [r.get("correctness_score", 0) for r in results]

    # Funzione per colorare le barre in base al punteggio
    def colorize(scores):
        return [
            "green" if s >= 0.8 else
            "gold" if s >= 0.6 else
            "red"
            for s in scores
        ]

    # Calcolo delle medie
    avg_relevancy = np.mean(relevancy_scores)
    avg_faithfulness = np.mean(faithfulness_scores)
    avg_correctness = np.mean(correctness_scores)

    # Creazione del grafico
    plt.figure(figsize=(14, 7))
    x = np.arange(len(results))
    width = 0.25

    plt.bar(x - width, relevancy_scores, width, color=colorize(relevancy_scores),
            label=f"Answer Relevancy (avg={avg_relevancy:.2f})")
    plt.bar(x, faithfulness_scores, width, color=colorize(
        faithfulness_scores), label=f"Faithfulness (avg={avg_faithfulness:.2f})")
    plt.bar(x + width, correctness_scores, width, color=colorize(correctness_scores),
            label=f"Correctness (avg={avg_correctness:.2f})")

    plt.xlabel("Test Cases", fontsize=12)
    plt.ylabel("Score (0 - 1)", fontsize=12)
    plt.title("DeepEval Metrics Overview", fontsize=14)
    plt.xticks(x, labels, rotation=45, ha="right")
    plt.ylim(0, 1.1)
    plt.legend()
    plt.tight_layout()

    plt.show()

    # Stampa i valori medi anche in console
    print("\n📊  METRICHE MEDIE")
    print(f"Answer Relevancy: {avg_relevancy:.3f}")
    print(f"Faithfulness:     {avg_faithfulness:.3f}")
    print(f"Correctness:      {avg_correctness:.3f}")


def plot_metrics_average(file_path: str, threshold: float = 0.7):
    """
    Legge un file JSON contenente i risultati DeepEval e genera un grafico a barre con le medie
    delle metriche (Answer Relevancy, Faithfulness, Correctness), con una soglia visiva.

    Parametri:
        file_path (str): percorso del file JSON contenente i risultati.
        threshold (float): soglia (0-1) da mostrare come riferimento nel grafico.
    """

    # Carica dati dal file JSON
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            results = json.load(f)
    except Exception as e:
        print(f"❌ Errore durante la lettura del file '{file_path}': {e}")
        return

    # Verifica che sia lista e non vuota
    if not isinstance(results, list) or len(results) == 0:
        print("⚠️ Il file JSON non contiene una lista valida di risultati.")
        return

    # Estrai i punteggi
    relevancy_scores = [r.get("answer_relevancy_score", 0) for r in results]
    faithfulness_scores = [r.get("faithfulness_score", 0) for r in results]
    correctness_scores = [r.get("correctness_score", 0) for r in results]

    # Calcola medie
    avg_relevancy = np.mean(relevancy_scores)
    avg_faithfulness = np.mean(faithfulness_scores)
    avg_correctness = np.mean(correctness_scores)

    # Labels e valori
    metrics = ["Answer Relevancy", "Faithfulness", "Correctness"]
    averages = [avg_relevancy, avg_faithfulness, avg_correctness]

    # Colori in base soglia
    colors = ["green" if val >= threshold else "red" for val in averages]

    # Creazione del grafico
    plt.figure(figsize=(8, 5))
    bars = plt.bar(metrics, averages, color=colors, alpha=0.8)

    # Linea soglia
    plt.axhline(y=threshold, color='blue', linestyle='--',
                label=f"Soglia {threshold*100:.0f}%")

    # Annotazioni sopra le barre
    for bar, val in zip(bars, averages):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, height + 0.02, f"{val:.2f}",
                 ha='center', va='bottom', fontsize=11)

    plt.ylim(0, 1.0)
    plt.ylabel("Punteggio medio (0-1)")
    plt.title("Medie delle metriche DeepEval")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Stampa media in console
    print("\n📊 MEDIE METRICHE")
    print(f"Answer Relevancy: {avg_relevancy:.3f}")
    print(f"Faithfulness:     {avg_faithfulness:.3f}")
    print(f"Correctness:      {avg_correctness:.3f}")


if __name__ == "__main__":
    # Inserisci qui il percorso del tuo file JSON di risultati
    json_path = r"evaluation_results_2025-10-22.json"
    plot_metrics_average(json_path)
