# Usage example
from metric.spice import SPICE


if __name__ == "__main__":
    spice = SPICE("pt", "own-pt:1.0.0")

    candidate = "Um gato preto está sentado em um tapete vermelho."
    references = [
        "Um felino escuro descansa sobre um carpete colorido.",
        "Há um gato de cor escura em um tapete.",
    ]

    # candidate = "A black cat is sitting on a red carpet."
    # references = [
    #     "A dark-colored feline rests on a colorful carpet.",
    #     "There is a dark-colored cat on a carpet.",
    # ]

    results = spice.compute_score(candidate, references)
    print(f"SPICE F1: {results['f1']:.4f}")
    print(f"Precision: {results['precision']:.4f}")
    print(f"Recall: {results['recall']:.4f}")
