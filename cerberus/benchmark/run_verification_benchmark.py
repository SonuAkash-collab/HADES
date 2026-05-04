from __future__ import annotations

from cerberus.core import build_source_graph, verify_claim
from shared.extractor import extract_claim_triples, extract_source_triples


DATASET = {
    "source_text": (
        "The Apollo 11 mission launched on a Saturn V rocket. "
        "Neil Armstrong and Buzz Aldrin descended to the lunar surface in the Eagle module. "
        "Michael Collins remained in lunar orbit aboard the Command Module Columbia. "
        "The landing site was the Sea of Tranquility."
    )
}


CLAIMS = [
    {
        "text": "Neil Armstrong and Buzz Aldrin descended to the lunar surface in the Eagle module.",
        "is_true": True,
    },
    {
        "text": "Michael Collins remained in lunar orbit aboard the Command Module Columbia.",
        "is_true": True,
    },
    {
        "text": "Apollo 11 launched on a Saturn V rocket.",
        "is_true": True,
    },
    {
        "text": "The Soviet Union landed on the moon during Apollo 11.",
        "is_true": False,
    },
    {
        "text": "Michael Collins walked on the moon.",
        "is_true": False,
    },
    {
        "text": "Apollo 11 launched from a moon base.",
        "is_true": False,
    },
]


def main() -> int:
    source_text = DATASET["source_text"]
    source_triples = extract_source_triples(source_text)
    source_graph = build_source_graph(source_triples)

    tp = 0
    tn = 0
    fp = 0
    fn = 0

    print("CERBERUS VERIFICATION BENCHMARK")
    print("=" * 90)
    print(f"Source triples extracted: {len(source_triples)}")
    print(f"Claims evaluated: {len(CLAIMS)}")
    print("=" * 90)

    for index, claim in enumerate(CLAIMS, start=1):
        claim_text = str(claim["text"])
        expected_true = bool(claim["is_true"])

        claim_triples = extract_claim_triples(claim_text)
        if claim_triples:
            triple = claim_triples[0]
            result = verify_claim(triple, source_graph)
            predicted_true = result.is_verified
            triple_text = triple.as_text()
            reason = result.reason
            label = result.label
        else:
            predicted_true = False
            triple_text = "<no triple extracted>"
            reason = "No claim triple extracted; treated as not verified."
            label = "none"

        if expected_true and predicted_true:
            tp += 1
            bucket = "TP"
        elif (not expected_true) and (not predicted_true):
            tn += 1
            bucket = "TN"
        elif (not expected_true) and predicted_true:
            fp += 1
            bucket = "FP"
        else:
            fn += 1
            bucket = "FN"

        print(f"[{index}] {bucket} | expected={expected_true} predicted={predicted_true}")
        print(f"    claim:  {claim_text}")
        print(f"    triple: {triple_text}")
        print(f"    label:  {label}")
        print(f"    reason: {reason}")

    total = tp + tn + fp + fn
    accuracy = ((tp + tn) / total) if total else 0.0
    precision = (tp / (tp + fp)) if (tp + fp) else 0.0
    recall = (tp / (tp + fn)) if (tp + fn) else 0.0

    print("=" * 90)
    print("CONFUSION COUNTS")
    print(f"TP={tp} TN={tn} FP={fp} FN={fn}")
    print("METRICS")
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print("=" * 90)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
