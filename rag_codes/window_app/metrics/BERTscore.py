import argparse
import json
from bert_score import score

def compute_bertscore(query, chunks, lang="pl"):
    """
    Compute BERTScore (precision, recall, F1) between a query and each chunk.
    Returns a list of dicts with metrics for each chunk.
    """
    # Candidates: chunks; References: query repeated
    refs = [query] * len(chunks)
    cands = chunks

    # Compute scores
    P, R, F1 = score(cands, refs, lang=lang, verbose=False, rescale_with_baseline=True)

    # Convert tensors to floats and assemble results
    results = []
    for chunk, p, r, f1 in zip(chunks, P.tolist(), R.tolist(), F1.tolist()):
        results.append({
            "chunk": chunk,
            "precision": p,
            "recall": r,
            "f1": f1
        })
    return results

def main():
    parser = argparse.ArgumentParser(
        description="Compute BERTScore between a query and a list of text chunks, output results to JSON"
    )
    parser.add_argument(
        "--query", "-q",
        type=str,
        required=True,
        help="The query text to compare against chunks."
    )
    parser.add_argument(
        "--chunks-file", "-c",
        type=str,
        required=True,
        help="Path to a JSON file containing {'chunks': [chunk1, chunk2, ...]}."
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="bertscore_results.json",
        help="Output JSON file path (default: bertscore_results.json)."
    )
    parser.add_argument(
        "--lang", "-l",
        type=str,
        default="pl",
        help="Language code for BERTScore (default: pl)."
    )

    args = parser.parse_args()

    # Load chunks
    with open(args.chunks_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
        chunks = data.get('chunks', [])
        if not isinstance(chunks, list):
            raise ValueError("The JSON file must contain a list under the key 'chunks'.")

    # Compute BERTScore metrics
    metrics = compute_bertscore(args.query, chunks, lang=args.lang)

    # Prepare output
    output_data = {
        "query": args.query,
        "results": metrics
    }

    # Write to JSON file
    with open(args.output, 'w', encoding='utf-8') as out_f:
        json.dump(output_data, out_f, ensure_ascii=False, indent=2)

    print(f"Saved BERTScore results to {args.output}")

if __name__ == '__main__':
    main()
