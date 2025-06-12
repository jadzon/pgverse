import json
import matplotlib.pyplot as plt

def load_scores(path="rag_metrics/bertscore_results.json"):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def extract_points(data):
    questions = [entry['query'] for entry in data]
    precision_pts, recall_pts, f1_pts = [], [], []

    # zbieramy wszystkie punkty
    for idx, entry in enumerate(data):
        for score in entry['scores']:
            precision_pts.append((idx, score['precision']))
            recall_pts.append((idx, score['recall']))
            f1_pts.append((idx, score['f1']))

    def compute_stats(pts):
        # globalna średnia
        ys = [y for _, y in pts]
        mean_all = sum(ys) / len(ys)
        # dla każdego pytania bierzemy maksimum
        local_maxima = []
        for i in range(len(questions)):
            vals = [y for x, y in pts if x == i]
            if vals:
                local_maxima.append(max(vals))
        # średnia tych lokalnych maksimów
        mean_local_max = sum(local_maxima) / len(local_maxima)
        return {'mean': mean_all, 'local_max_mean': mean_local_max}

    stats = {
        'precision': compute_stats(precision_pts),
        'recall':    compute_stats(recall_pts),
        'f1':        compute_stats(f1_pts),
    }

    return questions, precision_pts, recall_pts, f1_pts, stats

def plot_all_in_one(questions, p_pts, r_pts, f1_pts, stats):
    fig, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
    metrics = [('precision', p_pts), ('recall', r_pts), ('f1', f1_pts)]

    for ax, (name, pts) in zip(axes, metrics):
        x, y = zip(*pts)
        ax.scatter(x, y, label='Referencje')
        ax.axhline(stats[name]['mean'],
                   color='red',
                   linestyle='--',
                   label='Średnia globalna')
        ax.axhline(stats[name]['local_max_mean'],
                   color='green',
                   linestyle='--',
                   label='Średnia maksimów lokalnych')
        ax.set_ylabel(name.capitalize())
        ax.set_title(f'{name.capitalize()} dla wszystkich referencji')
        ax.legend()

    # wspólna oś X
    axes[-1].set_xticks(range(len(questions)))
    axes[-1].set_xticklabels(questions, rotation=45, ha='right')
    axes[-1].set_xlabel('Pytania')

    plt.tight_layout()
    plt.show()

def main():
    data = load_scores()
    questions, p_pts, r_pts, f1_pts, stats = extract_points(data)
    plot_all_in_one(questions, p_pts, r_pts, f1_pts, stats)

if __name__ == "__main__":
    main()
