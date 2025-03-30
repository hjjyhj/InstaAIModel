import json
from collections import defaultdict

# Hardcoded pipeline results (example)
pipeline_top_k = {
    "Food & Cooking": [
        "twogoodtoresist", "chefdelicious", "mindyscookingobsession", "gourmetqueen", "melbournemeals"
    ],
    "Technology": [
        "girlxdeparture", "techguru", "tripinaomi", "ai_innovator", "goldenriley_"
    ],
    # Add more categories as needed
}

K = 5  # Top-k influencers to evaluate
GROUND_TRUTH_PATH = "/scratch/eecs487w25_class_root/eecs487w25_class/shared_data/johnkimm_dir/influencers_by_category.json"

# NOTE: recall won't be really useful here since we're comparing against ALL ground truth per category
def precision_recall_f1_at_k(ground_truth, predictions, k=5):
    results_by_category = {}
    total_tp = total_fp = total_fn = 0

    for category, pred_users in predictions.items():
        gt_users = set(ground_truth.get(category, []))
        pred_users_k = pred_users[:k]

        tp = len([user for user in pred_users_k if user in gt_users])
        fp = len([user for user in pred_users_k if user not in gt_users])
        fn = len(gt_users - set(pred_users_k))

        precision = tp / k if k else 0
        recall = tp / len(gt_users) if gt_users else 0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0

        results_by_category[category] = {
            "precision@k": round(precision, 4),
            "recall@k": round(recall, 4),
            "f1@k": round(f1, 4)
        }

        total_tp += tp
        total_fp += fp
        total_fn += fn

    overall_precision = total_tp / (K * len(predictions)) if predictions else 0
    overall_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) else 0
    overall_f1 = (2 * overall_precision * overall_recall / (overall_precision + overall_recall)) if (overall_precision + overall_recall) else 0

    results_by_category["OVERALL"] = {
        "precision@k": round(overall_precision, 4),
        "recall@k": round(overall_recall, 4),
        "f1@k": round(overall_f1, 4)
    }

    return results_by_category

def main():
    with open(GROUND_TRUTH_PATH, 'r', encoding='utf-8') as f:
        ground_truth = json.load(f)

    results = precision_recall_f1_at_k(ground_truth, pipeline_top_k, k=K)

    print("Evaluation results:")
    for category, metrics in results.items():
        print(f"\n{category}")
        for metric, value in metrics.items():
            print(f"  {metric}: {value}")

if __name__ == "__main__":
    main()
