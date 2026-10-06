import os
import numpy as np
import matplotlib.pyplot as plt

import random

RANDOM_SEED = 42
random.seed(RANDOM_SEED)

from cbir_skimage import charger_images_dossier, IndexeurCBIR


# ---------------------------------------------------------
# Utility functions
# ---------------------------------------------------------

def get_class_from_path(file_path):
    """
    Extract BUSI class from the relative file path.
    Expected classes: benign, malignant, normal.
    """
    path = file_path.lower().replace("\\", "/")

    for class_name in ("benign", "malignant", "normal"):
        if f"/{class_name}/" in path:
            return class_name

    return "unknown"


def print_query_results(
    query_name,
    query_class,
    file_names,
    indices,
    distances
):
    """
    Print the query and its retrieved images.
    """
    print("\n" + "=" * 70)
    print(f"QUERY: {query_name}")
    print(f"CLASS: {query_class}")
    print("=" * 70)

    for rank, (idx, distance) in enumerate(
        zip(indices, distances),
        start=1
    ):

        result_class = get_class_from_path(
            file_names[idx]
        )

        print(
            f"Top {rank}: "
            f"{file_names[idx]} | "
            f"class={result_class} | "
            f"distance={distance:.4f}"
        )


def precision_at_k(
    query_class,
    file_names,
    indices,
    k
):
    """
    Precision@K based on the BUSI class.

    An image is considered relevant when it belongs
    to the same class as the query.
    """

    if len(indices) == 0:
        return 0.0

    selected_indices = indices[:k]

    relevant = 0

    for idx in selected_indices:

        result_class = get_class_from_path(
            file_names[idx]
        )

        if result_class == query_class:
            relevant += 1

    return relevant / len(selected_indices)


def save_metrics_summary(
    results,
    mean_p1,
    mean_p5,
    class_results,
    database_size,
    query_size
):
    """
    Save a concise and professional CBIR evaluation summary.
    """

    reports_dir = "../reports"

    os.makedirs(
        reports_dir,
        exist_ok=True
    )

    output_file = os.path.join(
        reports_dir,
        "metrics_summary.txt"
    )

    with open(
        output_file,
        "w",
        encoding="utf-8"
    ) as f:

        f.write(
            "BUSI MEDICAL CBIR EVALUATION\n"
        )

        f.write(
            "=" * 60 + "\n\n"
        )

        # -------------------------------------------------
        # Evaluation setup
        # -------------------------------------------------

        f.write(
            "EVALUATION SETUP\n"
        )

        f.write(
            "-" * 60 + "\n"
        )

        f.write(
            f"Total BUSI images: {database_size + query_size}\n"
        )

        f.write(
            f"Retrieval database: {database_size} images\n"
        )

        f.write(
            f"Query set: {query_size} images\n"
        )

        f.write(
            f"Queries per class: {QUERY_COUNT_PER_CLASS}\n"
        )

        f.write(
            "Classes: benign, malignant, normal\n"
        )

        f.write(
            "Query images excluded from retrieval database: Yes\n"
        )

        f.write(
            "Feature scaler fitted only on retrieval database: Yes\n"
        )

        f.write(
            "Feature descriptors: intensity histogram, GLCM texture, Hu moments\n"
        )

        f.write(
            "Distance metric: Euclidean distance\n"
        )

        f.write(
            "Retrieval size: Top-5\n"
        )

        f.write(
            "Relevance criterion: same BUSI class as the query\n\n"
        )

        # -------------------------------------------------
        # Global results
        # -------------------------------------------------

        f.write(
            "GLOBAL RESULTS\n"
        )

        f.write(
            "-" * 60 + "\n"
        )

        f.write(
            f"Mean Precision@1: {mean_p1:.4f}\n"
        )

        f.write(
            f"Mean Precision@5: {mean_p5:.4f}\n\n"
        )

        # -------------------------------------------------
        # Results by class
        # -------------------------------------------------

        f.write(
            "RESULTS BY CLASS\n"
        )

        f.write(
            "-" * 60 + "\n"
        )

        for class_name, values in class_results.items():

            if len(values) == 0:
                continue

            p1 = np.mean(
                [
                    v["precision_at_1"]
                    for v in values
                ]
            )

            p5 = np.mean(
                [
                    v["precision_at_5"]
                    for v in values
                ]
            )

            f.write(
                f"{class_name.capitalize():<12} "
                f"P@1={p1:.4f} | "
                f"P@5={p5:.4f} | "
                f"Queries={len(values)}\n"
            )

        f.write("\n")

        # -------------------------------------------------
        # Interpretation
        # -------------------------------------------------

        f.write(
            "INTERPRETATION\n"
        )

        f.write(
            "-" * 60 + "\n"
        )

        f.write(
            "Precision@1 measures whether the first retrieved image "
            "belongs to the same BUSI class as the query.\n"
        )

        f.write(
            "Precision@5 measures the proportion of the five retrieved "
            "images that belong to the same BUSI class as the query.\n"
        )

        f.write(
            "These metrics evaluate image retrieval consistency, "
            "not clinical diagnosis or cancer classification.\n"
        )

        f.write(
            "\n"
        )

        f.write(
            "Note: Results are based on classical image descriptors "
            "and Euclidean similarity and are intended for academic "
            "CBIR evaluation.\n"
        )

    print(
        f"\nMetrics summary saved to: {output_file}"
    )

# ---------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------

def main():

    print("=" * 70)
    print(
        "       BUSI MEDICAL CBIR EVALUATION"
    )
    print("=" * 70)

    # -----------------------------------------------------
    # 1. Dataset path
    # -----------------------------------------------------

    DATASET_DIR = "../data"

    if not os.path.exists(DATASET_DIR):

        DATASET_DIR = "./data"

    if not os.path.exists(DATASET_DIR):

        print(
            f"ERROR: Dataset folder not found: "
            f"{DATASET_DIR}"
        )

        return

    # -----------------------------------------------------
    # 2. Configuration
    # -----------------------------------------------------

    # None = load all available BUSI images
    NUM_IMAGES = None

    methods = (
        "couleur",
        "texture",
        "forme"
    )

    TOP_K = 5

    # Number of queries selected from each BUSI class
    QUERY_COUNT_PER_CLASS = 30

    # -----------------------------------------------------
    # 3. Load images
    # -----------------------------------------------------

    print("\nLoading BUSI images...")

    images, file_names = charger_images_dossier(
        DATASET_DIR,
        max_images=NUM_IMAGES
    )

    if len(images) < 10:

        print(
            f"ERROR: Only {len(images)} images "
            "were loaded."
        )

        return

    print(
        f"Images loaded: {len(images)}"
    )

    # -----------------------------------------------------
    # 4. Show class distribution
    # -----------------------------------------------------

    class_counts = {
        "benign": 0,
        "malignant": 0,
        "normal": 0,
        "unknown": 0
    }

    for file_name in file_names:

        class_name = get_class_from_path(
            file_name
        )

        class_counts[class_name] += 1

    print("\nClass distribution:")

    for class_name, count in class_counts.items():

        if count > 0:

            print(
                f"  {class_name}: {count}"
            )

    # -----------------------------------------------------
    # 5. Select balanced queries
    # -----------------------------------------------------

    QUERY_INDICES = []

    class_query_count = {
        "benign": QUERY_COUNT_PER_CLASS,
        "malignant": QUERY_COUNT_PER_CLASS,
        "normal": QUERY_COUNT_PER_CLASS
    }

    for class_name, count in class_query_count.items():

        class_indices = [
            i
            for i, path in enumerate(file_names)
            if get_class_from_path(path) == class_name
        ]

        if len(class_indices) < count:

            print(
                f"ERROR: Not enough images for "
                f"class '{class_name}'."
            )

            return

        selected_indices = random.sample(
            class_indices,
            count
        )

        QUERY_INDICES.extend(
            selected_indices
        )

    print("\nQuery selection:")

    for class_name in class_query_count:

        count = sum(
            1
            for i in QUERY_INDICES
            if get_class_from_path(
                file_names[i]
            ) == class_name
        )

        print(
            f"  {class_name}: {count} queries"
        )

    print(
        f"  Total queries: "
        f"{len(QUERY_INDICES)}"
    )

    # -----------------------------------------------------
    # 6. Build database
    # -----------------------------------------------------

    print(
        "\nBuilding retrieval database..."
    )

    query_set = set(
        QUERY_INDICES
    )

    database_indices = [
        i
        for i in range(len(images))
        if i not in query_set
    ]

    database_images = [
        images[i]
        for i in database_indices
    ]

    database_file_names = [
        file_names[i]
        for i in database_indices
    ]

    print(
        f"Database images: "
        f"{len(database_images)}"
    )

    print(
        f"Query images: "
        f"{len(QUERY_INDICES)}"
    )

    # Safety check
    if (
        len(database_images)
        + len(QUERY_INDICES)
        != len(images)
    ):

        print(
            "ERROR: Database/query split is invalid."
        )

        return

    # -----------------------------------------------------
    # 7. Index database
    # -----------------------------------------------------

    print(
        "\nExtracting features and indexing..."
    )

    indexer = IndexeurCBIR()

    indexer.indexer(
        database_images,
        database_file_names,
        methodes=methods
    )

    # -----------------------------------------------------
    # 8. Evaluation
    # -----------------------------------------------------

    results = []

    for query_number, query_index in enumerate(
        QUERY_INDICES,
        start=1
    ):

        if query_index >= len(images):

            continue

        # Original query image
        query_image = images[
            query_index
        ]

        # Original query filename
        query_name = file_names[
            query_index
        ]

        query_class = get_class_from_path(
            query_name
        )

        print(
            f"\nEvaluating query "
            f"{query_number}/"
            f"{len(QUERY_INDICES)}: "
            f"{query_name}"
        )

        # -------------------------------------------------
        # Search in DATABASE ONLY
        # -------------------------------------------------

        indices, distances = indexer.rechercher(
            query_image,
            top_k=TOP_K,
            metrique="euclidienne",
            methodes=methods
        )

        # -------------------------------------------------
        # Print retrieved images
        # -------------------------------------------------

        print_query_results(
            query_name,
            query_class,
            database_file_names,
            indices,
            distances
        )

        # -------------------------------------------------
        # Metrics
        # -------------------------------------------------

        p1 = precision_at_k(
            query_class,
            database_file_names,
            indices,
            1
        )

        p5 = precision_at_k(
            query_class,
            database_file_names,
            indices,
            5
        )

        print(
            f"\nPrecision@1 = "
            f"{p1:.2f}"
        )

        print(
            f"Precision@5 = "
            f"{p5:.2f}"
        )

        results.append(
            {
                "query_name": query_name,
                "query_class": query_class,
                "precision_at_1": p1,
                "precision_at_5": p5
            }
        )

    # -----------------------------------------------------
    # 9. Global metrics
    # -----------------------------------------------------

    if len(results) == 0:

        print(
            "\nERROR: No valid query results."
        )

        return

    mean_p1 = np.mean(
        [
            r["precision_at_1"]
            for r in results
        ]
    )

    mean_p5 = np.mean(
        [
            r["precision_at_5"]
            for r in results
        ]
    )

    print(
        "\n" + "=" * 70
    )

    print(
        "GLOBAL RESULTS"
    )

    print(
        "=" * 70
    )

    print(
        f"Mean Precision@1 = "
        f"{mean_p1:.4f}"
    )

    print(
        f"Mean Precision@5 = "
        f"{mean_p5:.4f}"
    )

    # -----------------------------------------------------
    # 10. Results by class
    # -----------------------------------------------------

    class_results = {
        "benign": [],
        "malignant": [],
        "normal": []
    }

    for result in results:

        query_class = result[
            "query_class"
        ]

        if query_class in class_results:

            class_results[
                query_class
            ].append(result)

    print(
        "\nRESULTS BY CLASS"
    )

    print(
        "-" * 50
    )

    class_p1 = {}
    class_p5 = {}

    for class_name, values in class_results.items():

        if len(values) == 0:

            continue

        p1 = np.mean(
            [
                v["precision_at_1"]
                for v in values
            ]
        )

        p5 = np.mean(
            [
                v["precision_at_5"]
                for v in values
            ]
        )

        class_p1[
            class_name
        ] = p1

        class_p5[
            class_name
        ] = p5

        print(
            f"{class_name}: "
            f"P@1={p1:.4f}, "
            f"P@5={p5:.4f}"
        )

    # -----------------------------------------------------
    # 11. Reports directory
    # -----------------------------------------------------

    reports_dir = "../reports"

    os.makedirs(
        reports_dir,
        exist_ok=True
    )

    
    # -----------------------------------------------------
    # 12. Graph - Precision by class
    # -----------------------------------------------------

    available_classes = list(
        class_p1.keys()
    )

    if available_classes:

        x = np.arange(
            len(available_classes)
        )

        width = 0.35

        plt.figure(
            figsize=(8, 5)
        )

        plt.bar(
            x - width / 2,
            [
                class_p1[c]
                for c in available_classes
            ],
            width,
            label="Precision@1"
        )

        plt.bar(
            x + width / 2,
            [
                class_p5[c]
                for c in available_classes
            ],
            width,
            label="Precision@5"
        )

        plt.ylim(
            0,
            1
        )

        plt.xticks(
            x,
            available_classes
        )

        plt.xlabel(
            "BUSI class"
        )

        plt.ylabel(
            "Precision"
        )

        plt.title(
            "CBIR Precision by BUSI Class"
        )

        plt.legend()

        plt.tight_layout()

        plt.savefig(
            os.path.join(
                reports_dir,
                "precision_by_class.png"
            ),
            dpi=300
        )

        plt.close()

    # -----------------------------------------------------
    # 13. Save metrics
    # -----------------------------------------------------

    save_metrics_summary(
        results,
        mean_p1,
        mean_p5,
        class_results,
        database_size=len(database_images),
        query_size=len(QUERY_INDICES)
    )

    # -----------------------------------------------------
    # 14. Final message
    # -----------------------------------------------------

    print(
        "\n" + "=" * 70
    )

    print(
        "EVALUATION COMPLETE"
    )

    print(
        "=" * 70
    )

    print(
        f"\nReports saved in: "
        f"{reports_dir}"
    )

    print(
        "\nGenerated files:"
    )
    print(
        "  - precision_by_class.png"
    )

    print(
        "  - metrics_summary.txt"
    )


if __name__ == "__main__":
    main()