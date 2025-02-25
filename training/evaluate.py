import os
import csv
import sys

from training.cem import CEMOrd
from training.ordinal_classification_format import OrdinalClassificationFormat


def generate_single_tsv_file_for_one_output(output, gold, cem_ord):
    output_file = "RESULTS.tsv"
    try:
        with open(output_file, mode='w', encoding='utf-8', newline='') as file:
            writer = csv.writer(file, delimiter='\t', quoting=csv.QUOTE_MINIMAL)

            # Table headers
            writer.writerow(["Test Case", "Score"])

            # Gold test cases
            for topic, _ in gold.table_of_topics.items():
                score = cem_ord.result.results.get(topic, "-")
                writer.writerow([topic, f"{score:.4f}" if score != "-" else score])

            # Output test cases not in gold
            for topic, _ in output.table_of_topics.items():
                if topic in gold.table_of_topics:
                    continue
                score = cem_ord.result.get(topic, "-")
                writer.writerow([topic, f"{score:.4f}" if score != "-" else score])

    except Exception as e:
        print(f"Error writing TSV file: {e}")


def main():
    if len(sys.argv) != 3:
        print(
            "The number of parameters must be 2: python3 evaluate.py pathGoldStandard pathSystemOutput\n"
            "Example: python3 evaluate.py test/resources/GOLD.tsv test/resources/SYS.tsv"
        )
        sys.exit(0)

    gold_standard_file = sys.argv[1]
    output_file = sys.argv[2]

    if not gold_standard_file:
        print("The name of the gold standard file cannot be empty")
        sys.exit(0)

    if not output_file:
        print("The name of the system output file cannot be empty")
        sys.exit(0)

    # Parse files
    gold = OrdinalClassificationFormat()
    gold.parse_file(True, gold_standard_file)
    if gold.stop:
        sys.exit(0)

    output = OrdinalClassificationFormat()
    output.parse_file(False, output_file)
    if output.stop:
        sys.exit(0)

    # Evaluate
    cem_ord = CEMOrd(gold, output)
    cem_ord.evaluate()

    # Generate TSV
    generate_single_tsv_file_for_one_output(output, gold, cem_ord)


if __name__ == "__main__":
    main()
