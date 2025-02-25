import csv
from collections import defaultdict


def is_numeric(value):
    try:
        float(value)
        return True
    except ValueError:
        return False


class OrdinalClassificationFormat:
    def __init__(self):
        self.is_gold = False
        self.path_file = ""
        self.stop = False
        self.frequency_of_classes = defaultdict(int)
        self.table_of_topics = defaultdict(dict)

    def parse_file(self, is_gold, path_file):
        self.is_gold = is_gold
        self.path_file = path_file
        try:
            with open(path_file, 'r', encoding='utf-8') as file:
                reader = csv.reader(file, delimiter='\t')
                self.parser_internal(reader)
        except FileNotFoundError:
            print(f"File not found: {path_file}")
            self.stop = True

    def parser_internal(self, reader):
        print(f"Parsing file {self.path_file}")
        in_line = 0
        row_with_no_3_columns = 0

        for record in reader:
            in_line += 1

            # Check if the record has 3 columns
            if len(record) != 3:
                message = "Format error" if self.is_gold else "Format warning"
                print(f"{message}: the number of columns must be 3. Line {in_line}")
                row_with_no_3_columns += 1
                if self.is_gold:
                    self.stop = True
                continue

            topic, item_id, value = record

            # Check for empty values
            if not topic or not item_id or not value:
                message = "Format error" if self.is_gold else "Format warning"
                print(f"{message}: the columns in the rows cannot be empty. Line {in_line}")
                if self.is_gold:
                    self.stop = True
                continue

            # Check for duplicate ids in the same topic
            if topic in self.table_of_topics and item_id in self.table_of_topics[topic]:
                message = "Format error" if self.is_gold else "Format warning"
                print(f"{message}: duplicated ids at test case level are not allowed. Line {in_line}")
                if self.is_gold:
                    self.stop = True
                continue

            # Check if the value is numeric
            if not is_numeric(value):
                message = "Format error" if self.is_gold else "Format warning"
                print(f"{message}: the value is not a valid number. Line {in_line}")
                if self.is_gold:
                    self.stop = True
                continue

            # Add the item to the table of topics
            self.table_of_topics[topic][item_id] = value
            self.frequency_of_classes[value] += 1

        if in_line == 0:
            print("Format error: The file is empty.")
            self.stop = True
        elif row_with_no_3_columns == in_line:
            print("Format error: The number of columns must be 3 in all lines.")
            self.stop = True

    def get_elements_classified_by_class_at_topic_level(self, topic, class_for_search):
        topic_values = self.table_of_topics.get(topic, {})
        return [item_id for item_id, val in topic_values.items() if val == class_for_search]
