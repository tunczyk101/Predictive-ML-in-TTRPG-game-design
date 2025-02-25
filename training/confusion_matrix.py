import math

from sortedcontainers import SortedSet


class ConfusionMatrix:
    def __init__(self):
        self.confusion_matrix = {}
        self.index_class = {}
        self.frequency_classes_in_gold_per_topic = {}
        self.frequency_classes_in_output_per_topic = {}

    def generate_confusion_matrix(self, output, gold):
        for topic, gold_values in gold.table_of_topics.items():
            output_values = output.table_of_topics.get(topic, {})
            self.index_class[topic] = {}
            self.frequency_classes_in_gold_per_topic[topic] = {}
            self.frequency_classes_in_output_per_topic[topic] = {}

            self.parse_confusion_matrix_for_topic(topic, gold_values, output_values)

    def parse_confusion_matrix_for_topic(self, topic, gold_values, output_values):
        conf_mat = self.identify_gold_classes_and_calculate_their_frequency(topic, gold_values)
        self.identify_output_classes_and_calculate_their_frequency(topic, output_values)

        if output_values:
            for id_, gold_value in gold_values.items():
                output_value = output_values.get(id_)
                if output_value:
                    pos_gold = self.index_class[topic][gold_value]
                    if self.index_class[topic].get(output_value):
                        pos_output = self.index_class[topic][output_value]
                        conf_mat[pos_gold][pos_output] += 1

        self.confusion_matrix[topic] = conf_mat

    def identify_gold_classes_and_calculate_their_frequency(self, topic, gold_values):
        num_classes_in_gold = 0
        for gold_value in gold_values.values():
            if gold_value not in self.index_class[topic]:
                self.index_class[topic][gold_value] = len(self.index_class[topic])
                num_classes_in_gold += 1

            self.frequency_classes_in_gold_per_topic[topic][gold_value] = \
                self.frequency_classes_in_gold_per_topic[topic].get(gold_value, 0) + 1

        return [[0] * num_classes_in_gold for _ in range(num_classes_in_gold)]

    def identify_output_classes_and_calculate_their_frequency(self, topic, output_values):
        if output_values:
            for output_value in output_values.values():
                self.frequency_classes_in_output_per_topic[topic][output_value] = \
                    self.frequency_classes_in_output_per_topic[topic].get(output_value, 0) + 1

    def get_class_name(self, topic, index):
        for class_name, idx in self.index_class[topic].items():
            if idx == index:
                return class_name
        return None

    def get_index_by_class_name_for_topic(self, topic, class_name):
        return self.index_class[topic].get(class_name, -1)

    def get_diagonal_for_accuracy(self, topic):
        diagonal = 0
        conf_mat = self.confusion_matrix[topic]
        for i in range(len(conf_mat)):
            diagonal += conf_mat[i][i]
        return diagonal

    def get_number_instances_in_gold(self, topic):
        return sum(self.frequency_classes_in_gold_per_topic[topic].values())

    def get_diagonal_for_class(self, topic, index_class):
        return self.confusion_matrix[topic][index_class][index_class]

    def get_anti_diagonal_for_class_in_matrix_2x2(self, topic, index_class):
        index_output = len(self.confusion_matrix[topic]) - 1 - index_class
        return self.confusion_matrix[topic][index_class][index_output]

    def get_number_instances_per_class_in_gold(self, topic, index_class):
        class_name = self.get_class_name(topic, index_class)
        return self.frequency_classes_in_gold_per_topic[topic].get(class_name, 0)

    def get_number_instances_per_class_output(self, topic, index_class):
        class_name = self.get_class_name(topic, index_class)
        return self.frequency_classes_in_output_per_topic.get(topic, {}).get(class_name, 0)

    def get_pos_in_matrix(self, topic, index_gold, index_output):
        return self.confusion_matrix[topic][index_gold][index_output]

    def get_ordered_classes_between_two_classes(self, topic, ci_class, cj_class):
        classes = SortedSet(key=lambda x: float(x))

        # Add both gold and output classes to generate the ordinal index
        classes.update(self.index_class.get(topic, {}).keys())
        classes.update(self.frequency_classes_in_output_per_topic.get(topic, {}).keys())

        # Check order range and decide inclusion of boundary classes
        begin, end = ci_class, cj_class
        include_begin = False
        include_end = False
        if float(ci_class) < float(cj_class):
            include_end = True
        else:
            begin, end = cj_class, ci_class
            include_begin = True

        # Get the subset of classes between the two boundaries
        subset = classes.irange(begin, end, (include_begin, include_end))

        # Get index of each class in the subset, return -1 if not found
        subset_classes = [self.get_index_by_class_name_for_topic(topic, cls) for cls in subset]

        return subset_classes

    def proximity_CEM(self, topic, ci_class, cj_class):
        items_in_gold = self.get_number_instances_in_gold(topic)
        items_gold_class_ci = self.get_number_instances_per_class_in_gold(topic, self.get_index_by_class_name_for_topic(topic, ci_class))

        sum_items_classes = 0.0
        if ci_class != cj_class:
            subset_classes = self.get_ordered_classes_between_two_classes(topic, ci_class, cj_class)
            for idx in subset_classes:
                if idx != -1:
                    sum_items_classes += self.get_number_instances_per_class_in_gold(topic, idx)

        proximity = 0.0
        if items_in_gold != 0.0:
            proximity = ((items_gold_class_ci / 2) + sum_items_classes) / items_in_gold
        if proximity > 0.0:
            proximity = -1 * (math.log10(proximity) / math.log10(2))

        return proximity
