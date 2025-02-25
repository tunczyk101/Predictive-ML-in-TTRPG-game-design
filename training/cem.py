from training.confusion_matrix import ConfusionMatrix
from training.evall_result import EvALLResult


class CEMOrd:
    def __init__(self, gold_standard, output):
        self.gold_standard = gold_standard
        self.output = output
        self.confusion_matrix = ConfusionMatrix()
        self.confusion_matrix.generate_confusion_matrix(self.output, self.gold_standard)
        self.result = EvALLResult()

    def evaluate(self):
        for topic, values_gold in self.gold_standard.table_of_topics.items():
            values_output = self.output.table_of_topics.get(topic)

            cem_ord = 0.0
            sum_numerator = 0.0
            sum_denominator = 0.0

            if values_output is not None:
                for id_gold, class_gold in values_gold.items():
                    if id_gold in values_output:
                        class_output = values_output[id_gold]
                        sum_numerator += self.confusion_matrix.proximity_CEM(topic, class_output, class_gold)

                    sum_denominator += self.confusion_matrix.proximity_CEM(topic, class_gold, class_gold)

                if sum_denominator != 0.0:
                    cem_ord = sum_numerator / sum_denominator

            self.result.results[topic] = cem_ord
