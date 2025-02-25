class EvALLResult:
    def __init__(self):
        self.results = {}
        self.aggregated_result = None

    def normalize_result(self):
        if len(self.results) == 0:
            self.aggregated_result = None
        else:
            total = 0.0
            num_elems = 0

            for value in self.results.values():
                if value is not None:
                    total += value
                    num_elems += 1

            if total == 0:
                self.aggregated_result = 0.0
            elif num_elems != 0:
                self.aggregated_result = total / num_elems
