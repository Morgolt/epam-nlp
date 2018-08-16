from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix


class NERQualityReport:

    def __init__(self, matches: list, ground_truth: list) -> None:
        self.matches = matches
        self.ground_truth = ground_truth

    def calculate_report(self):
        # first, calculate absolute metrics (full match, correct type)
        self.matches
