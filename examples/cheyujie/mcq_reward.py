import re


def extract_answer(solution_str):
    pattern = r"答案\s*[:：]\s*([A-Za-z0-9,，]+)"
    match = re.search(pattern, solution_str, flags=re.MULTILINE)
    if match:
        matched = match.group(1).strip().rstrip("。.")
        matched = matched.replace(" 和 ", "").replace(" ", "").replace("，", ",")
        answers = set(matched.split(",")) if "," in matched else set(matched)
        return "".join(sorted(list(answers)))
    else:
        return ""


def calculate_score(prediction, reference):
    """
    Calculate accuracy for multiple-choice questions.

    Args:
        prediction (str): Predicted answer.
        reference (str): Correct answer.

    Returns:
        float: Accuracy score (1.0 for correct, 0.0 for incorrect).
    """
    prediction = set(prediction.strip().upper())
    reference = set(reference.strip().upper())
    # if the prediction has answer that not in reference, it is wrong
    if not prediction or not prediction.issubset(reference):
        return 0.0
    common = prediction.intersection(reference)
    return round(len(common) / len(reference), 2) if reference else 0.0


def compute_score(data_source, solution_str, ground_truth, extra_info=None):
    assert data_source == "mcq", "data_source must be mcq"
    prediction = extract_answer(solution_str)
    score = calculate_score(prediction, ground_truth)
    solution_str = solution_str.replace('\n', '')
    print(f"{solution_str[:20]}...{solution_str[-20:]}")
    print(f"prediction: {prediction}, ground_truth: {ground_truth}, score: {score}")
    return score


if __name__ == "__main__":
    print(compute_score("mcq", "XXX，答案: A", "A"))
    print(compute_score("mcq", "XXX，答案：ABC", "ABC"))
    print(compute_score("mcq", "XXX，答案：BC", "ABC"))
    print(compute_score("mcq", "XXX，答案：BCD", "ABC"))
