import re

def _check_accuracy(answer: str, expected: str) -> bool:
    answer_lower = answer.lower().strip()
    expected_lower = expected.lower().strip()

    if expected_lower in answer_lower:
        print("DEBUG: Tier 1 match")
        return True

    expected_words = [
        w for w in re.sub(r'[^a-z0-9\s]', ' ', expected_lower).split()
        if len(w) > 2
    ]
    if expected_words and all(w in answer_lower for w in expected_words):
        print("DEBUG: Tier 2 match")
        return True

    answer_no_commas = answer_lower.replace(',', '')
    expected_no_commas = expected_lower.replace(',', '')
    answer_nums = set(re.findall(r'\d+', answer_no_commas))
    expected_nums = set(re.findall(r'\d+', expected_no_commas))
    
    print(f"DEBUG: expected_nums={expected_nums}, answer_nums={answer_nums}")
    
    if expected_nums and expected_nums.issubset(answer_nums):
        print("DEBUG: Tier 3 subset check")
        significant = [w for w in expected_words if not w.isdigit() and len(w) > 3]
        if not significant:
            print("DEBUG: Tier 3 match (no significant words)")
            return True
        if any(w in answer_lower for w in significant):
            print("DEBUG: Tier 3 match (significant word match)")
            return True

    expected_name_parts = [
        w for w in expected_lower.split()
        if len(w) > 3 and w.isalpha()
    ]
    if expected_name_parts:
        if all(part in answer_lower for part in expected_name_parts):
            return True

    return False

# Test Case 9
expected = "31,000+"
got = "Nvidia has 40,000 global startups using NVIDIA AI technology to power AI factories."
print(f"Test Case 9 Result: {_check_accuracy(got, expected)}")
