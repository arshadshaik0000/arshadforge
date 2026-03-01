from typing import List, Dict, Any, Optional


def prune_vector_citations(
    answer_value: str,
    validated_page: int,
    citations: List[Dict[str, Any]],
    anchor_phrase: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Prune irrelevant vector citations using deterministic scoring.

    A simple scoring scheme ensures only genuinely relevant hits survive:

    * +3 if the quote contains the exact numeric/string answer
    * +2 if the quote contains an optional anchor phrase (e.g. "employment")
    * +2 if the citation page exactly matches the validated page
    * +1 if the citation page is ±1 from the validated page
    * -2 if the quote does *not* contain the answer token at all

    Only citations with score >= 2 are kept.

    This makes provenance explainable and prevents noisy context in
    verification/data-synthesis tests.
    """
    pruned: List[Dict[str, Any]] = []

    answer_value = answer_value.lower().strip()
    for c in citations:
        score = 0
        # support multiple field names from various tools
        quote = c.get("quote", "") or c.get("content", "")
        quote = quote.lower()
        page = c.get("page", 0)

        # exact answer (numeric or string) present
        if answer_value and answer_value in quote:
            score += 3
        else:
            score -= 2

        # anchor phrase such as "employment" can add additional weight
        if anchor_phrase and anchor_phrase.lower() in quote:
            score += 2

        # page alignment incentives
        if page == validated_page:
            score += 2
        elif abs(page - validated_page) == 1:
            score += 1

        if score >= 2:
            pruned.append(c)

    return pruned
