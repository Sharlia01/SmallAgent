"""对具有明确参照年的时间表达做确定性规范化，不推断模糊范围。"""

from datetime import date


def normalize_relative_years(text: str, reference_date: str) -> str:
    if not reference_date or not any(token in text for token in ("今年", "去年", "明年")):
        return text
    year = date.fromisoformat(reference_date).year
    for token, offset in (("今年", 0), ("去年", -1), ("明年", 1)):
        text = text.replace(token, f"{year + offset}年")
    return text
