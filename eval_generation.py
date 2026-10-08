
import csv
import re

from app import ConversationMemory, answer_question, load_clients
from eval_rag import TEST_CASES


KEYWORD_CHECKS: dict[str, list[str]] = {
    "ما هو مبدأ عدم اليقين لهايزنبرغ؟": ["هايزنبرغ", "الموضع", "الزخم"],
    "ما هي تجربة قطة شرودنغر؟": ["شرودنغر", "قطة", "تراكب"],
    "ما هو التشابك الكمومي؟": ["تشابك", "أينشتاين"],
    "ما الفرق بين البت والكيوبت؟": ["بت", "كيوبت"],
    "ما وظيفة بوابة هادامارد؟": ["هادامارد", "تراكب"],
    "ما هي بوابة CNOT؟": ["CNOT", "تشابك", "تحكم"],
    "لماذا لا تُستخدم الحوسبة الكمومية في كل شيء؟": ["تشوش", "أخطاء", "Decoherence"],
    "ما هي خوارزمية شور؟": ["شور", "تحليل", "عوامل أولية", "RSA"],
    "ماذا تفعل خوارزمية غروفر؟": ["غروفر", "بحث", "تربيعي"],
    "ما هي خوارزمية VQE؟": ["VQE", "طاقة"],
    "كيف تعمل الكيوبتات فائقة التوصيل؟": ["فائقة التوصيل", "تبريد", "IBM"],
    "ما هو تصحيح الأخطاء الكمومية؟": ["تصحيح", "كيوبت منطقي", "منطقي"],
    "ما هو عصر NISQ؟": ["NISQ", "بريسكيل"],
    "How does Grover's algorithm speed up search?": [
        "Grover",
        "quadratic",
        "square root",
    ],
    "What are trapped ion qubits?": ["ion", "laser", "trapped"],
}

REFUSAL_MARKERS = (
    "not available",
    "no sufficiently relevant",
    "لا تتوفر معلومات كافية",
)

ARABIC_RANGE = re.compile(r"[\u0600-\u06FF]")
LATIN_RANGE = re.compile(r"[a-zA-Z]")


def _is_refusal(answer: str) -> bool:
    lowered = answer.lower()
    return any(marker in lowered for marker in REFUSAL_MARKERS)


def _keyword_hit(answer: str, keywords: list[str]) -> bool | None:
    if not keywords:
        return None  # لا يوجد فحص محتوى معرّف لهذا السؤال
    lowered = answer.lower()
    return any(keyword.lower() in lowered for keyword in keywords)


def _question_is_arabic(question: str) -> bool:
    arabic_chars = len(ARABIC_RANGE.findall(question))
    latin_chars = len(LATIN_RANGE.findall(question))
    return arabic_chars >= latin_chars


def _language_matches(question: str, answer: str) -> bool:
    question_arabic = _question_is_arabic(question)
    answer_arabic_chars = len(ARABIC_RANGE.findall(answer))
    answer_latin_chars = len(LATIN_RANGE.findall(answer))
    if answer_arabic_chars + answer_latin_chars == 0:
        return True  # إجابة بدون حروف (رقم بس مثلًا) - ما نحكم عليها
    answer_is_arabic = answer_arabic_chars >= answer_latin_chars
    return question_arabic == answer_is_arabic


def main() -> None:
    llm_client, embedding_model, qdrant_client = load_clients()

    print(f"Running {len(TEST_CASES)} questions through the full pipeline...")
    print("(each uses a fresh, independent conversation memory)\n")

    rows = []
    positive_correct = positive_total = 0
    negative_correct = negative_total = 0
    keyword_checked = keyword_passed = 0
    language_ok = 0

    for index, (question, expected) in enumerate(TEST_CASES, start=1):
       uv run black app.py chat_app.py storage.py eval_rag.py eval_generation.py compare_embeddings.py
        memory = ConversationMemory()

        answer, sources = answer_question(
            llm_client=llm_client,
            embedding_model=embedding_model,
            qdrant_client=qdrant_client,
            query=question,
            memory=memory,
        )

        refused = _is_refusal(answer)
        lang_ok = _language_matches(question, answer)
        language_ok += int(lang_ok)

        if expected is None:
            negative_total += 1
            correct = refused
            negative_correct += int(correct)
            keyword_result = None
        else:
            positive_total += 1
            source_ok = expected in sources
            correct = source_ok and not refused
            positive_correct += int(correct)

            keyword_result = _keyword_hit(answer, KEYWORD_CHECKS.get(question, []))
            if keyword_result is not None:
                keyword_checked += 1
                keyword_passed += int(keyword_result)

        verdict = "PASS" if correct else "FAIL"
        print(f"{index:>2}. [{verdict}] {question[:50]}")

        rows.append(
            {
                "n": index,
                "question": question,
                "expected_source": expected or "NONE",
                "sources_returned": ", ".join(sources),
                "refused": refused,
                "language_ok": lang_ok,
                "keyword_check": (
                    "n/a"
                    if keyword_result is None
                    else ("PASS" if keyword_result else "FAIL")
                ),
                "answer": answer,
                "result": verdict,
            }
        )

    print("\n" + "-" * 60)
    print(
        f"Positive questions correct (source + no false refusal): {positive_correct}/{positive_total}"
    )
    print(
        f"Out-of-scope questions correctly refused                : {negative_correct}/{negative_total}"
    )
    print(
        f"Answers matching question language                      : {language_ok}/{len(TEST_CASES)}"
    )
    if keyword_checked:
        print(
            f"Keyword content check (subset only)                     : {keyword_passed}/{keyword_checked}"
        )

    with open(
        "eval_generation_results.csv", "w", newline="", encoding="utf-8-sig"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(
        "\nFull results (including full answer text) saved to eval_generation_results.csv"
    )


if __name__ == "__main__":
    main()
