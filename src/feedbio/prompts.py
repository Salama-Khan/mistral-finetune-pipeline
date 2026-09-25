# Same wording as the training JSONL. Inference used to ask for a mark scheme
# that the train set never provided.
SYSTEM_PROMPT = (
    "You are a strict but helpful GCSE Biology examiner. "
    "Your job is to mark answers and provide constructive feedback."
)


def format_user_prompt(question: str, max_marks: int, student_answer: str) -> str:
    return (
        f"Question: {question}\n"
        f"Max Marks: {max_marks}\n"
        f"Student Answer: {student_answer}"
    )
