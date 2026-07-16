"""LLM service — async call to Groq.

KEY DESIGN POINT (interview gold):
The Groq API call is I/O-bound: the worker is just waiting on the network
for the model to generate. With AsyncGroq + await, the event loop serves
other requests during that wait. This is the concrete reason this service
is FastAPI/async instead of sync Django views — under concurrent load,
N users can be "waiting on Groq" simultaneously on a single worker.
"""

from groq import AsyncGroq

SYSTEM_PROMPT = (
    "You are a helpful assistant that answers questions strictly using the "
    "provided document context. If the answer is not in the context, say "
    "\"I couldn't find that in the document.\" Do not invent information."
)

PROMPT_TEMPLATE = """Answer the question using ONLY the context below.

Context:
{context}

Question: {question}

Answer:"""


def build_prompt(question: str, chunks: list[dict]) -> str:
    context = "\n\n---\n\n".join(
        f"[Source: {c['document_id']} | relevance: {c['score']}]\n{c['text']}"
        for c in chunks
    )
    return PROMPT_TEMPLATE.format(context=context, question=question)


async def generate_answer(
    client: AsyncGroq,
    model: str,
    question: str,
    chunks: list[dict],
    temperature: float = 0.1,
    max_tokens: int = 1024,
) -> str:
    if not chunks:
        return "I couldn't find any relevant content in your documents for that question."

    response = await client.chat.completions.create(
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_prompt(question, chunks)},
        ],
    )
    return response.choices[0].message.content or ""
