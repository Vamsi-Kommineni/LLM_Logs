"""The four questions from the 2024 version of this project, asked through Groq and logged.

    pip install "llm-logs[groq]"
    export GROQ_API_KEY=...            # or put it in .env and load it your usual way
    export GROQ_MODEL=...              # optional, see below
    python examples/groq_quickstart.py
    llm-logs stats --db logs/llm.db
    llm-logs tail  --db logs/llm.db

Providers retire models often. If the default below has stopped working, list
the current ones with ``curl -H "Authorization: Bearer $GROQ_API_KEY"
https://api.groq.com/openai/v1/models`` and set ``GROQ_MODEL``.
"""

from __future__ import annotations

import os

from groq import Groq

import llm_logs as ll

MODEL = os.environ.get("GROQ_MODEL", "openai/gpt-oss-20b")

INSTRUCTIONS = (
    "Use the provided pieces of context to answer the query. If you don't know the answer, "
    "just say that you don't know, don't try to make up an answer."
)

QUESTIONS = [
    (
        "The Friedrich Schiller University Jena is a university rich in tradition and strong in "
        "research with a wide range of subjects. With almost 18,000 students and more than 8,600 "
        "employees, the university significantly shapes Jena's character as a cosmopolitan and "
        "future-oriented city.",
        "How many students are there in Friedrich Schiller University Jena?",
    ),
    (
        "We surveyed a range of habitats and recorded 817 audio-files from 678 individuals of 35 "
        "bat species across Thailand between 2003 to 2009 and Xishuangbanna in China from 2017 to "
        "2018. In addition 21 audio-files from 21 individuals of four species collected in "
        "Malaysia, which extracted from a public bioacoustic database.",
        "How many audio files were collected?",
    ),
    (
        "We stopped the network training after 70 epochs to prevent overfitting. The training "
        "lasted 8 days on our configuration; we trained and ran our code on a computer with 64GB "
        "of RAM, an i7 3.50GHz CPU and a Titan X GPU card for 900,000 images.",
        "What is the hardware used to execute the code?",
    ),
    (
        "Google says that 15 % of the company's total energy consumption went towards machine "
        "learning related computing across research, development, and production. NVIDIA has "
        "estimated that 80-90% of machine learning workload is inference processing.",
        "How much energy consumption was used towards machine learning?",
    ),
]


@ll.trace(ignore_args=["client"])
def answer(client: Groq, context: str, query: str, *, model: str, temperature: float = 0.0):  # type: ignore[no-untyped-def]
    """The model and temperature are recorded because they are arguments of this call."""
    return client.chat.completions.create(
        model=model,
        temperature=temperature,
        messages=[
            {"role": "system", "content": INSTRUCTIONS},
            {"role": "user", "content": f"Context:\n{context}\n\nQuery: {query}"},
        ],
    )


def main() -> None:
    ll.configure(sinks=[ll.JsonlSink("logs/"), ll.SqliteSink("logs/llm.db")])
    client = Groq()  # reads GROQ_API_KEY by itself; this script never touches the key

    with ll.span("quickstart", session_id="demo", metadata={"source": "legacy prompts"}):
        for context, query in QUESTIONS:
            response = answer(client, context, query, model=MODEL)
            print(f"Q: {query}\nA: {response.choices[0].message.content}\n")

    ll.shutdown()
    print(ll.stats())


if __name__ == "__main__":
    main()
