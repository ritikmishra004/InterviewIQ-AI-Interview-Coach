"""
Example runner for Adaptive Interview Policy.
Run this script to verify the #policy pattern using Strands Decider 2B.
"""

from patterns.adaptive_interview_policy.policy import AdaptiveInterviewPolicy


def main():
    import sys
    # If offline or test mode, set enable_local_engine=False or use fallback
    use_local = "--live-model" in sys.argv
    policy = AdaptiveInterviewPolicy(enable_local_engine=use_local)

    scenarios = [
        {
            "title": "Substantive Technical Answer",
            "question": "What is the Global Interpreter Lock (GIL) in CPython?",
            "answer": "The GIL is a mutex that protects access to Python objects, preventing multiple native threads from executing Python bytecodes simultaneously in CPython.",
            "topic": "Python",
        },
        {
            "title": "Short Code Snippet",
            "question": "How do you reverse a list in Python?",
            "answer": "return arr[::-1]",
            "topic": "Python",
        },
        {
            "title": "Partial / Ambiguous Answer",
            "question": "Explain the bias-variance tradeoff in Machine Learning.",
            "answer": "It is about error and it depends on data.",
            "topic": "Machine Learning",
        },
        {
            "title": "Evasive / Gibberish Answer",
            "question": "How does attention work in Transformer models?",
            "answer": "idk skip please",
            "topic": "Deep Learning",
        },
    ]

    print("=" * 70)
    print("Testing Adaptive Interview Policy (#policy / Strands Decider 2B)")
    print("=" * 70)

    for item in scenarios:
        print(f"\nScenario: {item['title']}")
        print(f"Question: {item['question']}")
        print(f"Answer:   {item['answer']}")

        res = policy.decide(
            question=item["question"],
            candidate_answer=item["answer"],
            topic=item["topic"],
        )

        print(f"Decision:    {res['decision']}")
        print(f"Confidence:  {res['confidence']:.2%}")
        print(f"Latency:     {res['latency_ms']:.1f} ms")
        print(f"Action:      {res['action_taken']}")
        print(f"Fallback?:   {res['is_fallback']}")

    print("\n" + "=" * 70)
    print("Pattern verification complete.")
    print("=" * 70)


if __name__ == "__main__":
    main()
