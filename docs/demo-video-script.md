# InterviewIQ — Hackathon Demo Video Script

**Target Duration:** < 3 minutes (Approx. 2 minutes 40 seconds)  
**Required Tags:** `#strands-decider-2b` `#policy`

---

### [0:00 - 0:30] Introduction & The Problem
- **Visual:** Screen shows the InterviewIQ Dark-mode dashboard (`localhost:5173`).
- **Speaker:**  
  *"Hi everyone! This is InterviewIQ, an Adaptive AI Interview Agent built for the AWS Strands Decider Hackathon. Traditional AI interview apps send every candidate response straight to heavy LLMs. If a user types 'idk' or is slightly ambiguous, cloud tokens are wasted, and models often hallucinate scores. Today, we're demonstrating how Strands Decider 2B solves this through the #policy pattern paired with Amazon Bedrock."*

---

### [0:30 - 1:15] Scenario 1: Fast Triage & Generative Evaluation (`EVALUATE`)
- **Visual:** Click "Practice Interview" -> Select "Python" -> Question: *"What is the Global Interpreter Lock (GIL) in CPython?"* -> Enter: *"The GIL is a mutex that protects Python objects, allowing only one thread to execute bytecodes at a time."* -> Click **Submit Answer**.
- **Visual:** The "Agent Decision Gate" panel highlights `EVALUATE`, displaying `Decider Confidence: 85%`, `Latency: <150ms`, `#policy`, and `#strands-decider-2b`.
- **Speaker:**  
  *"When we submit a substantive technical answer, Strands Decider 2B immediately triages it in under 150 milliseconds. The pointer head scores the options, chooses EVALUATE with calibrated confidence, and routes the response to Amazon Bedrock. Bedrock produces comprehensive feedback with 4-criterion rubric scoring, strengths, and an improved model answer."*

---

### [1:15 - 1:55] Scenario 2: Adaptive Clarification Flow (`ASK_CLARIFICATION`)
- **Visual:** Move to next question: *"Explain the bias-variance tradeoff."* -> Enter: *"It's about error and it depends on data."* -> Click **Submit Answer**.
- **Visual:** The Agent Decision Panel displays `ASK CLARIFICATION`. A blue targeted follow-up prompt appears: *"Could you clarify how model complexity influences bias versus variance specifically?"*
- **Visual:** Candidate enters clarification: *"Higher complexity lowers bias but increases variance."* -> Click **Submit Clarification**. Full evaluation renders.
- **Speaker:**  
  *"Notice what happens when the answer is relevant but ambiguous. Decider 2B doesn't fail the candidate. It selects ASK_CLARIFICATION! Bedrock generates one targeted follow-up question. The candidate clarifies, and the agent evaluates their full combined knowledge, just like a human interviewer."*

---

### [1:55 - 2:25] Scenario 3: Evasive Answer Redirection (`REDIRECT`)
- **Visual:** Next question -> Candidate enters: *"idk pass please"* -> Click **Submit Answer**.
- **Visual:** Agent Decision Gate shows `REDIRECT`. An amber warning appears: *"Action: Redirected · Please Retry. Your response did not address the interview question."* No score is fabricated; the input box remains ready.
- **Speaker:**  
  *"Now watch an evasive response. When someone writes 'idk pass', Decider 2B routes to REDIRECT with over 90% confidence. It protects Bedrock token quota, prevents hallucinated feedback, and prompts the user to retry without penalizing their interview score."*

---

### [2:25 - 2:40] Conclusion & Reusable Pattern
- **Visual:** Show `patterns/adaptive_interview_policy/` directory in code editor and GitHub repo.
- **Speaker:**  
  *"We have also published this standalone reusable policy pattern for the community under Apache-2.0. Check out the repository on GitHub, and start building smarter agents with Strands Decider 2B and Amazon Bedrock. Thank you!"*
