# Course Project: Machine Learning Systems

## 1. Project Overview

The goal of this project is to provide hands-on experience in the intersection of machine learning algorithms and system design. Unlike pure ML projects that focus solely on model accuracy, this project requires you to focus on **system efficiency, scalability, latency, memory usage, and hardware utilization.**

Students are expected to identify a performance bottleneck in an existing ML workflow and propose a solution involving **system optimization** or **algorithm-system co-design**.

## 2. Project Scope & Topics

Your project must address a *systems* challenge. Pure algorithmic improvements (e.g., "Improving ResNet accuracy by 1% using a new loss function") are **not** accepted unless accompanied by a systems analysis (e.g., "Improving convergence speed to reduce training cost").

**Suggested Topics:**

* **Inference Optimization:** Model compression (quantization, pruning, distillation) and its impact on latency/throughput on specific hardware (CPU/GPU/Edge).
* **Training Acceleration:** Distributed training strategies, gradient compression, or mixed-precision training.
* **Memory Management:** Optimizing activation offloading, paging, or memory-efficient attention mechanisms (e.g., FlashAttention implementations).
* **Compiler Optimization:** Utilizing Tensor compilers (e.g., TVM, XLA) to optimize operator fusion and code generation.
* **Algorithm-System Co-design:** Designing a neural network architecture specifically tailored for a hardware constraint.

## 3. Team Formation

* **Group Size:** Strictly **4 students** per group.
* **Submission Method:** Please submit your group members via [Feishu Form](https://sii-czxy.feishu.cn/share/base/form/shrcn8v97lXcrDNQYUr8We2qtyh).
* **Policy:** Any student who has not submitted group information by the deadline will be **randomly assigned** to a group by the teaching staff. No exceptions.

## 4. Project Timeline & Deliverables

| Milestone | Deadline | Weight | Description |
| :--- | :--- | :--- | :--- |
| **Team Registration** | **Nov 27** | N/A | Submit member names. |
| **Project Proposal** | **Dec 4** | **5%** | A short document outlining the problem and plan. |
| **Final Report** | **Dec 31** | **20%** | The technical paper detailing your work. |
| **Presentation** | **Jan 8** | **20%** | Group presentation and Q&A. |

**All submission deadlines are in Beijing Time (GMT+8), with a cutoff time at 23:59.**

For each deliverable, including the team member list, proposal and final report, submit exactly once per group through the designated forms; individual members do not need to submit duplicates.

---

## 5. Detailed Requirements

### A. Project Proposal (Due: Dec 4)

* **Length:** 1-2 pages (PDF).
* **Content:**
  1. **Problem Statement:** What system bottleneck are you solving?
  2. **Proposed Method:** What techniques (optimization/co-design) will you use?
  3. **Evaluation Plan:** What metrics will you measure? (e.g., Latency in ms, Throughput in samples/sec, FLOPs, GPU memory usage).
* **Constraint:** The proposal **must be system-related**.
  * *Pass:* "Accelerating Transformer Inference on Mobile CPUs."
  * *Fail:* "Using Transformers for Sentiment Analysis."
* **Grading Policy:** This component is worth **5 points**. If the proposal is deemed "Not System Related," the group will receive feedback and must **resubmit** within 3 days to proceed.
* **Submission:** Upload PDF to [Feishu Form](https://sii-czxy.feishu.cn/share/base/form/shrcn49MLsk63p8w9IhhZCfPPhc).

### B. Final Report (Due: Dec 31)

* **Format:** 4-6 pages (Standard Conference Two-Column Format, e.g., MLSys template or ACM template).
* **Language:** English.
* **Content:**
  * **Introduction & Motivation:** Why does this performance issue matter?
  * **Methodology:** Technical details of your system optimization or co-design.
  * **Experiments:**
    * **Baselines:** What are you comparing against?
    * **System Metrics:** You must report speedup, memory reduction, or efficiency gains. Accuracy should be reported to ensure the optimization didn't break the model.
  * **Ablation Studies:** Which part of your method contributed most to the gain?
  * **Collaboration:** Which part is implemented by each team member?
  * **Conclusion:** What did you learn from this project?
* **Submission:** Upload PDF and artifact (including code, evaluation data, etc.) to [Feishu Form](https://sii-czxy.feishu.cn/share/base/form/shrcnVqCdkJvmoFt4VeNQPxO45f).

### C. Final Presentation (Date: Jan 8)

* **Format:** In-class presentation.
* **Time:** TBD, will be announced later.
* **Expectation:** All team members should participate. Focus on the "How" (implementation) and the "Results" (graphs/charts).

## 6. Grading Criteria

Your project will be evaluated based on:

1. **System Novelty & Difficulty:** Did you implement a complex system optimization or just run a script?
2. **Experimental Rigor:** Are the benchmarks fair? Did you measure the right metrics?
3. **Completeness:** Did you finish what was proposed?
4. **Clarity:** Is the report well-written and the presentation clear?

As a default, all members of the same group receive the same grade, but significant differences in individual contributions may result in adjusted scores for specific teammates.
