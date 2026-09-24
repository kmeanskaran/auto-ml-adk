# Designing Production-Level Agents with Google ADK

## PoC: Autonomous ML Engineer

### 1. PoC idea

Build an **Autonomous ML Engineer** that receives a dataset and an ML objective, then independently works through a simplified machine learning lifecycle.

The purpose is **not to build an AutoML platform**.

The purpose is to create a realistic autonomous agent that gives us a good environment to demonstrate what is required to run agents reliably in production using **Google ADK and Gemini Enterprise Agent Platform**.

The user should be able to provide something like:

> "Build a model to predict customer churn using this dataset. Optimize for F1 score."

The agent then handles the workflow.

---

## 2. Core workflow

```text
                 ML Objective
                      |
                      ↓
              ML Orchestrator
                      |
          ┌───────────┼───────────┐
          ↓           ↓           ↓
     Data Agent   Feature Agent  Training Agent
          |           |           |
          └───────────┼───────────┘
                      ↓
               Evaluation Agent
                      |
                 Is it good?
                  /       \
                No         Yes
                ↓           ↓
          Improve Plan    Complete
                |
                └──────→ Training
```

The system should have a small number of specialized agents rather than dozens of agents.

### ML Orchestrator

Responsible for coordinating the overall workflow.

### Data Agent

* Inspect dataset
* Understand columns
* Detect missing values
* Identify basic data-quality issues
* Prepare the dataset

### Feature Agent

* Identify useful features
* Apply simple transformations
* Create the feature pipeline

### Training Agent

* Select a suitable model
* Train candidate models
* Track experiments
* Store model artifacts and metrics

### Evaluation Agent

* Evaluate the model
* Compare metrics against the objective
* Decide whether another iteration is required
* Produce the final evaluation

---

# 3. The interesting part: autonomous iteration

The agent should not simply run:

```text
Data → Train → Done
```

It should be able to reason over the result.

For example:

```text
Training
   ↓
F1 = 0.71
   ↓
Evaluation Agent
   ↓
Below target
   ↓
Improvement Plan
   ↓
Feature Engineering
   ↓
Training
   ↓
F1 = 0.84
   ↓
Evaluation
   ↓
Target achieved
```

This gives the PoC a genuine **agentic loop**.

---

# 4. Production-level behavior

The PoC then treats this ML agent as a potentially long-running production workload.

For example, a training experiment could take a long time.

The workflow could be:

```text
Start Experiment
       ↓
Data Processing
       ↓
Training
       ↓
Checkpoint
       ↓
Execution interrupted
       ↓
Resume
       ↓
Evaluation
```

The important question becomes:

> **How do we make sure the agent does not lose its workflow when something goes wrong?**

This is where the production architecture around the agent becomes important.

---

# 5. Agent tools

The agents should interact with tools rather than directly manipulating everything.

Example tools:

```text
read_dataset()
profile_dataset()
clean_dataset()
create_features()
run_training()
evaluate_model()
save_artifact()
get_experiment_history()
```

The agent decides **which tool to use**, while the application controls what the tool is actually allowed to do.

This creates a natural boundary between:

```text
Agent reasoning
      ↓
Tool request
      ↓
Production controls
      ↓
Tool execution
```

---

# 6. Production Harness concept

The main PoC should have a reusable layer around the ML agents.

```text
                  Production Harness
                         |
        ┌────────────────┼────────────────┐
        ↓                ↓                ↓
      State          Tool Control     Evaluation
        ↓                ↓                ↓
     Recovery        Guardrails       Regression
        ↓                ↓                ↓
   Human Approval   Observability     Tracing
                         |
                         ↓
                  Autonomous ML Agent
```

The harness should not contain ML-specific logic.

For example, it should not know what feature engineering means.

It should provide generic capabilities such as:

* maintain execution state
* control tool calls
* enforce policies
* pause for approval
* recover failed workflows
* record execution
* evaluate behavior
* track execution/cost
* provide observability

That makes the harness concept reusable for other agents later.

---

# 7. Google ADK's role

Google ADK should be the core framework for implementing the agent system.

Use it for:

### Agent orchestration

Create the ML Orchestrator and specialist agents.

### Workflow

Represent the lifecycle as a structured workflow.

### Sessions/state

Maintain the context of an experiment.

### Callbacks/plugins

Use lifecycle interception for production controls around model and tool execution.

### Tools

Expose the ML operations to agents through controlled tools.

### Human interaction

Allow workflows to pause when an action requires approval.

### Evaluation

Evaluate agent behavior and trajectories rather than only looking at the final answer.

---

# 8. Gemini Enterprise Agent Platform role

The PoC can then move the agent from local development toward a managed production environment.

The platform layer can provide:

```text
ADK Agent
    ↓
Agent Runtime
    ↓
Memory / State
    ↓
Code Execution
    ↓
Gateway / Identity
    ↓
Observability
```

Use the platform features only where they solve an actual problem in the PoC.

For example:

### Agent Runtime

Run the autonomous ML agent as a deployed service.

### Code Execution

Allow the ML agent to generate and execute Python for data analysis and model training in an isolated environment.

### Memory Bank

Store useful long-term experiment context.

### Agent Gateway

Provide controlled connectivity to tools/services.

### Agent Identity

Give the deployed agent a controlled identity rather than treating it as an anonymous application.

### Observability

See what the agent, model and tools actually did during an execution.

---

# 9. Example production scenario

The main demo should show one complete experiment.

User:

> "Train a churn prediction model and achieve at least 0.80 F1."

The system:

```text
User
 ↓
ML Orchestrator
 ↓
Data Agent
 ↓
Dataset analysis
 ↓
Feature Agent
 ↓
Feature creation
 ↓
Training Agent
 ↓
Model training
 ↓
Evaluation Agent
 ↓
F1 = 0.73
 ↓
Below target
 ↓
Improvement plan
 ↓
Feature Agent
 ↓
Training Agent
 ↓
F1 = 0.84
 ↓
Evaluation
 ↓
Target achieved
 ↓
Final Report
```

During the demonstration, intentionally introduce a failure:

```text
Training
   ↓
Execution interrupted
   ↓
Harness detects interruption
   ↓
Restore state
   ↓
Resume training/evaluation
```

Then demonstrate a controlled action:

```text
Agent requests expensive experiment
          ↓
      Harness policy
          ↓
    Approval required
          ↓
       Human approves
          ↓
       Tool executes
```

Finally show the execution trace and evaluation results.

---

# 10. Evaluation

The PoC should contain a small set of predefined scenarios.

For example:

### Scenario 1: Normal dataset

Expected:

```text
Inspect → Clean → Train → Evaluate
```

### Scenario 2: Missing values

Expected:

```text
Inspect → Detect → Clean → Train → Evaluate
```

### Scenario 3: Poor model

Expected:

```text
Train → Evaluate → Improve → Retrain
```

### Scenario 4: Failed execution

Expected:

```text
Checkpoint → Failure → Resume
```

### Scenario 5: Restricted operation

Expected:

```text
Tool Request → Policy → Approval → Execute
```

The evaluation should inspect both:

**Final outcome**

and

**Agent trajectory.**

For example:

```text
Expected:
Data → Clean → Train → Evaluate

Actual:
Data → Train → Train → Evaluate
```

Even if the final model is acceptable, the trajectory may reveal inefficient or incorrect behavior.

---

# 11. What the final PoC demonstrates

At the end, we should be able to demonstrate:

### Agent capability

The system can autonomously complete an ML workflow.

### Multi-agent orchestration

Different agents collaborate through ADK.

### Long-running execution

The workflow can survive operations that take time.

### State and recovery

The workflow can resume rather than restart.

### Tool governance

Agents do not have unrestricted access to tools.

### Human approval

Sensitive or expensive actions can require approval.

### Code execution

ML code can be executed in a controlled environment.

### Memory

The system can retain useful experiment context.

### Observability

We can understand what happened during an execution.

### Evaluation

We can test whether the agent behaves correctly.

### Regression

We can detect behavioral changes after modifying the agent.

---

# 12. What we should keep deliberately small

The ML side should remain simple.

Use:

* one public dataset
* one ML problem
* a few models
* a small number of agents
* basic feature engineering
* simple evaluation metrics
* a simple UI or API

Do **not** spend the majority of the PoC building an advanced ML platform.

The interesting engineering is the **agent architecture and production harness**.

---

# 13. Final PoC architecture

```text
                         USER
                           |
                           ↓
                Autonomous ML Engineer
                           |
                    ADK Orchestrator
                           |
             ┌─────────────┼─────────────┐
             ↓             ↓             ↓
        Data Agent    Feature Agent   Training Agent
             |             |             |
             └─────────────┼─────────────┘
                           ↓
                    Evaluation Agent
                           |
                     Improvement Loop
                           |
                           ↓
                    ADK Tools
                           |
              ┌────────────┴────────────┐
              ↓                         ↓
       Code Execution              Artifacts
          Sandbox
              |
              ↓
       ┌────────────────────────────────┐
       │       PRODUCTION HARNESS       │
       │                                │
       │ State / Recovery               │
       │ Tool Control                   │
       │ Guardrails                     │
       │ Human Approval                 │
       │ Evaluation                     │
       │ Observability                  │
       │ Identity / Access              │
       └───────────────┬────────────────┘
                       ↓
       Gemini Enterprise Agent Platform
```

## One-line definition of the PoC

> **Build an autonomous ML Engineer with Google ADK, then surround it with a reusable production harness that makes the agent stateful, recoverable, observable, controllable, evaluable, and safe to operate.**

That should be the scope for the vibe-coding build. The implementation should prove this concept rather than trying to become a complete ML platform.
