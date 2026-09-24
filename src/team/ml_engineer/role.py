"""ML Engineer.

Fits the candidates the data scientist named, compares F1 with the objective,
raises the strategy when the target is missed, then writes metadata, a serving
contract, and DESIGN.md. Training is the step the harness can pause for a human.
"""

NAME = "ml_engineer"
TITLE = "ML Engineer"
SUMMARY = "Train, evaluate, record metadata, and publish the serving contract."
SPECIALISTS = (
    {
        "name": "training_agent",
        "summary": "Fit the selected candidates and keep the best F1.",
        "prompt": "training.md",
        "tools": ("run_training",),
    },
    {
        "name": "evaluation_agent",
        "summary": "Decide whether the metric is good enough.",
        "prompt": "evaluation.md",
        "tools": ("evaluate_model", "record_improvement_plan", "exit_loop"),
    },
    {
        "name": "metadata_agent",
        "summary": "Write metadata for the chosen model.",
        "prompt": "metadata.md",
        "tools": ("write_model_metadata",),
    },
    {
        "name": "serving_agent",
        "summary": "Publish the FastAPI contract for the saved model.",
        "prompt": "serving.md",
        "tools": ("publish_serving_contract",),
    },
    {
        "name": "report_agent",
        "summary": "Write the solution report from the shared experiment state.",
        "prompt": "report.md",
        "tools": ("recall_similar_experiments", "write_solution_design"),
    },
)

PRIMARY_TOOLS = {
    "training_agent": ["run_training"],
    "evaluation_agent": [
        "evaluate_model",
        "record_improvement_plan",
        "exit_loop",
    ],
    "metadata_agent": ["write_model_metadata"],
    "serving_agent": ["publish_serving_contract"],
    "report_agent": ["recall_similar_experiments", "write_solution_design"],
}

PLANS = {
    1: (
        "The majority-class baseline is below the target. "
        "Next pass adds interaction features and trains logistic regression "
        "and a random forest in parallel."
    ),
    2: (
        "The second pass is still short of the target. "
        "Next pass adds another interaction and a gradient boosting candidate."
    ),
}
