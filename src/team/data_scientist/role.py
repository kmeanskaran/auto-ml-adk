"""Data Scientist.

Turns the cleaned table into a feature spec and recommends two model families.
The SME picks one. Feature refresh in the improvement loop reuses the spec tool.
"""

NAME = "data_scientist"
TITLE = "Data Scientist"
SUMMARY = "Data preparation, features, and model choice."
SPECIALISTS = (
    {
        "name": "feature_agent",
        "summary": "Create the feature pipeline for this strategy level.",
        "prompt": "features.md",
        "tools": ("create_features",),
    },
    {
        "name": "feature_refresh_agent",
        "summary": "Refresh features after the strategy level changes.",
        "prompt": "features.md",
        "tools": ("create_features",),
    },
    {
        "name": "model_selector",
        "summary": "Choose which model families to train.",
        "prompt": "selection.md",
        "tools": ("select_models",),
    },
)

PRIMARY_TOOLS = {
    specialist["name"]: list(specialist["tools"]) for specialist in SPECIALISTS
}
