"""Researcher.

Reads the raw table without changing it. Schema, quality, target balance, and
the business reading run together; the research lead then writes one
statistical report the rest of the team shares.
"""

NAME = "researcher"
TITLE = "Researcher"
SUMMARY = "Business understanding and the statistical report."
PANEL = "research_panel"
SPECIALISTS = (
    {
        "name": "schema_agent",
        "summary": "Read the dataset schema.",
        "prompt": "schema.md",
        "tools": ("inspect_schema",),
    },
    {
        "name": "quality_agent",
        "summary": "Profile data quality.",
        "prompt": "quality.md",
        "tools": ("profile_quality",),
    },
    {
        "name": "target_agent",
        "summary": "Describe the prediction target.",
        "prompt": "target.md",
        "tools": ("analyze_target",),
    },
    {
        "name": "business_analyst",
        "summary": "Connect the business problem to the columns.",
        "prompt": "business.md",
        "tools": ("write_business_brief",),
    },
    {
        "name": "research_lead",
        "summary": "Assemble the statistical report.",
        "prompt": "lead.md",
        "tools": ("compile_research_brief",),
    },
)

PRIMARY_TOOLS = {
    specialist["name"]: list(specialist["tools"]) for specialist in SPECIALISTS
}
