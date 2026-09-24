"""Data Engineer.

Takes the raw table the researcher described and makes it usable: drop
duplicate ids, drop unlabeled rows, fill remaining gaps, and optionally run a
short pandas snippet under policy. Also owns the synthetic sample used when
nobody uploads a file.
"""

NAME = "data_engineer"
TITLE = "Data Engineer"
SUMMARY = "Clean the table before modeling."
PROMPT = "system.md"
TOOLS = ("clean_dataset", "run_analysis_snippet")
PRIMARY_TOOLS = {
    NAME: ["clean_dataset"],
}
