"""Product Manager.

Owns the first conversation with the business. Reads the uploaded table only
to name the target column, then writes a one-line technical problem the rest
of the team can execute.
"""

NAME = "product_manager"
TITLE = "Product Manager"
SUMMARY = "Turn the business problem into a modeling task."
PROMPT = "system.md"
TOOLS = ("frame_technical_spec",)
