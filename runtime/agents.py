"""ADK workflow for one ML experiment.

Profile agents run together. Cleaning is next. Feature engineering, training,
and evaluation then repeat until the metric is met or the loop budget ends.
A design agent writes the solution the human can iterate on.

SequentialAgent, ParallelAgent, and LoopAgent are the workflow agents from the
tutorial. ADK marks them deprecated in favor of Workflow, which cannot yet sit
inside this tree, so the PoC stays on the workflow agents that run today.
"""

from __future__ import annotations

from typing import Any

from google.adk.agents import LlmAgent
from google.adk.agents import LoopAgent
from google.adk.agents import ParallelAgent
from google.adk.agents import SequentialAgent
from google.adk.models.base_llm import BaseLlm
from google.adk.tools.exit_loop_tool import exit_loop
from google.adk.tools.function_tool import FunctionTool
from google.genai import types

from runtime.guidance import on_agent_start
from runtime.guidance import on_before_model
from runtime.scripted_model import ScriptedLlm
from team.config import load_config
from team.data_engineer.role import NAME as DATA_ENGINEER
from team.data_engineer.role import SUMMARY as DATA_ENGINEER_SUMMARY
from team.data_engineer.tools import clean_dataset
from team.data_engineer.tools import run_analysis_snippet
from team.data_scientist.tools import create_features
from team.data_scientist.tools import select_models
from team.ml_engineer.role import NAME as ML_ENGINEER
from team.ml_engineer.role import SUMMARY as ML_ENGINEER_SUMMARY
from team.ml_engineer.tools import evaluate_model
from team.ml_engineer.tools import publish_serving_contract
from team.ml_engineer.tools import recall_similar_experiments
from team.ml_engineer.tools import record_improvement_plan
from team.ml_engineer.tools import run_training
from team.ml_engineer.tools import training_requires_approval
from team.ml_engineer.tools import write_model_metadata
from team.ml_engineer.tools import write_solution_design
from team.product_manager.role import NAME as PRODUCT_MANAGER
from team.product_manager.role import SUMMARY as PRODUCT_MANAGER_SUMMARY
from team.product_manager.tools import frame_technical_spec
from team.prompts import load_prompt
from team.researcher.role import NAME as RESEARCHER
from team.researcher.role import SUMMARY as RESEARCHER_SUMMARY
from team.researcher.tools import analyze_target
from team.researcher.tools import compile_research_brief
from team.researcher.tools import inspect_schema
from team.researcher.tools import profile_quality
from team.researcher.tools import write_business_brief


def resolve_model(name: str) -> BaseLlm:
  """Build the ADK model. ``scripted`` stays inside the process."""
  if name == "scripted":
    return ScriptedLlm(model="scripted-ml")
  from google.adk.models.lite_llm import LiteLlm

  if name == "ollama":
    name = load_config().litellm_model
  return LiteLlm(model=name)


def build_orchestrator(model: BaseLlm) -> SequentialAgent:
  """Full ML team. CLI and unstaged API runs use this tree."""
  team = assemble_team(model)
  ml_engineer = SequentialAgent(
      name=ML_ENGINEER,
      description=ML_ENGINEER_SUMMARY,
      sub_agents=[
          team["improvement"],
          team["delivery_panel"],
          team["report_agent"],
      ],
  )
  return SequentialAgent(
      name="ml_team",
      description="Product manager, researcher, data engineer, data scientist, and ML engineer.",
      sub_agents=[
          team["product_manager"],
          team["researcher"],
          team["data_engineer"],
          team["data_scientist"],
          ml_engineer,
      ],
  )


def build_stage_agent(model: BaseLlm, stage_id: str) -> SequentialAgent | LlmAgent | ParallelAgent | LoopAgent:
  """One lifecycle block for the SME board."""
  team = assemble_team(model)
  if stage_id == "stats":
    return SequentialAgent(
        name="stats_stage",
        description="Frame the task and write the statistical report.",
        sub_agents=[team["product_manager"], team["researcher"]],
    )
  if stage_id == "prepare":
    return team["data_engineer"]
  if stage_id == "model":
    return team["data_scientist"]
  if stage_id == "train":
    return team["improvement"]
  if stage_id == "deliver":
    return SequentialAgent(
        name="deliver_stage",
        description="Metadata, serving contract, and the solution note.",
        sub_agents=[team["delivery_panel"], team["report_agent"]],
    )
  raise ValueError(f"Unknown stage {stage_id!r}.")


def assemble_team(model: BaseLlm) -> dict[str, Any]:
  """Build the specialists once. Callers compose a full tree or a single stage."""
  cfg = load_config()
  content = types.GenerateContentConfig(temperature=cfg.provider.temperature)
  product_manager = _specialist(
      model,
      PRODUCT_MANAGER,
      PRODUCT_MANAGER_SUMMARY,
      load_prompt("team.product_manager", "system.md"),
      [frame_technical_spec],
      content,
  )
  research_panel = ParallelAgent(
      name="research_panel",
      description="Schema, quality, target, and business reading at the same time.",
      sub_agents=[
          _specialist(
              model,
              "schema_agent",
              "Read the dataset schema.",
              load_prompt("team.researcher", "schema.md"),
              [inspect_schema],
              content,
          ),
          _specialist(
              model,
              "quality_agent",
              "Profile data quality.",
              load_prompt("team.researcher", "quality.md"),
              [profile_quality],
              content,
          ),
          _specialist(
              model,
              "target_agent",
              "Describe the prediction target.",
              load_prompt("team.researcher", "target.md"),
              [analyze_target],
              content,
          ),
          _specialist(
              model,
              "business_analyst",
              "Connect the business problem to the columns.",
              load_prompt("team.researcher", "business.md"),
              [write_business_brief],
              content,
          ),
      ],
  )
  research_lead = _specialist(
      model,
      "research_lead",
      "Assemble the statistical report.",
      load_prompt("team.researcher", "lead.md"),
      [compile_research_brief],
      content,
  )
  researcher = SequentialAgent(
      name=RESEARCHER,
      description=RESEARCHER_SUMMARY,
      sub_agents=[research_panel, research_lead],
  )
  data_engineer = _specialist(
      model,
      DATA_ENGINEER,
      DATA_ENGINEER_SUMMARY,
      load_prompt("team.data_engineer", "system.md"),
      [
          clean_dataset,
          FunctionTool(run_analysis_snippet, require_confirmation=True),
      ],
      content,
  )
  feature_agent = _specialist(
      model,
      "feature_agent",
      "Create the feature pipeline for this strategy level.",
      load_prompt("team.data_scientist", "features.md"),
      [create_features],
      content,
  )
  model_selector = _specialist(
      model,
      "model_selector",
      "Choose which model families to train.",
      load_prompt("team.data_scientist", "selection.md"),
      [select_models],
      content,
  )
  data_scientist = ParallelAgent(
      name="data_scientist",
      description="Build features and choose models together.",
      sub_agents=[feature_agent, model_selector],
  )
  feature_refresh = _specialist(
      model,
      "feature_refresh_agent",
      "Refresh features after the strategy level changes.",
      load_prompt("team.data_scientist", "features.md"),
      [create_features],
      content,
  )
  training = _specialist(
      model,
      "training_agent",
      "Fit the selected candidates and keep the best F1.",
      load_prompt("team.ml_engineer", "training.md"),
      [FunctionTool(run_training, require_confirmation=training_requires_approval)],
      content,
  )
  evaluation = _specialist(
      model,
      "evaluation_agent",
      "Decide whether the metric is good enough.",
      load_prompt("team.ml_engineer", "evaluation.md"),
      [evaluate_model, record_improvement_plan, exit_loop],
      content,
  )
  improvement = LoopAgent(
      name="improvement_loop",
      description="Refresh features, train, and evaluate until the target is met.",
      max_iterations=cfg.runtime.max_improvement_loops,
      sub_agents=[feature_refresh, training, evaluation],
  )
  metadata_agent = _specialist(
      model,
      "metadata_agent",
      "Write metadata for the chosen model.",
      load_prompt("team.ml_engineer", "metadata.md"),
      [write_model_metadata],
      content,
  )
  serving_agent = _specialist(
      model,
      "serving_agent",
      "Publish the FastAPI contract for the saved model.",
      load_prompt("team.ml_engineer", "serving.md"),
      [publish_serving_contract],
      content,
  )
  report_agent = _specialist(
      model,
      "report_agent",
      "Write the solution report from the shared experiment state.",
      load_prompt("team.ml_engineer", "report.md"),
      [recall_similar_experiments, write_solution_design],
      content,
  )
  delivery_panel = ParallelAgent(
      name="delivery_panel",
      description="Metadata and the serving contract together.",
      sub_agents=[metadata_agent, serving_agent],
  )
  return {
      "product_manager": product_manager,
      "researcher": researcher,
      "data_engineer": data_engineer,
      "data_scientist": data_scientist,
      "improvement": improvement,
      "delivery_panel": delivery_panel,
      "report_agent": report_agent,
  }


def _specialist(
    model: BaseLlm,
    name: str,
    description: str,
    instruction: str,
    tools: list,
    generate_content_config: types.GenerateContentConfig,
) -> LlmAgent:
  return LlmAgent(
      name=name,
      model=model,
      description=description,
      instruction=instruction,
      tools=tools,
      generate_content_config=generate_content_config,
      before_agent_callback=on_agent_start,
      before_model_callback=on_before_model,
      disallow_transfer_to_parent=True,
      disallow_transfer_to_peers=True,
  )
