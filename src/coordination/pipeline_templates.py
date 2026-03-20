"""Pipeline template definitions and loading logic for the sequential strategy.

Templates define ordered stages with roles, tool restrictions, and
interface contracts. Named templates (linear, v_model, mbse) encode
engineering methodology structures. Custom templates are user-defined.

Each stage specifies:
  - name: unique identifier
  - role: system prompt describing the stage's responsibility
  - allowed_tools: which domain tools the stage can access
  - interface_output: description of what this stage passes to the next
"""

from dataclasses import dataclass, field


@dataclass
class PipelineStage:
    """A single stage in a pipeline template."""

    name: str
    role: str
    allowed_tools: list[str]  # ["*"] = all, [] = none, ["tool_a"] = specific
    interface_output: str


@dataclass
class PipelineTemplate:
    """An ordered sequence of stages."""

    name: str
    stages: list[PipelineStage]
    # Keys whose values are extracted from stage outputs and injected
    # into all subsequent stages as shared state.  E.g. ["SESSION_ID"]
    # means any stage that outputs "SESSION_ID: <value>" will have that
    # value propagated forward through the pipeline.
    shared_state_keys: list[str] = field(default_factory=list)


# -- Built-in templates --------------------------------------------------------

_LINEAR_STAGES = [
    PipelineStage(
        name="planner",
        role=(
            "Analyze the task and produce a detailed plan with steps, "
            "constraints, and expected outputs."
        ),
        allowed_tools=[],
        interface_output="A structured plan",
    ),
    PipelineStage(
        name="executor",
        role=(
            "Execute the plan using your available tools. "
            "Follow the plan exactly."
        ),
        allowed_tools=["*"],
        interface_output="Execution results and any generated files",
    ),
    PipelineStage(
        name="reviewer",
        role=(
            "Review the execution results. Check for errors, completeness, "
            "and correctness. If everything is correct, output TASK_COMPLETE."
        ),
        allowed_tools=[],
        interface_output="Review verdict",
    ),
]

_V_MODEL_STAGES = [
    PipelineStage(
        name="requirements_analyst",
        role=(
            "You are a requirements analyst. Analyze the task and produce "
            "a clear, unambiguous set of requirements. Each requirement "
            "must be specific, measurable, and testable. Define acceptance "
            "criteria for each requirement."
        ),
        allowed_tools=[],
        interface_output="Numbered list of requirements with acceptance criteria",
    ),
    PipelineStage(
        name="system_designer",
        role=(
            "You are a system designer. Given the requirements, produce a "
            "high-level system architecture. Define the major components, "
            "their responsibilities, and the interfaces between them."
        ),
        allowed_tools=[],
        interface_output="System architecture with component specifications",
    ),
    PipelineStage(
        name="detailed_designer",
        role=(
            "You are a detailed designer. Given the system architecture, "
            "produce detailed specifications for each component. Define "
            "exact parameters, dimensions, constraints, and implementation "
            "approach."
        ),
        allowed_tools=[],
        interface_output="Detailed component specifications",
    ),
    PipelineStage(
        name="implementer",
        role=(
            "You are an implementer. Given the detailed design, write the "
            "code or produce the artifact exactly as specified. Follow the "
            "design precisely."
        ),
        allowed_tools=["*"],
        interface_output="Implementation artifacts and execution results",
    ),
    PipelineStage(
        name="integration_verifier",
        role=(
            "You are an integration verifier. Check that the implementation "
            "matches the detailed design and system architecture. Verify "
            "all components work together. If everything is correct, output "
            "TASK_COMPLETE."
        ),
        allowed_tools=[],
        interface_output="Verification report",
    ),
]

_MBSE_STAGES = [
    PipelineStage(
        name="stakeholder_analyst",
        role=(
            "You are a stakeholder analyst. Identify all stakeholder needs, "
            "constraints, and expectations. Translate these into formal "
            "system requirements. Establish traceability between needs and "
            "requirements."
        ),
        allowed_tools=[],
        interface_output="Stakeholder needs and traced requirements",
    ),
    PipelineStage(
        name="system_architect",
        role=(
            "You are a system architect. Define the system architecture. "
            "Identify subsystems, interfaces, and information flows. Map "
            "each requirement to at least one architectural element."
        ),
        allowed_tools=[],
        interface_output="Architecture model with requirements mapping",
    ),
    PipelineStage(
        name="subsystem_designer",
        role=(
            "You are a subsystem designer. Produce detailed designs for "
            "each subsystem. Specify parameters, algorithms, and "
            "interfaces. Maintain traceability to architectural elements."
        ),
        allowed_tools=[],
        interface_output="Detailed subsystem designs with traceability",
    ),
    PipelineStage(
        name="implementer",
        role=(
            "You are an implementer. Implement each component from the "
            "subsystem designs. Execute code, generate artifacts. Record "
            "which design element each implementation addresses."
        ),
        allowed_tools=["*"],
        interface_output="Implementation with design element mapping",
    ),
    PipelineStage(
        name="validator",
        role=(
            "You are a system validator. Verify the implementation against "
            "the stakeholder requirements. Check requirements coverage. "
            "Report gaps or failures. If all requirements are satisfied, "
            "output TASK_COMPLETE."
        ),
        allowed_tools=[],
        interface_output="Validation report with requirements coverage",
    ),
]

_BUILTIN_TEMPLATES = {
    "linear": PipelineTemplate(name="linear", stages=list(_LINEAR_STAGES)),
    "v_model": PipelineTemplate(name="v_model", stages=list(_V_MODEL_STAGES)),
    "mbse": PipelineTemplate(name="mbse", stages=list(_MBSE_STAGES)),
}

TEMPLATE_NAMES = frozenset(_BUILTIN_TEMPLATES.keys())


# -- Loading and validation ----------------------------------------------------

def load_template(
    template_name: str,
    custom_stages: list[dict] | None = None,
    templates_config: dict | None = None,
) -> PipelineTemplate:
    """Load a pipeline template by name.

    Args:
        template_name: One of "linear", "v_model", "mbse", "custom",
            or a name defined in templates_config.
        custom_stages: Stage dicts for "custom" template.
        templates_config: Dict of template_name -> {stages: [...]}
            from the agents YAML file.

    Returns:
        A validated PipelineTemplate.

    Raises:
        ValueError: If template name is unknown, stages are empty,
            or stages have duplicate names.
    """
    if template_name == "custom":
        if not custom_stages:
            raise ValueError(
                "pipeline_template is 'custom' but no custom_stages provided."
            )
        stages = _parse_stages(custom_stages)
        template = PipelineTemplate(name="custom", stages=stages)
        _validate_template(template)
        return template

    # Check built-in templates.
    if template_name in _BUILTIN_TEMPLATES:
        return _BUILTIN_TEMPLATES[template_name]

    # Check templates from config (agents YAML).
    if templates_config and template_name in templates_config:
        raw = templates_config[template_name]
        raw_stages = raw.get("stages", [])
        if not raw_stages:
            raise ValueError(
                f"Template '{template_name}' in config has empty stages."
            )
        stages = _parse_stages(raw_stages)
        shared_keys = raw.get("shared_state_keys", [])
        template = PipelineTemplate(
            name=template_name, stages=stages, shared_state_keys=shared_keys,
        )
        _validate_template(template)
        return template

    available = sorted(set(_BUILTIN_TEMPLATES.keys()) | {"custom"})
    if templates_config:
        available += sorted(templates_config.keys())
    raise ValueError(
        f"Unknown pipeline template {template_name!r}. "
        f"Available: {available}"
    )


def resolve_tools(
    allowed_tools: list[str],
    all_domain_tools: dict,
) -> list:
    """Resolve tool restriction spec to actual tool objects.

    Args:
        allowed_tools: The stage's allowed_tools list.
            ["*"] = all domain tools, [] = none, ["name"] = specific.
        all_domain_tools: Dict of tool_name -> Tool instance.

    Returns:
        List of Tool instances.

    Raises:
        ValueError: If a named tool is not in all_domain_tools.
    """
    if not allowed_tools:
        return []

    if allowed_tools == ["*"]:
        return list(all_domain_tools.values())

    resolved = []
    for name in allowed_tools:
        if name not in all_domain_tools:
            raise ValueError(
                f"Tool '{name}' not found in domain tools. "
                f"Available: {sorted(all_domain_tools.keys())}"
            )
        resolved.append(all_domain_tools[name])
    return resolved


# -- Internal helpers ----------------------------------------------------------

def _parse_stages(raw_stages: list[dict]) -> list[PipelineStage]:
    """Parse a list of stage dicts into PipelineStage objects."""
    stages = []
    for raw in raw_stages:
        stage = PipelineStage(
            name=raw.get("name", ""),
            role=raw.get("role", ""),
            allowed_tools=raw.get("allowed_tools", []),
            interface_output=raw.get("interface_output", ""),
        )
        stages.append(stage)
    return stages


def _validate_template(template: PipelineTemplate) -> None:
    """Validate a pipeline template.

    Raises:
        ValueError: If stages are empty or have duplicate names.
    """
    if not template.stages:
        raise ValueError(
            f"Pipeline template '{template.name}' has no stages."
        )

    names = [s.name for s in template.stages]
    if len(names) != len(set(names)):
        dupes = [n for n in names if names.count(n) > 1]
        raise ValueError(
            f"Pipeline template '{template.name}' has duplicate stage names: "
            f"{sorted(set(dupes))}"
        )

    for stage in template.stages:
        if not stage.name:
            raise ValueError(
                f"Pipeline template '{template.name}' has a stage with empty name."
            )
