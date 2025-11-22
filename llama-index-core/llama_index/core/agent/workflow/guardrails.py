"""Guardrail validation functions for agentic workflows."""

import asyncio
import json
import logging
import re
from pathlib import Path
from typing import List, Dict, Any, Tuple, Sequence, Optional

from llama_index.core.llms import ChatMessage, LLM
from llama_index.core.tools import AsyncBaseTool


# Template filename constants
UNNECESSARY_TOOLS_TEMPLATE_NAME = "unnecessary_tools_validation"
MISSING_TOOLS_TEMPLATE_NAME = "missing_tools_validation"
INCORRECT_ARGUMENTS_TEMPLATE_NAME = "incorrect_arguments_validation"


def _load_template(template_name: str, version: str = "v2") -> str:
    """Load a guardrail template with version support.

    Args:
        template_name: Name of template (e.g., 'unnecessary_tools_validation')
        version: Template version ('v1', 'v2', etc.). Defaults to 'v2'.

    Returns:
        Template content as string
    """
    # Try versioned template first (in v1/, v2/, etc. subdirectories)
    template_path = Path(__file__).parent / "templates" / version / f"{template_name}.txt"

    # Fallback to non-versioned template if version doesn't exist
    if not template_path.exists():
        template_path = Path(__file__).parent / "templates" / f"{template_name}.txt"

    with template_path.open("r") as f:
        return f.read()


def _format_retry_history(retry_history: Optional[List[Dict[str, Any]]]) -> str:
    """Format retry history for display in validation prompts.

    Args:
        retry_history: List of previous retry attempts with predictions and feedback

    Returns:
        Formatted string showing retry history, or message if no history
    """
    if not retry_history:
        return "(No previous retry attempts)"

    formatted_lines = []
    for entry in retry_history:
        attempt_num = entry.get("attempt", 0)
        predicted_tools = entry.get("predicted_tools", [])
        predicted_response = entry.get("predicted_response", "")
        feedback = entry.get("guardrail_feedback", {})

        formatted_lines.append(f"Attempt {attempt_num}:")

        # Show what was predicted
        if predicted_tools:
            formatted_lines.append(f"  Predicted Tools:")
            for tool_entry in predicted_tools:
                if isinstance(tool_entry, dict):
                    tool_name = tool_entry.get("name", "unknown")
                    tool_args = tool_entry.get("arguments", {})
                    args_str = json.dumps(tool_args) if tool_args else "{}"
                    formatted_lines.append(f"    - {tool_name}({args_str})")
                else:
                    # Fallback for string format
                    formatted_lines.append(f"    - {tool_entry}")
        if predicted_response:
            formatted_lines.append(f"  Predicted Response: {predicted_response}")

        # Show feedback from each guardrail
        for guardrail_name, issues in feedback.items():
            if issues:
                formatted_lines.append(f"  {guardrail_name}:")
                for issue in issues:
                    formatted_lines.append(f"    - {issue}")

        formatted_lines.append("")  # Empty line between attempts

    return "\n".join(formatted_lines)


def _format_conversation_context(
    conversation_history: List[ChatMessage],
    available_tools: Sequence[AsyncBaseTool],
    tool_calls: List[Any],
) -> Dict[str, Any]:
    """
    Common helper to format conversation context for guardrail validation.

    Assumes single-turn structure:
    - conversation_history[0]: system message
    - conversation_history[1]: user message (current request)
    - conversation_history[2+]: tool calls and results

    Args:
        conversation_history: Full conversation history
        available_tools: Available tools
        tool_calls: Current tool calls to be predicted

    Returns:
        Dictionary with:
        - tool_descriptions: Dict[tool_name -> description]
        - system_message: System message content
        - current_request: User request content
        - previous_tool_calls: Dict[tool_name -> result]
        - predicted_tool_calls: Dict[tool_name -> tool_kwargs]
    """
    # Format tool descriptions as dictionary
    tool_descriptions = {}
    for tool in available_tools:
        description = tool.metadata.description
        # Split by newline and skip first line (function signature)
        lines = description.split('\n')
        cleaned_description = '\n'.join(lines[1:]) if len(lines) > 1 else description
        tool_descriptions[tool.metadata.name] = cleaned_description.strip()

    # Extract system message (first message)
    system_message = ""
    if len(conversation_history) > 0 and conversation_history[0].role == "system":
        system_message = conversation_history[0].content or ""

    # Extract user message (second message)
    current_request = ""
    if len(conversation_history) > 1 and conversation_history[1].role == "user":
        current_request = conversation_history[1].content or ""

    # Extract already executed tools (everything after user message)
    previous_tool_calls = {}
    if len(conversation_history) > 2:
        messages = conversation_history[2:]

        # Build mapping of tool_call_id -> tool_name from assistant messages
        tool_id_to_name = {}
        for msg in messages:
            if msg.role == "assistant":
                if hasattr(msg, 'blocks') and msg.blocks:
                    tool_call_blocks = [b for b in msg.blocks if hasattr(b, 'tool_name')]
                    for block in tool_call_blocks:
                        tool_id = getattr(block, 'tool_call_id', None)
                        tool_name = block.tool_name
                        if tool_id:
                            tool_id_to_name[tool_id] = tool_name

        # Extract tool results (store as lists to handle multiple calls of same tool)
        for msg in messages:
            if msg.role == "tool":
                tool_id = msg.additional_kwargs.get('tool_call_id', 'unknown')
                tool_name = tool_id_to_name.get(tool_id, 'unknown')
                result = msg.content or str(msg.blocks) if hasattr(msg, 'blocks') and msg.blocks else ""
                if tool_name not in previous_tool_calls:
                    previous_tool_calls[tool_name] = []
                previous_tool_calls[tool_name].append(result)

    # Format predicted tool calls as dictionary of lists (to handle multiple calls of same tool)
    predicted_tool_calls = {}
    for tc in tool_calls:
        if hasattr(tc, 'tool_name'):
            if tc.tool_name not in predicted_tool_calls:
                predicted_tool_calls[tc.tool_name] = []
            tool_kwargs = tc.tool_kwargs if hasattr(tc, 'tool_kwargs') else {}
            predicted_tool_calls[tc.tool_name].append(tool_kwargs)

    return {
        "tool_descriptions": tool_descriptions,
        "system_message": system_message,
        "current_request": current_request,
        "previous_tool_calls": previous_tool_calls,
        "predicted_tool_calls": predicted_tool_calls,
    }


def _extract_json_from_response(response_text: str, required_field: str) -> Dict[str, Any]:
    """
    Extract JSON from LLM response, handling markdown code blocks.

    Tries JSON blocks in reverse order (last to first) to get the final/corrected version.

    Args:
        response_text: The LLM response text
        required_field: A field that must exist in the JSON for it to be valid

    Returns:
        Parsed JSON dictionary

    Raises:
        ValueError: If no valid JSON found with required field
    """
    # Extract JSON from markdown code blocks using regex
    json_blocks = re.findall(r'```(?:json)?\s*([\s\S]*?)\s*```', response_text, re.MULTILINE)

    # Try each block in reverse order (last to first - LLM's final answer)
    for block in reversed(json_blocks):
        try:
            parsed = json.loads(block)
            if required_field in parsed:
                return parsed
        except json.JSONDecodeError:
            continue

    # If no code blocks worked, try parsing entire response
    try:
        parsed = json.loads(response_text.strip())
        if required_field in parsed:
            return parsed
    except json.JSONDecodeError:
        pass

    raise ValueError(f"No valid JSON found with required field '{required_field}'")


async def check_for_unnecessary_tools(
    llm: LLM,
    tool_calls: List[Any],
    conversation_history: List[ChatMessage],
    available_tools: Sequence[AsyncBaseTool],
    template_version: str = "v2",
    retry_history: Optional[List[Dict[str, Any]]] = None,
) -> Tuple[bool, Optional[Dict[str, Any]]]:
    """
    Guardrail 1: Check for unnecessary tools in the prediction.

    Uses an LLM to validate whether all predicted tool calls are necessary
    given the conversation context and available tools.

    Args:
        llm: The LLM instance to use for validation
        tool_calls: List of tool calls predicted by the agent
        conversation_history: Full conversation history including tool results
        available_tools: Tools available to the agent
        template_version: Template version to use ('v1', 'v2', etc.). Defaults to 'v2'.
        retry_history: History of previous retry attempts with predictions and feedback

    Returns:
        Tuple of (is_valid, feedback)
        - is_valid: True if all tool calls are necessary (or if guardrail fails), False otherwise
        - feedback: Dict with unnecessary_tools if issues found, None otherwise
    """
    logger = logging.getLogger(__name__)

    if not tool_calls:
        # No tool calls to validate
        return True, None

    # Use helper to format conversation context
    context = _format_conversation_context(
        conversation_history=conversation_history,
        available_tools=available_tools,
        tool_calls=tool_calls,
    )

    # Format tool descriptions for prompt with tool names
    tools_text = "\n\n".join([
        f"{name}:\n{desc}"
        for name, desc in context["tool_descriptions"].items()
    ])

    # Format previous tool calls for prompt (handle multiple calls of same tool)
    if context["previous_tool_calls"]:
        formatted_calls = []
        for name, results_list in context["previous_tool_calls"].items():
            for idx, result in enumerate(results_list):
                call_identifier = f"{name}[{idx}]" if len(results_list) > 1 else name
                formatted_calls.append(f"TOOL: {call_identifier}\nRESULT: {result}\n")
        previous_tool_calls_with_results = "\n".join(formatted_calls)
    else:
        previous_tool_calls_with_results = "(No tools executed yet)"

    # Format predicted tool calls for prompt (show count for multiple calls)
    predicted_tool_names = []
    for tool_name, calls_list in context["predicted_tool_calls"].items():
        count = len(calls_list)
        if count > 1:
            predicted_tool_names.append(f"{tool_name} ({count} times)")
        else:
            predicted_tool_names.append(tool_name)
    predicted_tool_calls = ", ".join(predicted_tool_names)

    # Load template with specified version
    template = _load_template(UNNECESSARY_TOOLS_TEMPLATE_NAME, template_version)

    # Format retry history
    retry_history_text = _format_retry_history(retry_history)

    # Construct validation prompt from template
    validation_prompt = template.format(
        system_message=context["system_message"],
        tools_text=tools_text,
        current_request=context["current_request"],
        previous_tool_calls_with_results=previous_tool_calls_with_results,
        retry_history=retry_history_text,
        predicted_tool_calls=predicted_tool_calls
    )

    # Call LLM for validation
    try:
        validation_message = ChatMessage(role="user", content=validation_prompt)
        response = await llm.achat(messages=[validation_message])

        response_text = response.message.content
        if not response_text:
            raise ValueError("Empty response from validation LLM")

        # Extract and parse JSON from response
        result = _extract_json_from_response(response_text, "output")
        output = result.get("output", {})
        is_valid = output.get("is_valid", True)
        issues = output.get("issues", [])

        if is_valid:
            logger.info("Guardrail 1: All tool calls validated as necessary")
            return True, None
        else:
            logger.warning(
                f"Guardrail 1: Found {len(issues)} unnecessary tool(s)"
            )
            return False, {"issues": issues}

    except Exception as e:
        error_msg = f"Validation error: {str(e)}"
        logger.error(f"Guardrail 1: {error_msg}, proceeding with original prediction (fail-open)")
        return True, None


async def check_for_missing_tools(
    llm: LLM,
    tool_calls: List[Any],
    conversation_history: List[ChatMessage],
    available_tools: Sequence[AsyncBaseTool],
    predicted_response: Optional[str] = None,
    template_version: str = "v2",
    retry_history: Optional[List[Dict[str, Any]]] = None,
) -> Tuple[bool, Optional[Dict[str, Any]]]:
    """
    Guardrail 2: Check for missing tools that should be in the prediction.

    Uses an LLM to validate whether all necessary tool calls are present
    given the conversation context and available tools.

    Args:
        llm: The LLM instance to use for validation
        tool_calls: List of tool calls predicted by the agent
        conversation_history: Full conversation history including tool results
        available_tools: Tools available to the agent
        predicted_response: The text response predicted by the agent (if any)
        template_version: Template version to use ('v1', 'v2', etc.). Defaults to 'v2'.
        retry_history: History of previous retry attempts with predictions and feedback

    Returns:
        Tuple of (is_valid, feedback)
        - is_valid: True if no tools are missing (or if guardrail fails), False otherwise
        - feedback: Dict with missing_tools if issues found, None otherwise
    """
    logger = logging.getLogger(__name__)

    # Use helper to format conversation context
    context = _format_conversation_context(
        conversation_history=conversation_history,
        available_tools=available_tools,
        tool_calls=tool_calls,
    )

    # Format tool descriptions for prompt with tool names
    tools_text = "\n\n".join([
        f"{name}:\n{desc}"
        for name, desc in context["tool_descriptions"].items()
    ])

    # Format previous tool calls for prompt (handle multiple calls of same tool)
    if context["previous_tool_calls"]:
        formatted_calls = []
        for name, results_list in context["previous_tool_calls"].items():
            for idx, result in enumerate(results_list):
                call_identifier = f"{name}[{idx}]" if len(results_list) > 1 else name
                formatted_calls.append(f"TOOL: {call_identifier}\nRESULT: {result}\n")
        previous_tool_calls_with_results = "\n".join(formatted_calls)
    else:
        previous_tool_calls_with_results = "(No tools executed yet)"

    # Format predicted tool calls for prompt (show count for multiple calls)
    predicted_tool_names = []
    for tool_name, calls_list in context["predicted_tool_calls"].items():
        count = len(calls_list)
        if count > 1:
            predicted_tool_names.append(f"{tool_name} ({count} times)")
        else:
            predicted_tool_names.append(tool_name)
    predicted_tool_calls = ", ".join(predicted_tool_names)

    # Load template with specified version
    template = _load_template(MISSING_TOOLS_TEMPLATE_NAME, template_version)

    # Format retry history
    retry_history_text = _format_retry_history(retry_history)

    # Construct validation prompt
    validation_prompt = template.format(
        system_message=context["system_message"],
        tools_text=tools_text,
        current_request=context["current_request"],
        previous_tool_calls_with_results=previous_tool_calls_with_results,
        retry_history=retry_history_text,
        predicted_tool_calls=predicted_tool_calls,
        predicted_response=predicted_response or "(No text response predicted)"
    )

    # Call LLM for validation
    try:
        validation_message = ChatMessage(role="user", content=validation_prompt)
        response = await llm.achat(messages=[validation_message])

        response_text = response.message.content
        if not response_text:
            raise ValueError("Empty response from validation LLM")

        # Extract and parse JSON
        result = _extract_json_from_response(response_text, "output")
        output = result.get("output", {})
        is_valid = output.get("is_valid", True)
        issues = output.get("issues", [])

        if is_valid:
            logger.info("Guardrail 2: No missing tools detected")
            return True, None
        else:
            logger.warning(
                f"Guardrail 2: Found {len(issues)} missing tool(s)"
            )
            return False, {"issues": issues}

    except Exception as e:
        error_msg = f"Validation error: {str(e)}"
        logger.error(f"Guardrail 2: {error_msg}, proceeding with original prediction (fail-open)")
        return True, None


async def check_for_incorrect_arguments(
    llm: LLM,
    tool_calls: List[Any],
    conversation_history: List[ChatMessage],
    available_tools: Sequence[AsyncBaseTool],
    template_version: str = "v2",
) -> Tuple[bool, Optional[Dict[str, Any]]]:
    """
    Guardrail 3: Check for incorrect arguments in tool calls.

    Validates each tool call separately against its description.

    Args:
        llm: The LLM instance to use for validation
        tool_calls: List of tool calls predicted by the agent
        conversation_history: Full conversation history including tool results
        available_tools: Tools available to the agent
        template_version: Template version to use ('v1', 'v2', etc.). Defaults to 'v2'.

    Returns:
        Tuple of (is_valid, feedback)
        - is_valid: True if all arguments are correct (or if guardrail fails), False otherwise
        - feedback: Dict with incorrect_arguments if issues found, None otherwise
    """
    logger = logging.getLogger(__name__)

    if not tool_calls:
        return True, None

    # Get context from helper
    context = _format_conversation_context(
        conversation_history=conversation_history,
        available_tools=available_tools,
        tool_calls=tool_calls,
    )

    # Format shared context once
    system_message = context["system_message"]
    current_request = context["current_request"]

    # Format previous tool calls for prompt (handle multiple calls of same tool)
    if context["previous_tool_calls"]:
        formatted_calls = []
        for name, results_list in context["previous_tool_calls"].items():
            for idx, result in enumerate(results_list):
                call_identifier = f"{name}[{idx}]" if len(results_list) > 1 else name
                formatted_calls.append(f"TOOL: {call_identifier}\nRESULT: {result}\n")
        previous_tool_calls_with_results = "\n".join(formatted_calls)
    else:
        previous_tool_calls_with_results = "(No tools executed yet)"

    # Load template with specified version
    template = _load_template(INCORRECT_ARGUMENTS_TEMPLATE_NAME, template_version)

    # Define inner async function to validate a single tool call
    async def validate_single_call(tool_name: str, idx: int, tool_kwargs: Dict[str, Any], calls_count: int) -> Tuple[str, Optional[List[str]]]:
        """
        Validate a single tool call.

        Returns:
            Tuple of (call_identifier, issues) where issues is None if valid
        """
        # Get THIS tool's description and format with tool name
        description = context["tool_descriptions"].get(tool_name, "Unknown tool")
        tool_description = f"{tool_name}:\n{description}"

        # Format THIS tool's call with arguments
        call_identifier = f"{tool_name}[{idx}]" if calls_count > 1 else tool_name
        tool_call_with_args = f"{tool_name}({json.dumps(tool_kwargs)})"

        # Build validation prompt for THIS tool only
        validation_prompt = template.format(
            system_message=system_message,
            tool_description=tool_description,
            current_request=current_request,
            previous_tool_calls_with_results=previous_tool_calls_with_results,
            tool_call_with_args=tool_call_with_args,
        )

        # Validate THIS tool
        try:
            validation_message = ChatMessage(role="user", content=validation_prompt)
            response = await llm.achat(messages=[validation_message])

            response_text = response.message.content
            if not response_text:
                raise ValueError("Empty response from validation LLM")

            # Extract and parse JSON
            result = _extract_json_from_response(response_text, "output")
            output = result.get("output", {})
            is_valid = output.get("is_valid", True)
            issues = output.get("issues", [])

            # Check if issues were found
            if is_valid:
                logger.info(f"Guardrail 3: Arguments for {call_identifier} are correct")
                return call_identifier, None
            else:
                # Issues found - return them
                logger.warning(f"Guardrail 3: Found issues with {call_identifier} arguments")
                return call_identifier, issues

        except Exception as e:
            error_msg = f"Validation error for {call_identifier}: {str(e)}"
            logger.error(f"Guardrail 3: {error_msg}, proceeding with original (fail-open)")
            return call_identifier, None

    # Collect all validation tasks across all tools and calls
    validation_tasks = []
    for tool_name, calls_list in context["predicted_tool_calls"].items():
        calls_count = len(calls_list)
        for idx, tool_kwargs in enumerate(calls_list):
            task = validate_single_call(tool_name, idx, tool_kwargs, calls_count)
            validation_tasks.append(task)

    # Run all validations in parallel
    validation_results = await asyncio.gather(*validation_tasks)

    # Collect all incorrect arguments from results
    all_incorrect_arguments = {}
    for call_identifier, issues in validation_results:
        if issues is not None:
            all_incorrect_arguments[call_identifier] = issues

    # Return combined results
    is_valid = len(all_incorrect_arguments) == 0
    if is_valid:
        logger.info("Guardrail 3: All tool arguments are correct")
        return True, None
    else:
        # Flatten all issues into a single list with tool names
        all_issues = []
        for tool_name, tool_issues in all_incorrect_arguments.items():
            for issue in tool_issues:
                all_issues.append(f"{tool_name}: {issue}")

        logger.warning(
            f"Guardrail 3: Found {len(all_incorrect_arguments)} tool(s) with incorrect arguments"
        )
        return False, {"issues": all_issues}
