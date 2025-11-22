import asyncio
import json
import logging
from pathlib import Path
from typing import List, Optional, Sequence

from llama_index.core.agent.workflow.base_agent import BaseWorkflowAgent
from llama_index.core.agent.workflow.guardrails import (
    check_for_unnecessary_tools,
    check_for_missing_tools,
    check_for_incorrect_arguments,
)
from llama_index.core.agent.workflow.workflow_events import (
    AgentInput,
    AgentOutput,
    AgentStream,
    ToolCallResult,
)
from llama_index.core.base.llms.types import ChatResponse
from llama_index.core.bridge.pydantic import BaseModel, Field
from llama_index.core.llms import ChatMessage
from llama_index.core.memory import BaseMemory
from llama_index.core.tools import AsyncBaseTool
from llama_index.core.workflow import Context

# Load retry feedback template
with (Path(__file__).parent / "templates" / "retry_feedback.txt").open("r") as f:
    RETRY_FEEDBACK_TEMPLATE = f.read()


class FunctionAgent(BaseWorkflowAgent):
    """Function calling agent implementation."""

    scratchpad_key: str = "scratchpad"
    initial_tool_choice: Optional[str] = Field(
        default=None,
        description="The tool to try and force to call on the first iteration of the agent.",
    )
    allow_parallel_tool_calls: bool = Field(
        default=True,
        description="If True, the agent will call multiple tools in parallel. If False, the agent will call tools sequentially.",
    )

    async def _get_response(
        self, current_llm_input: List[ChatMessage], tools: Sequence[AsyncBaseTool]
    ) -> ChatResponse:
        chat_kwargs = {
            "chat_history": current_llm_input,
            "allow_parallel_tool_calls": self.allow_parallel_tool_calls,
            "tools": tools,
        }

        # Only add tool choice if set and if its the first response
        if (
            self.initial_tool_choice is not None
            and current_llm_input[-1].role == "user"
        ):
            chat_kwargs["tool_choice"] = self.initial_tool_choice

        return await self.llm.achat_with_tools(  # type: ignore
            **chat_kwargs
        )

    async def _get_streaming_response(
        self,
        ctx: Context,
        current_llm_input: List[ChatMessage],
        tools: Sequence[AsyncBaseTool],
    ) -> ChatResponse:
        chat_kwargs = {
            "chat_history": current_llm_input,
            "tools": tools,
            "allow_parallel_tool_calls": self.allow_parallel_tool_calls,
        }

        # Only add tool choice if set and if its the first response
        if (
            self.initial_tool_choice is not None
            and current_llm_input[-1].role == "user"
        ):
            chat_kwargs["tool_choice"] = self.initial_tool_choice

        response = await self.llm.astream_chat_with_tools(  # type: ignore
            **chat_kwargs
        )
        # last_chat_response will be used later, after the loop.
        # We initialize it so it's valid even when 'response' is empty
        last_chat_response = ChatResponse(message=ChatMessage())
        async for last_chat_response in response:
            tool_calls = self.llm.get_tool_calls_from_response(  # type: ignore
                last_chat_response, error_on_no_tool_call=False
            )
            raw = (
                last_chat_response.raw.model_dump()
                if isinstance(last_chat_response.raw, BaseModel)
                else last_chat_response.raw
            )
            ctx.write_event_to_stream(
                AgentStream(
                    delta=last_chat_response.delta or "",
                    response=last_chat_response.message.content or "",
                    tool_calls=tool_calls or [],
                    raw=raw,
                    current_agent_name=self.name,
                    thinking_delta=last_chat_response.additional_kwargs.get(
                        "thinking_delta", None
                    ),
                )
            )

        return last_chat_response

    async def _validate_and_retry_with_guardrail(
        self,
        ctx: Context,
        current_llm_input: List[ChatMessage],
        last_chat_response: ChatResponse,
        tool_calls: List,
        tools: Sequence[AsyncBaseTool],
        scratchpad: List[ChatMessage],
    ) -> tuple[Optional[ChatResponse], Optional[List]]:
        """
        Validate tool calls and retry if unnecessary tools are detected.

        Args:
            ctx: Workflow context
            current_llm_input: Current conversation history
            last_chat_response: The LLM's response
            tool_calls: Extracted tool calls from response
            tools: Available tools
            scratchpad: Current scratchpad messages

        Returns:
            Tuple of (response, tool_calls):
                - response: Either the corrected response (if retry) or original response
                - tool_calls: Either the corrected tool calls (if retry) or original tool calls
        """
        logger = logging.getLogger(__name__)

        # Initialize with original values
        current_response = last_chat_response
        current_tool_calls = tool_calls
        retry_input = current_llm_input

        logger.info(f"####### NEW STEP BEGINS HERE#####")

        # Log context: user request and previous tool calls
        logger.info("=== CONTEXT ===")

        # Extract current user request (look for last user message before assistant predictions)
        user_request = ""
        for msg in reversed(current_llm_input):
            if msg.role == "user":
                user_request = msg.content or ""
                break
        if user_request:
            logger.info(f"User request: {user_request}")

        # Extract previous tool calls and results
        # First, build mapping of tool_call_id -> tool_name from assistant messages
        tool_id_to_name = {}
        for msg in current_llm_input:
            if msg.role == "assistant" and hasattr(msg, 'blocks') and msg.blocks:
                for block in msg.blocks:
                    if hasattr(block, 'tool_call_id') and hasattr(block, 'tool_name'):
                        tool_id_to_name[block.tool_call_id] = block.tool_name

        # Then extract tool results with proper names
        previous_tools = []
        for msg in current_llm_input:
            if msg.role == "tool":
                tool_id = msg.additional_kwargs.get('tool_call_id', 'unknown')
                result = msg.content or ""
                tool_name = tool_id_to_name.get(tool_id, "unknown")
                previous_tools.append((tool_name, result))

        if previous_tools:
            logger.info(f"Previous tool calls: {len(previous_tools)}")
            for tool_name, result in previous_tools:
                result_preview = result[:100] + "..." if len(result) > 100 else result
                logger.info(f"  - {tool_name} → {result_preview}")
        else:
            logger.info("Previous tool calls: None")

        # Log initial agent prediction (before any retries)
        logger.info("=== AGENT PREDICTION ===")
        logger.info(f"Tool calls: {len(tool_calls)}")
        for tc in tool_calls:
            logger.info(f"  - {tc.tool_name}({tc.tool_kwargs})")
        initial_predicted_response = None
        if hasattr(self.llm, 'get_response_text') and last_chat_response:
            initial_predicted_response = self.llm.get_response_text(last_chat_response)
            if initial_predicted_response:
                logger.info(f"Response text: {initial_predicted_response}")

        # Allow up to 2 retries
        MAX_RETRIES = 2

        for retry_attempt in range(MAX_RETRIES):
            try:
                # Extract text response if available
                predicted_response = None
                if hasattr(self.llm, 'get_response_text') and current_response:
                    predicted_response = self.llm.get_response_text(current_response)

                # Run all 3 guardrails in parallel
                logger.info(f"Running all guardrails in parallel on attempt {retry_attempt + 1}")

                (
                    (no_unnecessary_tools, unnecessary_tools_feedback),
                    (no_missing_tools, missing_tools_feedback),
                    (no_incorrect_arguments, incorrect_arguments_feedback),
                ) = await asyncio.gather(
                    check_for_unnecessary_tools(
                        llm=self.llm,
                        tool_calls=current_tool_calls,
                        conversation_history=retry_input,
                        available_tools=tools,
                    ),
                    check_for_missing_tools(
                        llm=self.llm,
                        tool_calls=current_tool_calls,
                        conversation_history=retry_input,
                        available_tools=tools,
                        predicted_response=predicted_response,
                    ),
                    check_for_incorrect_arguments(
                        llm=self.llm,
                        tool_calls=current_tool_calls,
                        conversation_history=retry_input,
                        available_tools=tools,
                    ),
                )

                # Check if all guardrails passed
                if no_unnecessary_tools and no_missing_tools and no_incorrect_arguments:
                    logger.info(f"Guardrails: All validations passed on attempt {retry_attempt + 1}\n\n")
                    break

                # Collect all feedback from failed guardrails in logical order
                all_feedback = []

                # 1. First, show unnecessary tools (should be removed)
                if unnecessary_tools_feedback is not None:
                    unnecessary_issues = unnecessary_tools_feedback.get("issues", [])
                    if unnecessary_issues:
                        logger.info(f"Guardrail 1: Found {len(unnecessary_issues)} unnecessary tool(s)")
                        for issue in unnecessary_issues:
                            logger.info(f"    {issue}")
                        all_feedback.append("UNNECESSARY TOOLS (should be removed):")
                        for issue in unnecessary_issues:
                            all_feedback.append(f"  - {issue}")
                        all_feedback.append("")  # Empty line

                # 2. Then, show incorrect arguments for remaining tools (should be fixed)
                if incorrect_arguments_feedback is not None:
                    argument_issues = incorrect_arguments_feedback.get("issues", [])
                    if argument_issues:
                        logger.info(f"Guardrail 3: Found {len(argument_issues)} tool(s) with incorrect arguments")
                        for issue in argument_issues:
                            logger.info(f"    {issue}")
                        all_feedback.append("INCORRECT ARGUMENTS (fix arguments for these tools):")
                        for issue in argument_issues:
                            all_feedback.append(f"  - {issue}")
                        all_feedback.append("")  # Empty line

                # 3. Finally, show missing tools (should be added)
                if missing_tools_feedback is not None:
                    missing_issues = missing_tools_feedback.get("issues", [])
                    if missing_issues:
                        logger.info(f"Guardrail 2: Found {len(missing_issues)} missing tool(s)")
                        for issue in missing_issues:
                            logger.info(f"    {issue}")
                        all_feedback.append("MISSING TOOLS (should be added):")
                        for issue in missing_issues:
                            all_feedback.append(f"  - {issue}")
                        all_feedback.append("")  # Empty line

                logger.info(
                    f"Guardrails: Retry attempt {retry_attempt + 1}/{MAX_RETRIES} - "
                    f"found issues"
                )

                # Construct feedback content (without IMPORTANT note - that's in template)
                feedback_content = "\n".join(all_feedback)

                # Extract agent's predicted response text
                predicted_response_text = ""
                if hasattr(self.llm, 'get_response_text') and current_response:
                    predicted_response_text = self.llm.get_response_text(current_response)

                # Format predicted response section
                predicted_response_section = ""
                if predicted_response_text:
                    predicted_response_section = f"YOUR RESPONSE:\n{predicted_response_text}\n"

                # Format predicted tool calls section with full arguments
                predicted_tools_section = ""
                if current_tool_calls:
                    tool_calls_formatted = []
                    for tc in current_tool_calls:
                        args_str = json.dumps(tc.tool_kwargs, indent=2)
                        tool_calls_formatted.append(f"  • {tc.tool_name}(\n{args_str}\n  )")

                    predicted_tools_section = "YOUR TOOL CALLS:\n" + "\n".join(tool_calls_formatted)

                # Format complete feedback message using template
                feedback_content_formatted = RETRY_FEEDBACK_TEMPLATE.format(
                    predicted_response_section=predicted_response_section,
                    predicted_tools_section=predicted_tools_section,
                    feedback=feedback_content
                )

                # Create single user message with complete context
                combined_feedback_message = ChatMessage(
                    role="user",
                    content=feedback_content_formatted
                )

                # Add combined feedback to input (single user turn with all context)
                retry_input = [*retry_input, combined_feedback_message]
                ctx.write_event_to_stream(
                    AgentInput(input=retry_input, current_agent_name=self.name)
                )
                try:
                    # Retry
                    if self.streaming:
                        retry_response = await self._get_streaming_response(
                            ctx, retry_input, tools
                        )
                    else:
                        retry_response = await self._get_response(retry_input, tools)

                    # Extract new tool calls from retry
                    retry_tool_calls = self.llm.get_tool_calls_from_response(
                        retry_response, error_on_no_tool_call=False
                    )

                    logger.info(
                        f"Guardrails: Retry {retry_attempt + 1} successful, got {len(retry_tool_calls) if retry_tool_calls else 0} new tool call(s)"
                    )

                    # Log what changed after retry
                    logger.info("=== AFTER RETRY ===")
                    logger.info(f"Tool calls: {len(retry_tool_calls) if retry_tool_calls else 0}")
                    if retry_tool_calls:
                        for tc in retry_tool_calls:
                            logger.info(f"  - {tc.tool_name}({tc.tool_kwargs})")
                    if hasattr(self.llm, 'get_response_text') and retry_response:
                        retry_response_text = self.llm.get_response_text(retry_response)
                        if retry_response_text:
                            logger.info(f"Response text: {retry_response_text}")

                    # Update for next iteration
                    current_response = retry_response
                    current_tool_calls = retry_tool_calls

                    # If no tool calls after retry, accept this outcome and exit
                    if not retry_tool_calls:
                        logger.info(
                            f"Guardrails: Retry {retry_attempt + 1} resulted in no tool calls, "
                            f"accepting this outcome"
                        )
                        break

                except Exception as retry_error:
                    logger.error(
                        f"Guardrails: Retry {retry_attempt + 1} failed with error: {str(retry_error)}, "
                        f"proceeding with original prediction"
                    )
                    break

            except Exception as validation_error:
                # Unexpected error in validation - fail open
                logger.error(
                    f"Guardrails: Unexpected validation error on attempt {retry_attempt + 1}: {str(validation_error)}, "
                    f"proceeding with original prediction"
                )
                break

        # Always return current state (either original or corrected after retries)
        return current_response, current_tool_calls

    async def take_step(
        self,
        ctx: Context,
        llm_input: List[ChatMessage],
        tools: Sequence[AsyncBaseTool],
        memory: BaseMemory,
    ) -> AgentOutput:
        """Take a single step with the function calling agent."""
        if not self.llm.metadata.is_function_calling_model:
            raise ValueError("LLM must be a FunctionCallingLLM")

        # Flag to enable/disable guardrails - set to False to skip validation
        enable_guardrails =  True

        scratchpad: List[ChatMessage] = await ctx.store.get(
            self.scratchpad_key, default=[]
        )
        current_llm_input = [*llm_input, *scratchpad]

        ctx.write_event_to_stream(
            AgentInput(input=current_llm_input, current_agent_name=self.name)
        )

        if self.streaming:
            last_chat_response = await self._get_streaming_response(
                ctx, current_llm_input, tools
            )
        else:
            last_chat_response = await self._get_response(current_llm_input, tools)

        tool_calls = self.llm.get_tool_calls_from_response(  # type: ignore
            last_chat_response, error_on_no_tool_call=False
        )

        # Run guardrails if enabled
        if enable_guardrails:
            # Run all guardrails: unnecessary tools, missing tools, incorrect arguments
            final_response, final_tool_calls = await self._validate_and_retry_with_guardrail(
                ctx=ctx,
                current_llm_input=current_llm_input,
                last_chat_response=last_chat_response,
                tool_calls=tool_calls,
                tools=tools,
                scratchpad=scratchpad,
            )
        else:
            # Skip guardrails - use original response
            final_response = last_chat_response
            final_tool_calls = tool_calls

        # Add final response to scratchpad (either original or corrected)
        scratchpad.append(final_response.message)
        await ctx.store.set(self.scratchpad_key, scratchpad)

        # Return agent output with final response
        raw = (
            final_response.raw.model_dump()
            if isinstance(final_response.raw, BaseModel)
            else final_response.raw
        )
        return AgentOutput(
            response=final_response.message,
            tool_calls=final_tool_calls or [],
            raw=raw,
            current_agent_name=self.name,
        )

    async def handle_tool_call_results(
        self, ctx: Context, results: List[ToolCallResult], memory: BaseMemory
    ) -> None:
        """Handle tool call results for function calling agent."""
        scratchpad: List[ChatMessage] = await ctx.store.get(
            self.scratchpad_key, default=[]
        )

        for tool_call_result in results:
            scratchpad.append(
                ChatMessage(
                    role="tool",
                    blocks=tool_call_result.tool_output.blocks,
                    additional_kwargs={"tool_call_id": tool_call_result.tool_id},
                )
            )

            if (
                tool_call_result.return_direct
                and tool_call_result.tool_name != "handoff"
            ):
                scratchpad.append(
                    ChatMessage(
                        role="assistant",
                        content=str(tool_call_result.tool_output.content),
                        additional_kwargs={"tool_call_id": tool_call_result.tool_id},
                    )
                )
                break

        await ctx.store.set(self.scratchpad_key, scratchpad)

    async def finalize(
        self, ctx: Context, output: AgentOutput, memory: BaseMemory
    ) -> AgentOutput:
        """
        Finalize the function calling agent.

        Adds all in-progress messages to memory.
        """
        scratchpad: List[ChatMessage] = await ctx.store.get(
            self.scratchpad_key, default=[]
        )
        await memory.aput_messages(scratchpad)

        # reset scratchpad
        await ctx.store.set(self.scratchpad_key, [])

        return output
