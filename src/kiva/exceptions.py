"""Exception definitions for the Kiva SDK.

This module provides a hierarchy of exceptions for error handling:

- SDKError: Base exception for all SDK errors
- ConfigurationError: Invalid configuration or setup
- AgentError: Agent execution failures with recovery suggestions
- WorkflowError: Workflow-level execution failures
"""


class SDKError(Exception):
    """Base exception for all Kiva SDK errors."""


class ConfigurationError(SDKError):
    """Raised when SDK configuration is invalid or incomplete.

    Examples:
        - Empty agents list
        - Missing required API credentials
        - Invalid model name
    """


class AgentError(SDKError):
    """Agent execution error with context and recovery suggestions.

    Provides detailed error information including the original exception,
    the agent that failed, and actionable recovery suggestions.

    Attributes:
        agent_id: Identifier of the agent that failed.
        original_error: The underlying exception that caused the failure.
        recovery_suggestion: Actionable suggestion for resolving the error.

    Example:
        >>> try:
        ...     await agent.ainvoke(...)
        ... except Exception as e:
        ...     raise AgentError(
        ...         message="Agent failed to process task",
        ...         agent_id="calculator",
        ...         original_error=e,
        ...     )
    """

    def __init__(
        self,
        message: str,
        agent_id: str,
        original_error: Exception | None = None,
        recovery_suggestion: str | None = None,
    ):
        """Initialize an AgentError.

        Args:
            message: Human-readable error description.
            agent_id: Identifier of the failed agent.
            original_error: The underlying exception, if any.
            recovery_suggestion: Custom recovery suggestion. If not provided,
                one will be generated based on the error type.
        """
        self.agent_id = agent_id
        self.original_error = original_error
        self.recovery_suggestion = recovery_suggestion or self._suggest_recovery(
            message, agent_id, original_error
        )
        super().__init__(f"{message}\n\nRecovery: {self.recovery_suggestion}")

    @staticmethod
    def _suggest_recovery(
        message: str, agent_id: str, original_error: Exception | None
    ) -> str:
        """Generate a recovery suggestion based on error type.

        Analyzes the error message and original exception to provide
        contextual recovery advice.

        Args:
            message: The error message.
            agent_id: The failed agent's identifier.
            original_error: The underlying exception.

        Returns:
            A multi-line string with numbered recovery steps.
        """
        error_str = (str(original_error) if original_error else message).lower()

        patterns = {
            ("timeout", "timed out"): (
                f"Agent '{agent_id}' timed out. Consider:\n"
                f"  1. Increasing the timeout limit\n"
                f"  2. Simplifying the task\n"
                f"  3. Breaking the task into smaller subtasks"
            ),
            ("rate limit", "429"): (
                f"Agent '{agent_id}' hit rate limits. Consider:\n"
                f"  1. Reducing max_parallel_agents\n"
                f"  2. Adding delays between requests\n"
                f"  3. Using a different API key with higher limits"
            ),
            ("api", "connection", "network"): (
                f"Agent '{agent_id}' encountered a connection error. Consider:\n"
                f"  1. Checking your network connection\n"
                f"  2. Verifying the API endpoint (base_url)\n"
                f"  3. Checking if the API service is available"
            ),
            ("auth", "key", "401", "403"): (
                f"Agent '{agent_id}' encountered an authentication error. Consider:\n"
                f"  1. Verifying your API key is correct\n"
                f"  2. Checking if the API key has the required permissions\n"
                f"  3. Ensuring the API key hasn't expired"
            ),
            ("tool", "function"): (
                f"Agent '{agent_id}' failed during tool execution. Consider:\n"
                f"  1. Checking the tool's implementation for errors\n"
                f"  2. Verifying the tool's input parameters\n"
                f"  3. Adding error handling to the tool"
            ),
        }

        for keywords, suggestion in patterns.items():
            if any(kw in error_str for kw in keywords):
                return suggestion

        return (
            f"Agent '{agent_id}' failed. Consider:\n"
            f"  1. Checking the agent's tools and prompts\n"
            f"  2. Reviewing the agent's configuration\n"
            f"  3. Enabling debug logging for more details"
        )


class WorkflowError(SDKError):
    """Workflow execution error with context information.

    Attributes:
        workflow: The workflow type that failed (router, supervisor, parliament).
        execution_id: Unique identifier for the failed execution.
    """

    def __init__(self, message: str, workflow: str, execution_id: str):
        """Initialize a WorkflowError.

        Args:
            message: Human-readable error description.
            workflow: The workflow type that failed.
            execution_id: The execution's unique identifier.
        """
        super().__init__(message)
        self.workflow = workflow
        self.execution_id = execution_id


def wrap_agent_error(
    error: Exception, agent_id: str, task: str | None = None
) -> AgentError:
    """Wrap an exception as an AgentError with context.

    Utility function for converting arbitrary exceptions into AgentError
    instances with appropriate context and recovery suggestions.

    Args:
        error: The original exception to wrap.
        agent_id: Identifier of the agent that raised the error.
        task: The task being processed when the error occurred.

    Returns:
        An AgentError instance wrapping the original exception.

    Example:
        >>> try:
        ...     result = await agent.ainvoke(task)
        ... except Exception as e:
        ...     raise wrap_agent_error(e, "calculator", task)
    """
    if isinstance(error, AgentError):
        return error

    if task:
        if len(task) > 100:
            message = f"Agent execution failed while processing task: {task[:100]}..."
        else:
            message = f"Agent execution failed while processing task: {task}"
    else:
        message = f"Agent execution failed: {error}"

    return AgentError(message=message, agent_id=agent_id, original_error=error)
