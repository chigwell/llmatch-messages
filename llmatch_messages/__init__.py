import re
import time
from typing import Any, Dict, List, Optional, Pattern, Type

from langchain_llm7 import ChatLLM7
from langchain_core.messages import HumanMessage, BaseMessage, AIMessage, SystemMessage


def llmatch(
    messages: List[BaseMessage],
    llm: Optional[ChatLLM7] = ChatLLM7(),
    pattern: Optional[str | Pattern] = "```(.*?)```",
    max_retries: int = 15,
    initial_delay: float = 1.0,
    backoff_factor: float = 1.5,
    verbose: bool = False,
    **kwargs: Any
) -> Dict[str, Any]:
    """
    Invokes an LLM with a given list of messages, retrying until a response
    matches the specified pattern or max_retries is reached.

    Args:
        messages: A list of BaseMessage objects (e.g., HumanMessage, AIMessage)
                  to be sent to the LLM. This list will be extended with
                  retry information if retries occur. Ensure messages instruct
                  the LLM on any desired output format if pattern matching is used.
        llm: An already initialized instance of the LLM. If None,
             a default ChatLLM7() instance is created.
        pattern: A regex string or compiled re.Pattern object to search for in the
                 LLM's response content. If None, the function returns the first
                 successful response without pattern matching.
        max_retries: Maximum number of attempts to make.
        initial_delay: Seconds to wait before the first retry.
        backoff_factor: Multiplier for the delay between retries (exponential backoff).
        verbose: If True, prints detailed logs of the process.
        **kwargs: Catches any other keyword arguments passed to the function for future use
                  (they are not used in the core logic currently).

    Returns:
        A dictionary containing:
        - 'success' (bool): True if a valid response (matching pattern if provided) was found.
        - 'extracted_data' (Optional[List[str]]): List of strings matching the pattern (groups),
                                                 or None if no pattern was provided or matched.
        - 'final_content' (Optional[str]): The content of the last successful or final failed LLM response.
        - 'retries_attempted' (int): Number of retries made (0 means success on first try).
        - 'error_message' (Optional[str]): Description of the error if success is False.
        - 'raw_response' (Any): The raw response object from the last LLM call (could be None if instantiation failed).
    """

    if not llm:
        try:
            llm = ChatLLM7() # Default instantiation if not provided
            if verbose:
                print("Initialized default ChatLLM7 instance.")
        except Exception as e:
            if verbose:
                print(f"Error: Failed to initialize default LLM. Error: {e}")
            return {
                "success": False, "extracted_data": None, "final_content": None,
                "retries_attempted": 0, "error_message": f"LLM initialization failed: {e}",
                "raw_response": None,
            }

    if not messages:
        if verbose:
            print("Error: No messages provided to send to the LLM.")
        return {
            "success": False, "extracted_data": None, "final_content": None,
            "retries_attempted": 0, "error_message": "No messages provided.",
            "raw_response": None,
        }

    # Ensure we are working with a mutable copy of messages
    current_messages = list(messages) # Make a shallow copy

    if pattern and isinstance(pattern, str):
        try:
            compiled_pattern = re.compile(pattern, re.DOTALL)
            if verbose:
                print(f"Compiled regex pattern: {pattern}")
        except re.error as e:
            if verbose:
                print(f"Error: Invalid regex pattern provided: {pattern}. Error: {e}")
            return {
                "success": False, "extracted_data": None, "final_content": None,
                "retries_attempted": 0, "error_message": f"Invalid regex pattern: {e}",
                "raw_response": None,
            }
    elif isinstance(pattern, re.Pattern):
        compiled_pattern = pattern
        if verbose:
            print(f"Using provided compiled regex pattern: {compiled_pattern.pattern}")
    else:
        compiled_pattern = None
        if verbose:
            print("No pattern provided. Will return first valid string response.")

    # --- Retry Loop ---
    retry_count = 0
    current_delay = initial_delay
    last_error = None
    final_content = None
    raw_response = None

    while retry_count <= max_retries:
        attempt = retry_count + 1
        if verbose:
            print(f"\n--- LLM Invocation: Attempt {attempt}/{max_retries + 1} ---")
            # To see the messages being sent (can be verbose):
            # for msg_idx, msg in enumerate(current_messages):
            #     print(f"  Message {msg_idx} ({type(msg).__name__}): {msg.content}")


        try:
            print(current_messages)
            response = llm.invoke(current_messages)

            raw_response = response

            # 1. Validate response structure
            if not response or not hasattr(response, 'content'):
                last_error = "Invalid response object structure received from LLM."
                if verbose:
                    print(f"Error: {last_error} Response: {response}")
                current_messages.append(AIMessage(content=str(response)))
                current_messages.append(HumanMessage(content=[{"type": "text", "text": f"Instruction: The previous response was invalid. Please ensure a valid response structure. Retrying due to error: {last_error}"}]))
                retry_count += 1
                if retry_count <= max_retries: time.sleep(current_delay); current_delay *= backoff_factor
                continue

            content = response.content
            final_content = content

            # 2. Validate content type
            if not isinstance(content, str):
                last_error = f"LLM response content is not a string (type: {type(content)})."
                if verbose:
                    print(f"Error: {last_error} Content: {content}")
                current_messages.append(AIMessage(content=str(content))) # AIMessage content is typically string
                current_messages.append(HumanMessage(content=[{"type": "text", "text": f"Instruction: The previous response content was not a string. Please provide a string response. Retrying due to error: {last_error}"}]))
                retry_count += 1
                if retry_count <= max_retries: time.sleep(current_delay); current_delay *= backoff_factor
                continue

            if verbose:
                print(f"Raw Response Content (attempt {attempt}): {content[:500]}{'...' if len(content) > 500 else ''}")

            # 3. Match Pattern (if provided)
            if compiled_pattern:
                extracted_data = compiled_pattern.findall(content)
                if extracted_data:
                    if verbose:
                        print(f"Success: Pattern '{compiled_pattern.pattern}' found.")
                        print(f"Extracted Data: {extracted_data}")
                    return {
                        "success": True,
                        "extracted_data": extracted_data,
                        "final_content": content,
                        "retries_attempted": retry_count,
                        "error_message": None,
                        "raw_response": raw_response,
                    }
                else:
                    last_error = f"Pattern '{compiled_pattern.pattern}' not found in response."
                    if verbose:
                        print(f"Info: {last_error}")
                    current_messages.append(AIMessage(content=content))
                    current_messages.append(HumanMessage(content=[{"type": "text", "text": f"Instruction: The previous response did not match the required pattern: {compiled_pattern.pattern}. Please ensure your response includes content matching this pattern. Retrying. Previous attempt output: {content[:200]}..."}]))
            else:
                # No pattern provided, first valid string response is success
                if verbose:
                    print("Success: Valid string response received (no pattern matching required).")
                return {
                    "success": True,
                    "extracted_data": None, # No pattern to extract
                    "final_content": content,
                    "retries_attempted": retry_count,
                    "error_message": None,
                    "raw_response": raw_response,
                }

        except Exception as e:
            last_error = f"Exception during LLM invocation: {e}"
            if verbose:
                print(f"Error: {last_error}")
            current_messages.append(AIMessage(content=f"System notice: An error occurred processing the last turn: {str(e)}"))
            current_messages.append(HumanMessage(content=[{"type": "text", "text": f"Instruction: An unexpected error occurred. Let's try that again. Error details for context (if helpful): {last_error}"}]))


        # --- Prepare for next retry ---
        retry_count += 1
        if retry_count <= max_retries:
            if verbose:
                print(f"Retrying in {current_delay:.2f} seconds...")
            time.sleep(current_delay)
            current_delay *= backoff_factor
        else:
            if verbose:
                print("Max retries reached. Last error:", last_error)


    # --- End of Retries ---
    if verbose:
        print("\n--- Max retries reached or loop exited without success ---")

    return {
        "success": False,
        "extracted_data": None,
        "final_content": final_content,
        "retries_attempted": retry_count -1 if retry_count > 0 else 0,
        "error_message": last_error or "Max retries reached without success.",
        "raw_response": raw_response,
    }