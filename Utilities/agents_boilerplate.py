from environment_classes import SystemDescription


def get_conversation_start(system_description: SystemDescription) -> list[dict]:
    """
    LEGACY path (conversation_in_instructions=False):
    System description text + diagram are sent as the first user message.
    Kept for backwards compatibility and easy A/B comparison.
    """
    input_item = {
        "role": "user",
        "content": [
            {
                "type": "input_text",
                "text": system_description.text_input
            }
        ]
    }
    if system_description.file_id:
        input_item['content'].append(
            {
                "type": "input_image",
                "file_id": system_description.file_id
            },
        )
    elif system_description.image_b64:
        input_item['content'].append(
            {
                "type": "input_image",
                "image_url": f"data:image/png;base64,{system_description.image_b64}",
            },
        )
    return [input_item]


def get_system_description_instructions(system_description: SystemDescription) -> list[dict]:
    """
    NEW path (conversation_in_instructions=True):
    Returns Agent instructions content blocks that embed the system description
    text and diagram directly in the system prompt, making them cacheable across
    turns. The conversation history starts empty and only contains the diagnostic
    dialogue (symptoms, actions, results).

    Returns a list of content blocks suitable for passing as Agent instructions
    when using the Responses API, or a plain string when only text is needed.
    """
    blocks: list[dict] = [{"type": "input_text", "text": system_description.text_input}]
    if system_description.file_id:
        blocks.append({"type": "input_image", "file_id": system_description.file_id})
    elif system_description.image_b64:
        blocks.append({
            "type": "input_image",
            "image_url": f"data:image/png;base64,{system_description.image_b64}",
        })
    return blocks


def get_updated_conversation(conversation: list[dict], new_text: str) -> list[dict]:
    return conversation + [{
        "role": "user",
        "content": [
            {"type": "input_text",
             "text": new_text}
        ]
    }]


def append_assistant_turn(conversation: list[dict], suggestion_text: str) -> list[dict]:
    """
    Appends the assistant's own suggestion as an assistant-role message so the
    model sees a proper alternating user/assistant dialogue on the next turn,
    rather than a sequence of user-only messages.
    """
    return conversation + [{
        "role": "assistant",
        "content": suggestion_text,
    }]