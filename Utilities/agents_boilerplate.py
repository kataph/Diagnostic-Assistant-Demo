from environment_classes import SystemDescription


def get_conversation_start(system_description: SystemDescription) -> list[dict]:
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
    return (conversation_start := [input_item])


def get_updated_conversation(conversation: list[dict], new_text: str) -> list[dict]:
    return conversation + [{
        "role": "user",
        "content": [
            {"type": "input_text",
             "text": new_text}
        ]
    }]