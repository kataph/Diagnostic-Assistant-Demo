from environment_classes import SystemDescription

# Design note — flat user-message history:
# The conversation is intentionally a flat sequence of user-role messages
# (system description + diagram, then symptoms, then action outcomes). No
# assistant turns are interleaved. Reasons:
#
# 1. The Agents SDK Agent.instructions only accepts a string, so images
#    cannot be moved there; the system description must stay as a user message.
#
# 2. Adding assistant turns would double token count with no new information:
#    each action outcome message already contains the DESCRIPTION field from
#    the assistant's prior suggestion, so the model is not reasoning blind.
#
# 3. The flat structure handles pre-session observations naturally — action
#    outcomes injected before the assistant starts are seamlessly part of
#    context without requiring strict user/assistant turn alternation.
#
# 4. Interleaving assistant turns would cause the model to re-read its own
#    prior suggestions as first-person decisions ("I said this"), which
#    anchors it to earlier hypotheses and makes it harder to revise. In the
#    flat structure, prior suggestions appear only as the DESCRIPTION field
#    in outcome messages — framed as "an action was taken", not "I decided
#    this" — which is a weaker anchor and leaves more room for hypothesis
#    revision.


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