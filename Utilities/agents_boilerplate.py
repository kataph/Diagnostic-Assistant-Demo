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
            return (conversation_start := [input_item])
        
def get_updated_conversation(conversation:list[dict], new_text:str)->list[dict]:
            return conversation + [{
                                    "role": "user",
                                    "content": [
                                        {"type": "input_text",
                                        "text": new_text}
                                    ]
                                }]
def update_conversation(conversation:list[dict], new_text:str)->None:
            conversation = get_updated_conversation(conversation, new_text)