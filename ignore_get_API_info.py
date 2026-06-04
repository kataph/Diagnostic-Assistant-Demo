
def get_available_models() -> None:
    """Doesnt work, but unnecessary"""
    import requests
    import os

    url = os.getenv("OPENAI_BASE_URL")
    key = os.getenv("OPENAI_API_KEY")
    print(f"{url}, {key}")
    headers = {
        "Accept": "application/json",
        "Authorization": f"Bearer {key}",
    }

    response = requests.get(url, headers=headers)
    response.raise_for_status()

    print("STATUS:", response.status_code)
    print("HEADERS:", response.headers)
    print("BODY:")
    print(response.text)

    response.raise_for_status()

    return response.json()

def test_chat_completions_works() -> None:
    import openai

    client = openai.OpenAI()

    response = client.chat.completions.create(
        model="gpt-4.1", # model to send to the proxy, see below for options available
        messages = [{
            "role": "user",
            "content": "this is a test request, write a short poem"
        }]
    )

    print(response.choices[0].message.content)
    
def test_multimodal_image_visible(
    image_path: str = "Knowledge_sources/Unstructured_knowledge_sources/3_cubes/3_cubes_schematics.png",
    model: str = "gpt-4.1",
) -> None:
    """
    Test whether the model can see an attached image.
    Mirrors the logic in run_diagnostic_scenario.get_vision_diagram /
    agents_boilerplate.get_conversation_start:
      1. Try to upload the file via the Files API (server-side reference).
      2. On failure, fall back to inline base64.
    """
    import base64
    import logging
    import openai

    client = openai.OpenAI()

    # -- obtain file_id or b64, same as get_vision_diagram() ------------------
    file_id, image_b64 = None, None
    try:
        with open(image_path, "rb") as f:
            result = client.files.create(file=f, purpose="vision")
        file_id = result.id
        print(f"Uploaded image as file_id={file_id}")
    except Exception as exc:
        logging.warning(
            f"Could not upload {image_path!r} ({type(exc).__name__}: {exc}). "
            "Falling back to inline base64."
        )
        try:
            with open(image_path, "rb") as f:
                image_b64 = base64.b64encode(f.read()).decode("utf-8")
            print("Using inline base64.")
        except Exception as exc2:
            print(f"Could not read image for base64 encoding: {exc2}")
            return

    # -- build the content block (Chat Completions format) --------------------
    if file_id:
        image_block = {"type": "image_url", "image_url": {"url": f"file-id-{file_id}"}}
    else:
        image_block = {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_b64}"}}

    response = client.chat.completions.create(
        model=model,
        messages=[{
            "role": "user",
            "content": [
                image_block,
                {
                    "type": "text",
                    "text": (
                        "Please describe in detail what you see in this image. "
                        "What kind of diagram or schematic is it? "
                        "What components or elements can you identify?"
                    ),
                },
            ],
        }],
    )

    print(response.choices[0].message.content)


async def test_multimodal_image_visible_agents(
    image_path: str = "Knowledge_sources/Unstructured_knowledge_sources/3_cubes/3_cubes_schematics.png",
    model: str = "gpt-4.1",
) -> None:
    """
    Same test as test_multimodal_image_visible, but going through the Agents SDK
    (Runner.run) exactly as diagnosticAssistantLLM.suggest_action does.
    Uses the same get_vision_diagram upload logic and the same agents_boilerplate
    content-block format (input_text / input_image).
    """
    import base64
    import logging
    import openai
    from agents import Agent, Runner
    from pydantic import BaseModel

    client = openai.OpenAI()

    # -- same upload/fallback logic as run_diagnostic_scenario.get_vision_diagram
    file_id, image_b64 = None, None
    try:
        with open(image_path, "rb") as f:
            result = client.files.create(file=f, purpose="vision")
        file_id = result.id
        print(f"Uploaded image as file_id={file_id}")
    except Exception as exc:
        logging.warning(
            f"Could not upload {image_path!r} ({type(exc).__name__}: {exc}). "
            "Falling back to inline base64."
        )
        try:
            with open(image_path, "rb") as f:
                image_b64 = base64.b64encode(f.read()).decode("utf-8")
            print("Using inline base64.")
        except Exception as exc2:
            print(f"Could not read image for base64 encoding: {exc2}")
            return

    # -- same content-block format as agents_boilerplate.get_conversation_start
    content = [{"type": "input_text", "text": "Please describe in detail what you see in this image. What kind of diagram or schematic is it? What components or elements can you identify? Also conclude with a table with (Symbol/Identified Component/Function) columns"}]
    if file_id:
        content.append({"type": "input_image", "file_id": file_id})
    elif image_b64:
        content.append({"type": "input_image", "image_url": f"data:image/png;base64,{image_b64}"})

    input_message = [{"role": "user", "content": content}]

    class ImageDescription(BaseModel):
        description: str

    agent = Agent(
        name="ImageDescriber",
        instructions="You are a helpful assistant.",
        output_type=ImageDescription,
        model=model,
    )

    result = await Runner.run(starting_agent=agent, input=input_message)
    print(result.final_output.description)


if __name__ == "__main__":
    import asyncio
    asyncio.run(test_multimodal_image_visible_agents())