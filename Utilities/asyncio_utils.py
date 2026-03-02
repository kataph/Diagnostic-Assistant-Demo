import asyncio

async def async_friendly_input(prompt: str = "") -> str:
    return await asyncio.to_thread(input, prompt)
