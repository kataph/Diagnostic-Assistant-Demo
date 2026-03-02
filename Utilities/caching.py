import asyncio
from functools import partial, wraps
import hashlib
import os
import pickle 
import inspect
import logging

#### sets up logging for the caching funtions
CACHE_FOLDER = "Cache"
cache_logger = logging.getLogger("cache_logger")
cache_logger.setLevel(10) # INFO is 20

os.makedirs(CACHE_FOLDER, exist_ok=True)
file_handler = logging.FileHandler(os.path.join(CACHE_FOLDER, "cache_logs.txt"), encoding="utf-8")

fmt = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
datefmt = "%Y-%m-%d %H:%M:%S"
formatter = logging.Formatter(fmt=fmt, datefmt=datefmt)
file_handler.setFormatter(formatter)
        
cache_logger.addHandler(file_handler)
        
        
        


from agents import Runner
from openai import OpenAI

from diskcache import Cache

# This is a async,agent-friendly caching mechanism from chatgpt...
cache = Cache(CACHE_FOLDER)
# cache.clear() # this will remove cache storage, comment to make caching persistent. Uncomment to reset the cache content.
_MISSING = object()  # one shared sentinel




def add_disk_cacheing_option(func):
    """The decorator adds a keyword argument to the function: 'cached': bool = False. If set to true, disk caching is used to cache the function. Note that with methods a representation of the class instance will be used to build the key."""
    if "cached" in inspect.signature(func).parameters:
        raise TypeError(f"'cached' argument already used in {func.__qualname__}")
    @wraps(func)
    def wrapper(*args, cached = False, **kwargs):
        # the repr thing is my idea after noticing that the FM tree in input leads to random missing
        if not cached:
            return func(*args, **kwargs)
        else:
            key = (func.__module__, func.__qualname__, tuple(repr(arg) for arg in args), frozenset(repr(kwarg) for kwarg in kwargs.items()))
            if key in cache:
                cache_logger.debug(f"[SYNC CACHE HIT] key={str(key)[:40]}...")
                return cache[key]
            cache_logger.debug(f"[SYNC CACHE MISS] key={str(key)[:40]}...")
            result = func(*args, **kwargs)
            cache[key] = result
            return result
    return wrapper

def add_disk_cacheing_option_for_methods(func):
    if "cached" in inspect.signature(func).parameters:
        raise TypeError(f"'cached' argument already used in {func.__qualname__}")
    """The decorator adds a keyword argument to the function: 'cached': bool = False. If set to true, disk caching is used to cache the function. Note that the class instance will not be used to build the key"""
    @wraps(func)
    def wrapper(*args, cached = False, **kwargs):
        # the repr thing is my idea after noticing that the FM tree in input leads to random missing
        if not cached:
            return func(*args, **kwargs)
        else:
            key = (func.__module__, func.__qualname__, tuple(repr(arg) for arg in args[1:]), frozenset(repr(kwarg) for kwarg in kwargs.items()))
            if key in cache:
                cache_logger.debug(f"[SYNC CACHE HIT] key={str(key)[:40]}...")
                return cache[key]
            cache_logger.debug(f"[SYNC CACHE MISS] key={str(key)[:40]}...")
            result = func(*args, **kwargs)
            cache[key] = result
            return result
    return wrapper

# class adder():
#     @add_disk_cacheing_option_for_methods
#     def add(self, x, y):
#         return x+y
    
# a=adder()
# print(a.add(11,1, cached = False))
# print(a.add(11,1, cached = True))
# quit()

def async_disk_cache_runner_run(func):
    @wraps(func) 
    async def wrapper(agent, *args, **kwargs):
        # 1) what we cache on: agent + input
        conversation = kwargs.get("input")

        # Prefer a stable, explicit id; customize as needed:
        agent_id = (
            getattr(agent, "name", None)
            # or getattr(agent, "id", None) # FC: commenting these 2 lines: they should be redundant TODO
            # or agent.__class__.__name__  # fallback: class name
        )

        # 2) build key
        raw = (func.__module__, func.__qualname__, agent_id, conversation)
        key = hashlib.sha256(pickle.dumps(raw)).hexdigest()

        # 3) try cache
        def get_from_cache():
            return cache.get(key, _MISSING)

        cached_value = await asyncio.to_thread(get_from_cache)
        if cached_value is not _MISSING:
            cache_logger.warning(f"[CACHE HIT] {agent_id} / key={key[:8]}... / raw={raw}")
            cache_logger.warning(f"[INPUT] {conversation} ***")
            return cached_value

        cache_logger.info(f"[CACHE MISS] {agent_id} / key={key[:8]}... / raw={raw}")

        # 4) call underlying function
        result = await func(agent, *args, **kwargs)

        # 5) store in cache
        def set_in_cache():
            cache.set(key, result)

        await asyncio.to_thread(set_in_cache)
        return result

    return wrapper

def async_disk_cache_CLI(func):
    @wraps(func)
    async def wrapper(*args, **kwargs):
        use_cache = kwargs.pop("cache", True)
        cache_logger.warning(f"CLI CACHING ACTIVATING WITH use_cache {use_cache}...")

        if not use_cache:
            return await func(*args, **kwargs)

        # Build cache key
        raw = (func.__module__, func.__qualname__, [repr(x) for x in args[1:]], kwargs)
        key = hashlib.sha256(pickle.dumps(raw)).hexdigest()

        # Read from disk (non-blocking)
        cached_value = await asyncio.to_thread(cache.get, key)
        if cached_value is not None:
            cache_logger.warning(f"[CACHE HIT] raw={raw} --- key={key[:8]}...")
            return cached_value
        cache_logger.warning(f"[CACHE MISS] raw={raw} --- key={key[:8]}...")

        # Execute function
        result = await func(*args, **kwargs)

        # Store result (non-blocking)
        await asyncio.to_thread(cache.set, key, result)

        return result

    return wrapper


# # Function to create a file with the Files API. This must also be cached otherwise the generated id will always be different and it will make the input of the agent calls different each time, since it is part of the conversation
# @add_disk_cacheing_option
# def create_file(file_path):
#   with open(file_path, "rb") as file_content:
#     result = client.files.create(
#         file=file_content,
#         purpose="vision",
#     )
#     return result.id

@async_disk_cache_runner_run
async def cached_runner_run(agent, input):
    o = await Runner.run(starting_agent=agent, input=input)
    return o.final_output

async def possibly_cached_runner_run(agent, input, cached: bool):
    if cached:
        o = await cached_runner_run(agent=agent, input=input)
        return o 
    else:
        o = await Runner.run(starting_agent=agent, input=input)
        return o.final_output
        