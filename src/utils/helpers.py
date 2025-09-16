"""Helper utility functions for DroidBot-GPT framework."""

from __future__ import annotations

import asyncio
import time
from typing import Any, Callable, Optional, TypeVar

from ..core.logger import log

T = TypeVar('T')


async def retry_with_backoff(
    func: Callable[..., T],
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    exponential_base: float = 2.0,
    exceptions: tuple = (Exception,),
    *args: Any,
    **kwargs: Any
) -> T:
    """Retry a function with exponential backoff.
    
    Args:
        func: Function to retry.
        max_retries: Maximum number of retry attempts.
        base_delay: Base delay between retries in seconds.
        max_delay: Maximum delay between retries in seconds.
        exponential_base: Base for exponential backoff calculation.
        exceptions: Tuple of exceptions to catch and retry.
        *args: Arguments to pass to the function.
        **kwargs: Keyword arguments to pass to the function.
        
    Returns:
        Result of the function call.
        
    Raises:
        Exception: Last exception if all retries fail.
    """
    last_exception = None
    
    for attempt in range(max_retries + 1):
        try:
            if asyncio.iscoroutinefunction(func):
                return await func(*args, **kwargs)
            else:
                return func(*args, **kwargs)
                
        except exceptions as e:
            last_exception = e
            
            if attempt == max_retries:
                log.error(f"Function failed after {max_retries} retries: {e}")
                raise
                
            # Calculate delay with exponential backoff
            delay = min(base_delay * (exponential_base ** attempt), max_delay)
            
            log.warning(f"Function failed (attempt {attempt + 1}/{max_retries + 1}): {e}")
            log.info(f"Retrying in {delay:.2f} seconds...")
            
            await asyncio.sleep(delay)
    
    # This should never be reached, but just in case
    raise last_exception


async def async_timeout(
    coro: Any,
    timeout: float,
    default: Optional[Any] = None
) -> Any:
    """Execute a coroutine with a timeout.
    
    Args:
        coro: Coroutine to execute.
        timeout: Timeout in seconds.
        default: Default value to return if timeout occurs.
        
    Returns:
        Result of the coroutine or default value if timeout.
    """
    try:
        return await asyncio.wait_for(coro, timeout=timeout)
    except asyncio.TimeoutError:
        log.warning(f"Operation timed out after {timeout} seconds")
        return default


def debounce(func: Callable[..., T], delay: float) -> Callable[..., T]:
    """Create a debounced version of a function.
    
    Args:
        func: Function to debounce.
        delay: Delay in seconds.
        
    Returns:
        Debounced function.
    """
    timer: Optional[asyncio.Task] = None
    
    async def debounced(*args: Any, **kwargs: Any) -> T:
        nonlocal timer
        
        if timer:
            timer.cancel()
            
        async def delayed_call():
            await asyncio.sleep(delay)
            return await func(*args, **kwargs)
            
        timer = asyncio.create_task(delayed_call())
        return await timer
        
    return debounced


def throttle(func: Callable[..., T], rate_limit: float) -> Callable[..., T]:
    """Create a throttled version of a function.
    
    Args:
        func: Function to throttle.
        rate_limit: Minimum time between calls in seconds.
        
    Returns:
        Throttled function.
    """
    last_call = 0.0
    
    async def throttled(*args: Any, **kwargs: Any) -> T:
        nonlocal last_call
        
        current_time = time.time()
        time_since_last = current_time - last_call
        
        if time_since_last < rate_limit:
            await asyncio.sleep(rate_limit - time_since_last)
            
        last_call = time.time()
        return await func(*args, **kwargs)
        
    return throttled


async def wait_for_condition(
    condition_func: Callable[[], bool],
    timeout: float = 30.0,
    check_interval: float = 0.5
) -> bool:
    """Wait for a condition to become true.
    
    Args:
        condition_func: Function that returns True when condition is met.
        timeout: Maximum time to wait in seconds.
        check_interval: Interval between condition checks in seconds.
        
    Returns:
        True if condition was met, False if timeout occurred.
    """
    start_time = time.time()
    
    while time.time() - start_time < timeout:
        if condition_func():
            return True
        await asyncio.sleep(check_interval)
        
    return False


def format_duration(seconds: float) -> str:
    """Format a duration in seconds to a human-readable string.
    
    Args:
        seconds: Duration in seconds.
        
    Returns:
        Formatted duration string.
    """
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.1f}h"


def format_bytes(bytes_value: int) -> str:
    """Format bytes to a human-readable string.
    
    Args:
        bytes_value: Number of bytes.
        
    Returns:
        Formatted bytes string.
    """
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if bytes_value < 1024.0:
            return f"{bytes_value:.1f}{unit}"
        bytes_value /= 1024.0
    return f"{bytes_value:.1f}PB"


def safe_get(dictionary: dict, key: str, default: Any = None) -> Any:
    """Safely get a value from a dictionary.
    
    Args:
        dictionary: Dictionary to search.
        key: Key to look for.
        default: Default value if key not found.
        
    Returns:
        Value from dictionary or default.
    """
    return dictionary.get(key, default)


def deep_get(dictionary: dict, keys: list, default: Any = None) -> Any:
    """Get a value from a nested dictionary using a list of keys.
    
    Args:
        dictionary: Dictionary to search.
        keys: List of keys to traverse.
        default: Default value if path not found.
        
    Returns:
        Value from nested dictionary or default.
    """
    current = dictionary
    
    for key in keys:
        if isinstance(current, dict) and key in current:
            current = current[key]
        else:
            return default
            
    return current


def merge_dicts(dict1: dict, dict2: dict) -> dict:
    """Merge two dictionaries, with dict2 taking precedence.
    
    Args:
        dict1: First dictionary.
        dict2: Second dictionary (takes precedence).
        
    Returns:
        Merged dictionary.
    """
    result = dict1.copy()
    result.update(dict2)
    return result


def chunk_list(lst: list, chunk_size: int) -> list:
    """Split a list into chunks of specified size.
    
    Args:
        lst: List to split.
        chunk_size: Size of each chunk.
        
    Returns:
        List of chunks.
    """
    return [lst[i:i + chunk_size] for i in range(0, len(lst), chunk_size)]


def flatten_list(nested_list: list) -> list:
    """Flatten a nested list.
    
    Args:
        nested_list: List that may contain nested lists.
        
    Returns:
        Flattened list.
    """
    flattened = []
    for item in nested_list:
        if isinstance(item, list):
            flattened.extend(flatten_list(item))
        else:
            flattened.append(item)
    return flattened


def unique_list(lst: list) -> list:
    """Remove duplicates from a list while preserving order.
    
    Args:
        lst: List to deduplicate.
        
    Returns:
        List with duplicates removed.
    """
    seen = set()
    return [x for x in lst if not (x in seen or seen.add(x))]


def is_valid_email(email: str) -> bool:
    """Check if a string is a valid email address.
    
    Args:
        email: Email string to validate.
        
    Returns:
        True if valid email, False otherwise.
    """
    import re
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return bool(re.match(pattern, email))


def is_valid_url(url: str) -> bool:
    """Check if a string is a valid URL.
    
    Args:
        url: URL string to validate.
        
    Returns:
        True if valid URL, False otherwise.
    """
    import re
    pattern = r'^https?://(?:[-\w.])+(?:[:\d]+)?(?:/(?:[\w/_.])*(?:\?(?:[\w&=%.])*)?(?:#(?:[\w.])*)?)?$'
    return bool(re.match(pattern, url)) 