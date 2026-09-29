from pathlib import Path
import re
import ollama

MODEL = "qwen3:8b"

# All file operations are restricted to this directory.
ROOT = Path(r"C:\Users\Lenovo\Downloads\AGENT").resolve()
ROOT.mkdir(parents=True, exist_ok=True)


def safe_path(path: str) -> Path:
    """Resolve a path and prevent access outside the workspace."""
    target = (ROOT / path).resolve()
    if not target.is_relative_to(ROOT):
        raise ValueError("Access outside the workspace is prohibited.")
    return target


def create_file(path: str, content: str) -> str:
    """Create a new file with the specified text content."""
    target = safe_path(path)
    if target.exists():
        return "Error: File already exists."
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content, encoding="utf-8")
    return f"Created: {target}"


def parse_size_string(size_text: str) -> int:
    """Convert values like '1 MB' or '512 KB' into bytes."""
    match = re.fullmatch(r"\s*(\d+(?:\.\d+)?)\s*(b|bytes|kb|mb|gb|tb)\s*", size_text.strip().lower())
    if not match:
        raise ValueError("Invalid size format. Use values like '1 MB' or '512 KB'.")

    value = float(match.group(1))
    unit = match.group(2)
    units = {
        "b": 1,
        "bytes": 1,
        "kb": 1024,
        "mb": 1024 ** 2,
        "gb": 1024 ** 3,
        "tb": 1024 ** 4,
    }
    return int(value * units[unit])


def create_file_of_size(path: str, size: str) -> str:
    """Create a file of the requested size in bytes."""
    target = safe_path(path)
    if target.exists():
        return "Error: File already exists."

    size_bytes = parse_size_string(size)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(b"\0" * size_bytes)
    return f"Created: {target} ({size_bytes} bytes)"


def parse_size_creation_request(prompt: str):
    """Match prompts such as 'create a 1 mb file' or 'create file notes.txt of 2 MB'."""
    size_match = re.search(r"(?P<size>\d+(?:\.\d+)?\s*(?:b|bytes|kb|mb|gb|tb))", prompt, flags=re.IGNORECASE)
    if not size_match:
        return None

    size = size_match.group("size")
    name_match = re.search(
        r"(?:named|called|as|file)\s+(?P<name>[A-Za-z0-9_.-]+)",
        prompt,
        flags=re.IGNORECASE,
    )
    if not name_match:
        name_match = re.search(
            r"(?:create|make|generate)\s+(?:a\s+|an\s+|this\s+|that\s+)?"
            r"(?P<name>[A-Za-z0-9_.-]+)\s+(?:of\s+)?(?P<size>\d+(?:\.\d+)?\s*(?:b|bytes|kb|mb|gb|tb))",
            prompt,
            flags=re.IGNORECASE,
        )

    raw_name = name_match.group("name") if name_match else None
    if raw_name and raw_name.lower() not in {"a", "an", "this", "that", "file"}:
        name = raw_name
    else:
        name = "generated_file.bin"

    return name, size


def rename_file(old_path: str, new_path: str) -> str:
    """Rename or move a file within the workspace."""
    source = safe_path(old_path)
    destination = safe_path(new_path)

    if not source.is_file():
        return "Error: Source file does not exist."
    if destination.exists():
        return "Error: Destination already exists."

    destination.parent.mkdir(parents=True, exist_ok=True)
    source.rename(destination)
    return f"Renamed to: {destination}"


def delete_file(path: str) -> str:
    """Delete a file after obtaining user confirmation."""
    target = safe_path(path)

    if not target.is_file():
        return "Error: File does not exist."

    answer = input(f"Confirm deletion of {target}? (yes/no): ")
    if answer.strip().lower() != "yes":
        return "Deletion cancelled."

    target.unlink()
    return f"Deleted: {target}"


def list_files(path: str = ".") -> str:
    """List files and directories in a workspace folder."""
    target = safe_path(path)
    if not target.is_dir():
        return "Error: Directory does not exist."

    return "\n".join(
        ("[DIR] " if p.is_dir() else "[FILE] ") + p.name
        for p in sorted(target.iterdir())
    ) or "Directory is empty."


TOOLS = [create_file, create_file_of_size, rename_file, delete_file, list_files]

SYSTEM_PROMPT = """
You are a local file-management assistant.
Use the available tools to create, rename, delete and list files.
All paths are relative to the provided workspace.
Never claim that an operation succeeded unless its tool confirms it.
Ask the user for missing file names or content.
Do not attempt to bypass tool restrictions.
"""


def run_agent():
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    print("Local Qwen3 File Agent")
    print("Workspace:", ROOT)
    print("Type 'exit' to quit.")

    while True:
        prompt = input("\nEnter file details: ").strip()
        if prompt.lower() == "exit":
            break
        if not prompt:
            continue

        size_request = parse_size_creation_request(prompt)
        if size_request:
            file_name, size = size_request
            print(f"\n[Tool: create_file_of_size]")
            print(create_file_of_size(file_name, size))
            continue

        messages.append({"role": "user", "content": prompt})

        while True:
            response = ollama.chat(
                model=MODEL,
                messages=messages,
                tools=TOOLS,
            )

            messages.append(response.message)

            if not response.message.tool_calls:
                print("\nAgent:", response.message.content)
                break

            for call in response.message.tool_calls:       
                name = call.function.name
                args = call.function.arguments

                tool = next(
                    (t for t in TOOLS if t.__name__ == name),
                    None
                )

                if tool is None:
                    result = "Error: Unknown tool."
                else:
                    try:
                        result = tool(**args)
                    except Exception as exc:
                        result = f"Error: {exc}"

                print(f"\n[Tool: {name}]")
                print(result)

                messages.append({
                    "role": "tool",
                    "tool_name": name,
                    "content": str(result),
                })


if __name__ == "__main__":
    run_agent()