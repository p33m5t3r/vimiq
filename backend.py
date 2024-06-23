import argparse
import asyncio
import os
import glob
from pathlib import Path
import logging
from logging.handlers import RotatingFileHandler
import json


# --------------------------- config ---------------------------
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
SYSTEM_DEFAULT = "you are a helpful assistant" 
CONFIG_FILE_PATTERN = ".vimiq.json"
DEBUG = False

# --------------------------- logging --------------------------
class JsonFormatter(logging.Formatter):
    def format(self, record):
        if isinstance(record.msg, dict):
            return json.dumps(record.msg)
        return super().format(record)

class JsonHandler(RotatingFileHandler):
    def __init__(self, filename, mode='a', maxBytes=0, backupCount=0, encoding=None, delay=False):
        super().__init__(filename, mode, maxBytes, backupCount, encoding, delay)
        self.json_data = {}

    def emit(self, record):
        try:
            msg = self.format(record)
            data = json.loads(msg)
            self.json_data.update(data)
            with open(self.baseFilename, 'w') as f:
                json.dump(self.json_data, f, indent=2)
        except Exception:
            self.handleError(record)

def setup_logger(name, log_file, level=logging.INFO):
    handler = JsonHandler(log_file, maxBytes=10*1024*1024, backupCount=5)
    handler.setFormatter(JsonFormatter())

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler)

    return logger

# Set up the logger
if DEBUG:
    logger = setup_logger('debug_logger', '.log.json')
else:
    logger = logging.getLogger('dummy')
    logger.addHandler(logging.NullHandler())

# Use the logger
def log_to_json(key, value):
    logger.info({key: value})


# --------------------------- utilities --------------------------
def find_config_file(user_cwd: str) -> str | None:
    """searches up from the user's working dir,
    ... until it finds a file that matches CONFIG_FILE_PATTERN,
    ... or we hit the user's home directory, at which point we return None"""

    if user_cwd is None:
        return None

    current_dir = Path(user_cwd).resolve()
    home_dir = Path.home()

    while current_dir != home_dir.parent:
        config_file = current_dir / CONFIG_FILE_PATTERN
        if config_file.exists():
            return str(config_file)
        current_dir = current_dir.parent

    return None

def get_project_root(config_file_abs_path: str) -> str | None:
    """ returns the project root based on where the config file was found """
    if config_file_abs_path:
        return str(Path(config_file_abs_path).parent)
    return None

def parse_config_file(config_file_abs_path: str | None) -> dict | None:
    """tries reading the config file and parsing into json"""
    if not config_file_abs_path:
        return None

    try:
        with open(config_file_abs_path, 'r') as f:
            return json.load(f)
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON in config file {config_file_abs_path}")
        return None
    except IOError:
        print(f"Error: Unable to read config file {config_file_abs_path}")
        return None

# some quote escaping shenanigans
def fmt_from_neovim(content: str) -> str:
    if content.startswith('\"'):
        content = content[1:]
    if content.endswith('\"'):
        content = content[:-1]
    return content

def send_to_neovim(text: str):
    print(text.replace("\n", "\\n"), flush=True, end='')

# --------------------------- llm implementation --------------------------
# this is all stuff you can change based on your desired look/feel
SYS_ROLE = ">>>> SYSTEM\n"
MODEL_ROLE = ">>>> MODEL\n"
INCLUDE_ROLE = ">>>> INCLUDE\n"
USER_ROLE = ">>>> USER\n"
ASSISTANT_ROLE = ">>>> ASSISTANT\n"

# defines more code-friendly internal representations for the role tags
# 'system', 'includes', 'user', and 'assistant' are hardcoded and used later
# ... will fix eventually
ROLES = {
    SYS_ROLE.strip(): "system",
    MODEL_ROLE.strip(): "model",
    INCLUDE_ROLE.strip(): "includes",
    USER_ROLE.strip(): "user",
    ASSISTANT_ROLE.strip(): "assistant"
}

def fmt_as_role(content: str | list[str], role: str) -> str:
    """ literally just prefixes the role. in its own function for consistency"""
    if not content:
        return ""
    if isinstance(content, list):
        return role + "\n".join(content)
    else:
        return role + content


def expand_context_files(root_dir_abs_path, 
                         current_file_abs_path, 
                         files: list[str]) -> str | None:
    """ expands given files/wildcards, puts them into a string in the form:
    filename:
        <content>

    filename: (user's current file)
        <content>
    ...
    """
    if not all([root_dir_abs_path, current_file_abs_path, files]):
        return None

    result = []
    
    for file_pattern in files:
        # Construct absolute path for the file pattern
        abs_pattern = os.path.join(root_dir_abs_path, file_pattern)
        
        # Expand the pattern
        for file_path in glob.glob(abs_pattern, recursive=True):
            try:
                with open(file_path, 'r') as f:
                    content = f.read()
                
                # Determine if this is the current file
                is_current = os.path.abspath(file_path) == os.path.abspath(current_file_abs_path)
                file_label = f"{os.path.relpath(file_path, root_dir_abs_path)}{'(current file)' if is_current else ''}:"
                
                # Format the file content
                formatted_content = '\n'.join(f"    {line}" for line in content.splitlines())
                
                # Add to result
                result.append(f"{file_label}\n{formatted_content}")
            
            except IOError as e:
                result.append(f"Error reading {file_path}: {str(e)}")
    
    return '\n\n'.join(result)

def fmt_chat_msges_default(request: dict):
    """ formats the user/assistant messages in anthropic/openai api compliant way.
        ... doesn't actually include the system prompt, since they differ on that"""
    messages = []
    
    # Zip user and assistant messages, handling cases where they might not be equal in number
    user_msgs = request.get('user', [])
    assistant_msgs = request.get('assistant', [])
    
    for i in range(max(len(user_msgs), len(assistant_msgs))):
        if i < len(user_msgs):
            messages.append({
                "role": "user",
                "content": [{"type": "text", "text": user_msgs[i]}]
            })
        if i < len(assistant_msgs):
            messages.append({
                "role": "assistant",
                "content": [{"type": "text", "text": assistant_msgs[i]}]
            })
    
    return messages

def fmt_system_msg_defult(request: dict) -> str:
    sys_req = request.get('system', None)
    sys_msg = SYSTEM_DEFAULT if not sys_req else sys_req[0]

    includes = request.get('includes')
    root_path = request.get('project_root')
    file_path = request.get('current_file')
    file_context = expand_context_files(root_path, file_path, includes)
    if not file_context:
        return sys_msg

    sys_msg += "\n the user has included the following files as context."
    sys_msg += " some or all of them may or may not be relevant to the conversation."
    sys_msg += " you may reference them as necessary:\n"
    return sys_msg + file_context

async def prepend_assistant_role():
    send_to_neovim("\n\n")
    send_to_neovim(ASSISTANT_ROLE)
    await asyncio.sleep(0.01)

async def postfix_user_role():
    send_to_neovim("\n")
    await asyncio.sleep(0.01)
    send_to_neovim(USER_ROLE)

async def llm_anthropic(request: dict, **kwargs):
    import anthropic
    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

    args = {
        "model": "claude-3-5-sonnet-20240620",
        "temperature": 0.1,
        "max_tokens": 4096,
        "system": fmt_system_msg_defult(request),
        "messages": fmt_chat_msges_default(request)
    }
    log_to_json('raw_request', request)
    log_to_json('llm_args', args)

    with client.messages.stream(**args) as stream:
        for text in stream.text_stream:
            send_to_neovim(text)


async def llm_openai(request: dict, **kwargs):
    import openai
    openai.api_key = OPENAI_API_KEY
    client = openai.OpenAI()

    messages = []
    sys_msg = {
        "role": "system",
        "content": [
            {
                "type": "text",
                "text": fmt_system_msg_defult(request)
            }
        ]
    }
    chat_messages = fmt_chat_msges_default(request)
    messages.append(sys_msg)
    messages.extend(chat_messages)
    
    args = {
        "model": "gpt-4o",
        "temperature": 0.1,
        "stream": True,
        "messages": messages,
    }
    log_to_json('raw_request', request)
    log_to_json('llm_args', args)

    stream = client.chat.completions.create(**args)
    for chunk in stream:
        if chunk.choices[0].delta.content is not None:
            send_to_neovim(chunk.choices[0].delta.content)


# this is what the ['model'] setting points at
LLM_IMPLS = {
    "default": llm_anthropic,
    "anthropic": llm_anthropic,
    "openai": llm_openai,
    "claude": llm_anthropic,        # example alias
    "chat": llm_openai              # example alias
}

# --------------------------- buffer management --------------------------
# assigns roles to chunks of text in the llm buffer, based on the defined ROLES
def parse_buffer_content(content: str) -> dict:
    roles = ROLES
    parsed_content = {role: [] for role in roles.values()}
    current_role = None
    current_content = []
    log_to_json('raw_buf_content', content)
    content = fmt_from_neovim(content) 

    for line in content.split('\n'):
        stripped_line = line.strip()
        if any(stripped_line.startswith(role.strip()) for role in roles):
            if current_role:
                parsed_content[roles[current_role]].append('\n'.join(current_content).strip())
                current_content = []
            current_role = next(role for role in roles if stripped_line.startswith(role.strip()))
        elif current_role:
            current_content.append(line)

    if current_role:
        parsed_content[roles[current_role]].append('\n'.join(current_content).strip())

    # parse included files into a single list
    file_patterns = sum(
        (s.split('\n') for s in parsed_content.get('includes', [])), []
    )
    file_patterns = [
        p for p in file_patterns if p.strip() and p.strip("\"")
    ]
    
    parsed_content['includes'] = file_patterns
    log_to_json('parsed_buf_content', parsed_content)
    return parsed_content

# prints the desired buffer contents to stdout
# ... just formats the context/file in a consistent manner
async def init_buffer(lua_input: str,
                      user_config_json: dict | None,
                      project_root_abs_path: str | None,
                      current_file_abs_path: str | None):

    relpath = ""
    if current_file_abs_path:
        if project_root_abs_path:
            relpath = os.path.relpath(current_file_abs_path, project_root_abs_path)
            if relpath == ".":
                relpath = os.path.basename(current_file_abs_path)
        else:
            relpath = os.path.basename(current_file_abs_path)

    if not relpath:
        relpath = current_file_abs_path or ""

    chunks = []
    if user_config_json:
        chunks.append(
            fmt_as_role(
                user_config_json.get("model"),
                MODEL_ROLE,
                )
        )

        chunks.append(
            fmt_as_role(
                user_config_json.get("system"),
                SYS_ROLE, 
            )
        )

        chunks.append(
            fmt_as_role(
                user_config_json.get("includes"),
                INCLUDE_ROLE, 
            )
        )

    # chunks.append(USER_ROLE[:-1])
    chunks.append(USER_ROLE)
    if lua_input:       # this means an error was passed in
        lua_input = fmt_from_neovim(lua_input)
        error_line, lsp_msg = lua_input.split("\n", 1)
        err_ctx_msg = f"can you assist with the following error? (in {relpath}):\n"
        err_ctx_msg += error_line + "\n"
        err_ctx_msg += "^^^\n"
        err_ctx_msg += lsp_msg
        chunks.append(err_ctx_msg)

    log_to_json('init_buffer_chunks', chunks)

    send_to_neovim("\n".join(chunks))

# evaluates the llm buffer and prints to stdout
async def eval_buffer(lua_input: str,
                      user_config_json: dict | None,
                      project_root_abs_path: str | None,
                      current_file_abs_path: str | None):
    """
    parses the lua input + config stuff into a request dict and call the right model.
    ... `request` is not a bloated dataclass, literally just a dict
    ... any other llm impls you provide will take this + **kwargs as args
    ... the default implementations assume yours looks like the one below:
    {
        'model': ['anthropic_sota_low_temp'],
        'system': ['you are a helpful assistant'],
        'includes': ['./backend.py', './init.lua'],
        'user': ['user first msg', 'user 2nd msg'],
        'assistant': ['assistant 1st'],['assistant 2nd'],
        'project_root': ['/home/anon/.../my_project'],
        'current_file': ['/home/anon/.../my_project/backend.py'],
     }
     ... we take in `user_config_json`, but don't use it here;
     ... the settings are implicitly derived from the given buffer.
     ... it exists as a param so you can easily hook in if you want custom behavior.
     ... it's not used by default so you can eval a blank buffer to suppress
     ... the default context, i.e. call an llm with a blank slate.
     """

    request = parse_buffer_content(lua_input)
    request['project_root'] = project_root_abs_path
    request['current_file'] = current_file_abs_path
    
    models_available = LLM_IMPLS
    model_chosen = request.get('model', ['default'])
    model_chosen = model_chosen[-1] if isinstance(model_chosen, list) else model_chosen

    if model_chosen not in models_available:
        print(f"Error: Model {model_chosen} not found")
        return

    await prepend_assistant_role()

    try:
        await models_available[model_chosen](request)
    except Exception as e:
        send_to_neovim(f"\nAn error occurred: {str(e)}\n")
    finally:
        await postfix_user_role()

# --------------------------- entrypoint & args -----------------------
# declared async for the sake of streaming llm api responses
async def main():
    parser = argparse.ArgumentParser(description="vimiq backend")

    # 2 default modes: either push things into a fresh buffer or evaluate a buffer's contents
    parser.add_argument('action', choices=['init', 'evaluate'], help="Action to perform")
    
    # this is where lua passes buffer contents, lsp error messages, etc...
    parser.add_argument('-i', '--input', type=str, help="Input data")

    # we take the working file so the llm knows what the user is looking at
    # ... in some cases, this may be None (if the user is in a scratch buffer or something)
    parser.add_argument('-f', '--file', type=str, help="Original working file")

    # we take the working directory to look for a config file (use this to define custom context) 
    parser.add_argument('-d', '--directory', type=str, help="Current working directory")
    args = parser.parse_args()
    log_to_json('parsed_args', args)

    # find the user's project-specific config file and the project root
    config_file_abs_path = find_config_file(args.directory)
    project_root_abs_path = get_project_root(config_file_abs_path)

    # try parsing the config json file into a dict
    user_config_json = parse_config_file(config_file_abs_path)
    """ a config file is expected to look like this:
    {
        "system": "you are a helpful assistant",
        "includes": ["relative/path/to/some_file.c", "another_file.h"]
    }
    plus whatever other fields you'd like to define
    """ 

    # the user's current file is taken as an absolute path in case we can't resolve a project root
    current_file_abs_path = args.file

    # this is a string that the actual nvim plugin gave us, which can contain effectively anything
    lua_input = args.input

    if args.action == 'init':
        await init_buffer(lua_input,
                          user_config_json,
                          project_root_abs_path,
                          current_file_abs_path)

    else:
        await eval_buffer(lua_input,
                          user_config_json,
                          project_root_abs_path,
                          current_file_abs_path)

    
if __name__ == "__main__":
    asyncio.run(main())


