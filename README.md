
# vimiq

A Neovim plugin for interacting with language models

This is a work-in progress, and may have bugs. But I didn't love the existing options, so i wrote an llm extension for myself. it aims to be extremely simple and easy to modify. you can see the demo here:


## setup

1. clone/move the repo to `.config/nvim/lua/vimiq` (or wherever you nvim plugin lua lives)
2. if you want to use anthropic or openai, then:
   ```
   pip install anthropic openai
   ```
3. also set your api keys in your shell:
   ```
   export ANTHROPIC_API_KEY="your_anthropic_api_key"
   export OPENAI_API_KEY="your_openai_api_key"
   ```
4. in your init.lua
   ```lua
    require('vimiq').setup({
        python_path = "/path/to/your/python/installation/or/virtualenv"
    })
   ```
5. on a per-project basis, you can create a `.vimiq.json` file to put context into your chats
    ```json
    {
        "model": ["default"],
        "system": "You are a helpful assistant.",
        "includes": ["./backend.py", "./init.lua"]
    }
    ```
6. you can edit `backend.py` to include your own llm implemenations:
    ```python
    LLM_IMPLS = {
        "default": llm_anthropic,
        "anthropic": llm_anthropic,
        "openai": llm_openai,
        "claude": llm_anthropic,        # example alias
        "chat": llm_openai              # example alias
    }
    ```

## usage

commands:
- `:InitBuffer`: start new llm buffer
- `:InitBufferWithLSPError`: start buffer with current line and lsp errors
- `:EvalBuffer`: send buffer to llm
- `:StopStreaming`: stop current stream

## structure

- `init.lua`: handles neovim stuff (buffers, commands, data passing)
- `backend.py`: llm logic (message formatting, api calls, response streaming)

separated like this so you can modify llm logic without touching neovim code.

## customization

- edit `backend.py` to change llm providers, message formatting, etc.
- update `init.lua` if you need new commands or data passing changes
- use `.vimiq.json` in project root for custom prompts or context files

## TODO

- add custom filetype syntax
- add locallama support
