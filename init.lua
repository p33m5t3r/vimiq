local M = {}
M.config = {
    python_path = "python"
}
function M.setup(opts)
    M.config = vim.tbl_deep_extend("force", M.config, opts or {})
end
local function get_backend_path()
    local current_file = debug.getinfo(1, "S").source:sub(2)
    local current_dir = vim.fn.fnamemodify(current_file, ":h")
    return current_dir .. "/backend.py"
end

local function get_python_path()
    return M.config.python_path
end

local llm_buffer_id = nil
local original_file_path = nil
local current_job_id = nil
local backend_path = get_backend_path()

-- Function to get the current working directory
local function get_current_directory()
    return vim.fn.getcwd()
end

-- Function to get the current working file
local function get_working_file()
    local methods = {
        function() return vim.api.nvim_buf_get_name(0) end,
        function() return vim.fn.expand('%:p') end,
        function() return vim.fn.getcwd() .. '/' .. vim.fn.expand('%:t') end
    }

    for _, method in ipairs(methods) do
        local file_path = method()
        if file_path and file_path ~= '' then
            return file_path
        end
    end

    -- If all methods fail, return a default value
    return vim.fn.getcwd() .. '/unnamed_file'
end


-- Function to get or create the LLM buffer
local function get_llm_buffer()
    if not llm_buffer_id or not vim.api.nvim_buf_is_valid(llm_buffer_id) then
        llm_buffer_id = vim.api.nvim_create_buf(false, true)
        vim.api.nvim_buf_set_option(llm_buffer_id, 'buftype', 'nofile')
        vim.api.nvim_buf_set_option(llm_buffer_id, 'bufhidden', 'hide')
        vim.api.nvim_buf_set_option(llm_buffer_id, 'swapfile', false)
    end
    return llm_buffer_id
end

-- Function to display the LLM buffer in a split
local function display_llm_buffer()
    local buf = get_llm_buffer()
    
    -- Check if there's already a window with the LLM buffer
    local existing_win = nil
    for _, win in ipairs(vim.api.nvim_list_wins()) do
        if vim.api.nvim_win_get_buf(win) == buf then
            existing_win = win
            break
        end
    end
    
    if existing_win then
        -- If a window exists, just focus on it
        vim.api.nvim_set_current_win(existing_win)
    else
        -- If no window exists, create a new split
        vim.cmd('vsplit')
        vim.api.nvim_win_set_buf(0, buf)
        vim.cmd('vertical resize ' .. math.floor(vim.o.columns * 0.4))
    end
end

-- Function to stream content from Python script to buffer
local function stream_to_buffer(cmd)
    local buffer = get_llm_buffer()

    local function on_stdout(_, data, _)
        if data then
            vim.schedule(function()
                for _, chunk in ipairs(data) do
                    if chunk ~= "" then
                        local split_lines = vim.split(chunk, "\\n", true)
                        for i, line in ipairs(split_lines) do
                            if i == 1 then
                                -- Append to the last line
                                local last_line = vim.api.nvim_buf_get_lines(buffer, -2, -1, false)[1]
                                vim.api.nvim_buf_set_lines(buffer, -2, -1, false, {last_line .. line})
                            else
                                -- Add as a new line
                                vim.api.nvim_buf_set_lines(buffer, -1, -1, false, {line})
                            end
                        end

                        -- Scroll to the bottom
                        local win = vim.fn.bufwinid(buffer)
                        if win ~= -1 then
                            local line_count = vim.api.nvim_buf_line_count(buffer)
                            vim.api.nvim_win_set_cursor(win, {line_count, 0})
                        end

                        vim.cmd('redraw')
                    end
                end
            end)
        end
    end

    local function on_exit(_, _, _)
        current_job_id = nil
    end

    current_job_id = vim.fn.jobstart(cmd, {
        on_stdout = on_stdout,
        on_exit = on_exit,
        stdout_buffered = false
    })
end

-- Function to stop the streaming
function M.stop_streaming()
    if current_job_id then
        vim.fn.jobstop(current_job_id)
        print("Streaming stopped")
    else
        print("No active streaming to stop")
    end
end

local function run_backend_command(input, arg)
    local current_dir = get_current_directory()
    local current_file = get_working_file()
    local python_path = get_python_path()

    local cmd = {
        python_path,
        backend_path,
        arg or "",
        "-f", current_file,
        "-d", current_dir
    }
    
    if input and input ~= "" then
        input = input:gsub('"', '\\"')
        table.insert(cmd, "-i")
        table.insert(cmd, '"' .. input .. '"')
    end
    
    stream_to_buffer(cmd)
end

-- Function to initialize the buffer
function M.init_buffer()
    display_llm_buffer()
    
    -- Clear the existing buffer content
    local buf = get_llm_buffer()
    vim.api.nvim_buf_set_lines(buf, 0, -1, false, {})
    
    -- Run the backend command
    run_backend_command(nil, "init")
    
    -- Wait a short time to ensure the backend has finished writing
    vim.defer_fn(function()
        local buffer = get_llm_buffer()
        
        -- Add a new line at the end of the buffer
        vim.api.nvim_buf_set_lines(buffer, -1, -1, false, {""})
        
        -- Move the cursor to the new empty line
        local win = vim.fn.bufwinid(buffer)
        if win ~= -1 then
            local line_count = vim.api.nvim_buf_line_count(buffer)
            vim.api.nvim_win_set_cursor(win, {line_count, 0})
        end
    end, 100)  -- 100ms delay, adjust if needed
end

-- Function to evaluate the buffer contents
function M.eval_buffer()
    if not llm_buffer_id or not vim.api.nvim_buf_is_valid(llm_buffer_id) then
        print("No valid LLM buffer. Initialize the buffer first.")
        return
    end

    local contents = table.concat(vim.api.nvim_buf_get_lines(llm_buffer_id, 0, -1, false), "\n")
    run_backend_command(contents, "evaluate")
end


-- Set up commands
vim.api.nvim_create_user_command('InitBuffer', M.init_buffer, {})
vim.api.nvim_create_user_command('EvalBuffer', M.eval_buffer, {})
vim.api.nvim_create_user_command('StopStreaming', M.stop_streaming, {})

return M

