
local Tensor = opt.gpu > 0 and torch.CudaTensor or torch.Tensor
local actions = torch.eye(opt.button_num):int()

function test()
  
    local game = vizdoom.DoomGame()

    game:setViZDoomPath(opt.vizdoom_path)
    game:setDoomGamePath(opt.wad_path)
    game:loadConfig(opt.vizdoom_config)
    game:setWindowVisible(true)
    game:setSoundEnabled(true)
    game:setMode(vizdoom.Mode.ASYNC_PLAYER)
    game:init()
    game:newEpisode()

    function game_step(action)
        local reward = game:makeAction(actions[action[1]], opt.skiprate)
        local finished = game:isEpisodeFinished()
        if finished then
            --sys.sleep(1)
            print("Episode reward: ", game:getTotalReward())
            game:newEpisode()
        end
        if game:isPlayerDead() then
            game:respawnPlayer()
        end
            
        local state = game:getState()
        state.labels = nil
        return state, reward, finished
    end

    function game_step_normalized(action)
        local state, reward, finished = game_step(action)
        return normalize(state), reward, finished
    end

    function normalize(state)
        state.screen = state.screenBuffer:float():cuda()
        for channel = 1, state.screen:size(1) do
          state.screen[{{channel},{}}]:add(-opt.screen_mean[channel]):div(opt.screen_std[channel])
        end
        state.screen = state.screen:view(1, unpack(state.screen:size():totable()))
        if state.depthBuffer ~= nil then
            state.depth = state.depthBuffer:float():div(255):cuda()
        end
        if state.labelsBuffer ~= nil then
            state.labels = state.labelsBuffer:float():div(255):cuda()
        end
        if state.gameVariables then
            state.variables = state.gameVariables:float():div(100):cuda()
        end

        return state
    end

    function game_state_normalized()
        local state = game:getState()
        state.labels = nil
        
        return normalize(state)
    end
  
    local state = game_state_normalized()
  
    print("testing...")
    model:evaluate()
  
    while true do
        action = model:testForward(state)
        state = game_step_normalized(action[1])
        --sys.sleep(0.05)
    end
  
end