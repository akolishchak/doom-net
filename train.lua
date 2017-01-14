--
--  train.lua
--  doom-net
--
--  Doom network
--
--  Created by Andrey Kolishchak on 12/11/16.
--
local Tensor = opt.gpu > 0 and torch.CudaTensor or torch.Tensor

local episodes = {}
local episode_timer = torch.Timer()
local input_cpu = torch.Tensor()
local state = {}
local reward, action, episode_return
local params, grad_params = model:getParameters()
print(params:size(), grad_params:size())
local game_return = {}
local sub_batch_size = opt.batch_size / opt.sub_batch_num
local logger = optim.Logger(opt.model .. '.log')
logger:setNames{'Reward'}

function train()
    
    env.start()

    cutorch.synchronize()
    collectgarbage()
    
    state.screen = Tensor(opt.batch_size, unpack(opt.screen_size))
    state.depth = opt.depth_size and Tensor(opt.batch_size, unpack(opt.depth_size)) or nil
    state.labels = opt.labels_size and Tensor(opt.batch_size, unpack(opt.labels_size)) or nil
    state.variables = opt.variable_size and Tensor(opt.batch_size, opt.variable_size) or nil
    reward = Tensor(opt.batch_size, 1)
    action = Tensor(opt.batch_size, 1)
    episode_return = Tensor(opt.batch_size):zero()
    collectgarbage()
    
    -- get initial state
    get_state()
    
    model:training()
    print("trainig...")
    
    for episode = 1, opt.episode_num do
        
        function feval(x)
            if x ~= params then
                params:copy(x)
            end
            grad_params:zero()
            
            for sub_batch = 1, opt.sub_batch_num do
                
                local start_game, last_game = sub_batch_size*(sub_batch-1)+1, sub_batch_size*sub_batch
                
                local sub_state = {}
                for k,v in pairs(state) do
                    sub_state[k] = v ~= nil and v[{{start_game, last_game},{}}] or nil
                end
                local sub_reward = reward[{{start_game, last_game},{}}]
            
                for step = 1, opt.episode_size do
                    action[{{start_game, last_game},{}}]:copy(model:forward(sub_state))
                    game_step(start_game, last_game)
                    model:backward(sub_reward)
                end
                model:clearState()
                collectgarbage()
            end
            
            return -episode_return:mean(), grad_params
        end
        
        collectgarbage()
        episode_timer:reset()
        local _, loss = optim.adam(feval, params, optim_state)
        model:clearState()
        
        if episode % 1 == 0 then
          local mean_reward = -loss[1];
          print(string.format("episode = %d, reward = %.6f, time = %.3f", episode, mean_reward, episode_timer:time().real))
          --print(string.format("params_mean = %.6f, params_std = %.6f, params_min = %.6f, params_max = %.6f, grad_mean = %.6f, grad_std = %.6f, grad_min = %.6f, grad_max = %.6f", params:mean(), params:std(), params:min(), params:max(), grad_params:mean(), grad_params:std(), grad_params:min(), grad_params:max()))
          logger:add{mean_reward}
        end
    end
    
    torch.save(opt.model .. '.t7', model)
    torch.save(opt.model .. '.t7.optim.t7', optim_state)
end

function get_state()
    for game = 1, opt.batch_size do
        doom_threads:addjob(
            game,
            function()
                return tid, doom:state_normalized()
            end,                
            function(index, game_state)
                state.screen[{{index}, {}}]:copy(game_state.screenBuffer)
                if game_state.depthBuffer then
                    state.depth[{{index}, {}}]:copy(game_state.depthBuffer)
                end
                if game_state.labelsBuffer then
                    state.labels[{{index}, {}}]:copy(game_state.labelsBuffer)
                end
                if game_state.gameVariables then
                    state.variables[{{index}, {}}]:copy(game_state.gameVariables)
                end
            end
        )
    end
    doom_threads:synchronize()
end

function game_step(start_game, last_game)
    
    for game = start_game, last_game do
        doom_threads:addjob(
            game,
            
            function()
                local index, step_state, step_reward, game_finished = tid, doom:step_normalized(action[tid][1])
                state.screen[{{index}, {}}]:copy(step_state.screenBuffer)
                if step_state.depthBuffer then
                  state.depth[{{index}, {}}]:copy(step_state.depthBuffer)
                end
                if step_state.labelsBuffer then
                  state.labels[{{index}, {}}]:copy(step_state.labelsBuffer)
                end
                if step_state.gameVariables then
                    state.variables[{{index}, {}}]:copy(step_state.gameVariables)
                end
                reward[index][1] = step_reward
                if game_finished then
                    episode_return[index] = doom:get_return()
                end                
            end
        )
    end
    doom_threads:synchronize()
    cutorch.synchronize()
end

