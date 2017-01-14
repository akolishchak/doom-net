--
--  doom_instance.lua
--  doom-net
--
--  Doom network
--
--  Created by Andrey Kolishchak on 12/11/16.
--
package.path = package.path .. ";./vizdoom/?.lua"
vizdoom = require "vizdoom.init"

local doom_instance = torch.class('doom_instance')

function doom_instance:__init(opt)
    self.opt = opt
    self.game = self:init()
end

function doom_instance:init()
    local game = vizdoom.DoomGame()
    
    game:setViZDoomPath(self.opt.vizdoom_path)
    game:setDoomGamePath(self.opt.wad_path)
    game:loadConfig(self.opt.vizdoom_config)
    game:init()
    game:newEpisode()
    self.button_num = #game:getAvailableButtons()
    self.actions = torch.eye(self.button_num):int()
    self.episode_return = 0
    
    return game
end

function doom_instance:step(action)

    local reward = self.game:makeAction(self.actions[action], self.opt.skiprate)
    local finished = self.game:isEpisodeFinished()
    if finished then
        self.episode_return = self.game:getTotalReward()
        self.game:newEpisode()
    end
    if self.game:isPlayerDead() then
        self.game:respawnPlayer()
    end
        
    local state = self.game:getState()
    state.labels = nil
    
    return state, reward, finished
end

function doom_instance:step_normalized(action)
    local state, reward, finished = self:step(action)
    
    return self:normalize(state), reward, finished
end

function doom_instance:normalize(state)
    state.screenBuffer = state.screenBuffer:float()
    for channel = 1, state.screenBuffer:size(1) do
      state.screenBuffer[{{channel},{}}]:add(-self.opt.screen_mean[channel]):div(self.opt.screen_std[channel])
    end
    if state.depthBuffer ~= nil then
        state.depthBuffer = state.depthBuffer:float():div(255)
    end
    if state.labelsBuffer ~= nil then
        state.labelsBuffer = state.labelsBuffer:float():div(255)
    end
    if state.gameVariables ~= nil then
        state.gameVariables = state.gameVariables:float():div(100)
    end

    return state
end

function doom_instance:state()
    local state = self.game:getState()
    state.labels = nil
    
    return state
end

function doom_instance:state_normalized()
    local state = self:state()
    
    return self:normalize(state)
end

function doom_instance:stats()
    local stats = {}
    stats.screen_mean = torch.FloatTensor(opt.screen_size[1]):zero()
    stats.screen_std = torch.FloatTensor(opt.screen_size[1]):zero()
    
    local init_steps = 200
    for i = 1, init_steps do
        local state, _, _ = self:step(torch.random(self.button_num))
        local data = state.screenBuffer:float():view(state.screenBuffer:size(1), -1)
        stats.screen_mean:add(data:mean(2))
        stats.screen_std:add(data:std(2))
    end
    
    self.game:newEpisode()

    stats.screen_mean:div(init_steps)
    stats.screen_std:div(init_steps)
   
   return stats
end

function doom_instance:release()
    self.game:close()
end

function doom_instance:get_button_num()
    return self.button_num
end

function doom_instance:get_return()
    return self.episode_return
end    
