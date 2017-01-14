--
--  doom_env.lua
--  doom-net
--
--  Doom network
--
--  Created by Andrey Kolishchak on 12/11/16.
--
require 'doom_instance'

local threads = require 'threads'
threads.serialization('threads.sharedserialize')


local doom_env = {}
local options = opt
doom_threads = nil
local cache_file_name = opt.vizdoom_config .. '.cache'

function doom_env.init()
    local game = doom_instance(opt)
    local state = game:state()

    opt.button_num = game:get_button_num()
    opt.screen_size = state.screenBuffer:size():totable()
    opt.depth_size = state.depthBuffer and state.depthBuffer:size():totable() or nil
    opt.labels_size = state.labelsBuffer and state.labelsBuffer:size():totable() or nil
    opt.variable_size = state.gameVariables and state.gameVariables:size(1) or nil

    if paths.filep(cache_file_name) then
        local cache = torch.load(cache_file_name)
        opt.screen_mean = cache.screen_mean
        opt.screen_std = cache.screen_std
        print("loaded cached environment statistics from ", cache_file_name)
    end
    
    game:release()
end

function doom_env.start()
    
    opt.screen_mean = torch.Tensor(opt.screen_size[1]):zero()
    opt.screen_std = torch.Tensor(opt.screen_size[1]):zero()
    
    print("start game instances...")

    doom_threads = threads.Threads(
        options.batch_size,
        
        function()
            require 'cutorch'
            require 'sys'
        end,
        function()
            package.path = package.path .. ";./vizdoom/?.lua"
            vizdoom = require "vizdoom.init"
            torch.setdefaulttensortype('torch.FloatTensor')
        end,
        
        function(idx)
            print('start game thread:', idx)
            opt = options -- upvalue
            action_num = button_num
            tid = idx
            torch.manualSeed(opt.manual_seed)
            if opt.gpu > 0 then cutorch.manualSeedAll(opt.manual_seed) end
            
            require 'doom_instance'
            doom = doom_instance(opt)
        end
    )

    doom_threads:specific(true)
    
    for game = 1, options.batch_size do
        doom_threads:addjob(
            game,
            
            function()
                return doom:stats()
            end,
            
            function(stats)
                opt.screen_mean:add(stats.screen_mean)
                opt.screen_std:add(stats.screen_std)
            end
        )
    end

    doom_threads:synchronize()
    opt.screen_mean:div(opt.batch_size)
    opt.screen_std:div(opt.batch_size)
    
    local cache = {}
    cache.screen_mean = opt.screen_mean
    cache.screen_std = opt.screen_std
    torch.save(cache_file_name, cache)
end

function doom_env.release()
    if doom_threads ~= nil then
        doom_threads:terminate()
    end
end

return doom_env