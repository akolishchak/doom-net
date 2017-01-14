--
--  main.lua
--  doom-net
--
--  Doom network
--
--  Created by Andrey Kolishchak on 12/11/16.
--
require 'torch'
require 'cutorch'
require 'nn'
require 'nngraph'
require 'optim'
require 'paths'

torch.setdefaulttensortype('torch.FloatTensor')

--require('mobdebug').start()

cmd = torch.CmdLine()
cmd:text()
cmd:text('Doom Network')
cmd:text()
cmd:text('Options')
cmd:option('-learning_rate',        1e-4,   'learning rate')
cmd:option('-episode_size', 		10,     'number of steps in a episode')
cmd:option('-batch_size',           20,     'number of game instances running in parallel')
cmd:option('-sub_batch_num',        1,      'number of processings per batch')
cmd:option('-episode_num',          1000,   'number of episodes for training')
cmd:option('-episode_discount',     0.95,   'episode discount')
cmd:option('-gpu', 				    2,      '0 - cpu, 1 - cunn, 2 - cudnn')
cmd:option('-gpu_num', 			    1,      'number of GPUs')
cmd:option('-manual_seed',          1,      'seed value')
cmd:option('-model',                'models/aac.lua',      'path to model file, .t7 or .lua')
cmd:option('-vizdoom_config',     'environments/basic.cfg',   'vizdoom config path')
--cmd:option('-vizdoom_config',       'environments/rocket_basic.cfg',   'vizdoom config path')
--cmd:option('-vizdoom_config',       'environments/health_gathering.cfg',   'vizdoom config path')
--cmd:option('-vizdoom_config',       'environments/predict_position.cfg',   'vizdoom config path')
--cmd:option('-vizdoom_config',       'environments/defend_the_line.cfg',   'vizdoom config path')
--cmd:option('-vizdoom_config',       'environments/my_way_home.cfg',   'vizdoom config path')
cmd:option('-vizdoom_path',         paths.home .. '/test/ViZDoom/bin/vizdoom',   'path to vizdoom')
cmd:option('-wad_path',             paths.home .. '/test/ViZDoom/scenarios/Doom2.wad',   'wad file path')
--cmd:option('-wad_path',             paths.home .. '/test/ViZDoom/scenarios/freedoom2.wad',   'wad file path')
cmd:option('-skiprate',             1,      'number of skipped frames')


opt = cmd:parse(arg)

torch.manualSeed(opt.manual_seed)
if opt.gpu > 0 then cutorch.manualSeedAll(opt.manual_seed) end


env = paths.dofile('doom_env.lua')
env.init()

paths.dofile('model.lua')

paths.dofile('train.lua')
train()

paths.dofile('test.lua')
test()

env.release()
