--
--  aac.lua
--  doom-net
--
--  Doom network
--  Advantage Actor-Critic Model
--
--  Created by Andrey Kolishchak on 12/11/16.
--
require 'models.MultinomialAction'
--
-- This is not regular nn.Module and cannot be chained with other modules.
-- This class is inherited from nn.Module to support torch save/load of full class state.
--
require 'nn'
local Model, parent = torch.class('nn.AACModel', 'nn.Module')

function Model:__init(opt)
    parent.__init(self)
    
    self.opt = opt
    
    self.conv, conv_size = self:createConvnet({
        {opt.screen_size[1], 32, 3, 1, 0, 1},
        {32, 64, 3, 1, 0, 2}, 
        {64, 64, 3, 1, 0, 2},
        {64, 64, 3, 2, 0, 2}
    })

    local features = 64

    self.shared = nn.Sequential()
    self.shared:add(nn.Linear(conv_size, features))
    self.shared:add(nn.BatchNormalization(features))
    self.shared:add(nn.ReLU(true))
    
    self.act = nn.Sequential()
    self.act:add(nn.Linear(features, opt.button_num))
    self.act:add(nn.SoftMax())
    self.act:add(nn.MultinomialAction())
    
    self.pred = nn.Sequential()
    self.pred:add(nn.Linear(features, 1))

    local model = nn.Sequential()
    model:add(self.conv)
    model:add(self.shared)
    model:add(nn.ConcatTable()
                :add(self.act)
                :add(self.pred)
            )
                
    self.criterion = nn.MSECriterion()
                    
    if opt.gpu > 0 then
        model:cuda()
        self.criterion:cuda()
        cudnn.convert(model, cudnn)
        --cudnn.convert(self.criterion, cudnn)
        cudnn.benchmark = true
    end
    
    self.model_clones = {}
    for i = 1,opt.episode_size do
        self.model_clones[i] = model:clone('weight','bias','gradWeight','gradBias', 'running_mean', 'running_std', 'running_var')
    end
    
    self.reward_loss = opt.gpu > 0 and torch.CudaTensor() or torch.Tensor()
    self.ret = opt.gpu > 0 and torch.CudaTensor() or torch.Tensor()
    
    self:clearState()
    collectgarbage()
end

function Model:createConvnet(params)
    local width = self.opt.screen_size[2]
    local height = self.opt.screen_size[3]
    local channel

    print(width, height)
    local conv = nn.Sequential()
    for _,layer in pairs(params) do
        local iC, oC, k, s, pad, pool = layer[1], layer[2], layer[3], layer[4], layer[5], layer[6]
        conv:add(nn.SpatialConvolution(iC, oC, k, k, s, s, pad, pad))
        --conv:add(nn.SpatialBatchNormalization(oC, 1e-3))
        conv:add(nn.ReLU(true))
        if pool > 1 then conv:add(nn.SpatialMaxPooling(pool, pool, pool, pool)) end

        width = torch.floor(((width  + 2*pad - k) / s + 1)/pool)
        height = torch.floor(((height + 2*pad - k) / s + 1)/pool)
        channel = oC
    end

    print(width, height)
    local size = width*height*channel
    conv:add(nn.Reshape(size))

    return conv, size
end

function Model:forward(state)
    self.step = self.step + 1
    assert(self.step <= self.opt.episode_size, "number of steps exceeds episode size")
    
    self.input[self.step] = state.screen:clone()
    local output = self.model_clones[self.step]:forward(self.input[self.step])
    self.baseline[self.step] = output[2]
    return output[1]
end

function Model:testForward(state)
    local output = self.model_clones[1]:forward(state.screen)
    return output[1]
end

function Model:backward(reward)
    self.reward[self.step] = reward:clone():mul(1e-2)
    if self.step < self.opt.episode_size then return end
    
    self.ret:resizeAs(reward):copy(self.baseline[self.step])
    for t = self.step, 1, -1 do
        self.ret:mul(self.opt.episode_discount):add(self.reward[t])
        self.reward_loss:resizeAs(self.ret):copy(self.ret)
        
        local loss = self.criterion:forward(self.baseline[t], self.reward_loss)
        local dbaseline = self.criterion:backward(self.baseline[t], self.reward_loss)
        
        self.reward_loss:add(-self.baseline[t]):mul(-1)
        self.model_clones[t]:backward(self.input[t], {self.reward_loss, dbaseline})
        
        self.model_clones[t]:clearState()
    end
end

function Model:clearState()
    self.reward = {}
    self.input = {}
    self.baseline = {}
    self.step = 0
    self.ret:set()
    self.reward_loss:set()
end

function Model:getParameters()
    -- https://github.com/torch/nn/issues/897
    local container = nn.Container()
    for i = 1, #self.model_clones do container:add(self.model_clones[i]) end
    return container:getParameters()
end

function Model:training()
    for i = 1, #self.model_clones do self.model_clones[i]:training() end
end

function Model:evaluate()
    for i = 1, #self.model_clones do self.model_clones[i]:evaluate() end
end

function Model:__tostring__()
    local name = torch.type(self)
    return string.format('%s:\n%s', name, self.model_clones[1])
end

function create_model()
    return nn.AACModel(opt)
end