--
--  model.lua
--  doom-net
--
--  Doom network
--
--  Created by Andrey Kolishchak on 12/11/16.
--

if opt.gpu > 0 then
    require 'cunn'
    if opt.gpu == 2 then
        require 'cudnn'
        cudnn.benchmark = true
    end
end


if opt.model:match('%.lua$') then
    paths.dofile(opt.model)
    model = create_model()
    optim_state = {learningRate = opt.learning_rate}
else
    require (opt.model:match('^.*%.lua')) -- assumes that model name starts with model lua file path
    model = torch.load(opt.model)
    optim_state = torch.load(opt.model .. '.optim.t7')
end


collectgarbage()

