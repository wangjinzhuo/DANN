require 'torch'
require 'nn'

--noutputs = 101
--model:add(nn.LogSoftMax())
criterion = nn.ClassNLLCriterion()


