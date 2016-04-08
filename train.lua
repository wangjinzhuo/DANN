require 'torch'
require 'xlua'
require 'optim'
require 'cunn'
require 'cutorch'
require 'models/t4/3C_2CRcl_3fc'  -- 1 + 4 = 5, remember to modify saveModel. -- t6/1c-4CRcl out of memory
require 'loss'
require 'get_train_clip'

cutorch.setDevice(2)


savedModel = 't4/3C_2CRcl_3fc'  -- Model to save
batchsize = 50 -- 1 = pure stochastic
learningRate = 1e-3 
weightDecay = 0 -- SGD only
momentum = 0.9 -- SGD only

print(model)
print(criterion)

model:cuda()
criterion:cuda()


classes = {}
for i = 1, 101 do
	classes[i] = string.format("%d",i)
end

confusion = optim.ConfusionMatrix(classes)
if model then
	parameters, gradParameters = model:getParameters()
end

-- configuring optimizer
optimState = {
	learningRate = learningRate,
	weightDecay = weightDecay,
	momentum = momentum,
	learningRateDecay = 1e-7
}
optimMethod = optim.sgd


function train()
	epoch = epoch or 1
    batchId = batchId or 1
	local time = sys.clock()
    td = td or sys.clock()
	model:training() -- set model to training model (differing from  testing, like Dropout)
  
    -- 1: trainlist01.txt
    train_video_num = get_train_video_num(1)
    print('train video num= ', train_video_num)
    train_points = get_train_points(1)
    -- Conducting 7500 batches with 50 batch-size 
    for batch = 1, 8000 do		
        --create mini batch
		local inputs = {}
		local targets = {}
		for i = 1, batchsize do
			--load new shuffle sample
            local shuffle_video = torch.randperm(train_video_num)
            if #train_points[shuffle_video[1]] > 60 then
                local shuffle_point = torch.randperm(#train_points[shuffle_video[1]])
                local input = get_train_clip(1, shuffle_video[1], train_points[shuffle_video[1]][shuffle_point[1]])
                local target = tonumber(get_train_label(1, shuffle_video[1]))
                input = input:cuda()
                table.insert(inputs, input)
                table.insert(targets, target)
            else
                local err_file = io.open('less_60_video.txt', 'w+')
                err_file:write(shuffle_video[1], ' has ', tostring(#train_points[shuffle_video[1]]), ' frames \n')
                err_file:close()
            end
		end
		local feval = function(x)
				-- get new parameters
				if x ~= parameters then
					parameters:copy(x)
				end
				gradParameters:zero()
				local f = 0 -- f is the average of all criterions
				for i = 1, #inputs do
					-- estimate f
					local output = model:forward(inputs[i])
					local err = criterion:forward(output, targets[i])
					f = f + err
					
					-- estimate df/dW
					local df_do = criterion:backward(output, targets[i])
					model:backward(inputs[i], df_do)

					--update confusion
					confusion:add(output, targets[i])
				end
				-- normalize gradients and f(X)
				gradParameters:div(#inputs)
				f = f / #inputs

				-- return f and df/dX
				return f, gradParameters
		end
		-- optimize on current mini-batch
		_, average_loss= optimMethod(feval, parameters, optimState)
        if batchId == 1 or batchId % 100 == 0 then
            print(savedModel,': ', string.format('The %dth batch loss: %f, Time cost: %f', batchId, average_loss[1], sys.clock() - td))
            td = sys.clock()
        end
        if batchId % 1000 == 0 then
            torch.save(string.format('trained_models/%s_%d.t7', savedModel, batchId), model, 'binary')
        end
        batchId = batchId + 1
	end
	time = sys.clock() - time
    print(string.format('Epoch %d, total time: %f', epoch, time))
    --print(confusion)

	-- next epoch
	confusion:zero()
	epoch = epoch + 1
end

while true do
    train()
end
