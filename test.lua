require 'torch'
require 'xlua'
require 'optim'
require 'cunn'
require 'nn'
require 'image'
require 'generateInput'
require 'models/RCLayer'
require 'models/RCLMaxPooling'

function test()
    local time = sys.clock()
    local model = torch.load('./trained_models/B_5C_3fc_10000.t7')  -- specify the model to test
    print(model)
    print('Model loaded done ..')
    model:cuda()
	model:evaluate()

    classes = {}
    for i = 1, 101 do
        classes[i] = string.format("%d",i)
    end
    confusion = optim.ConfusionMatrix(classes)

    local correctClip = 0
    local correctVideo = 0
    local totalClip = 0
    local numVideo = 3784   --
    for vid = 1, numVideo do
        local td = sys.clock()
        -- get test data
	    local input = getTestClip()
        local target = getTestLabel()
        local num_clips = #input
        
        local scores = torch.zeros(101)
        local correct = 0  -- correct clips in the current vedio

        -- test each clip of the  vedio
        for i = 1, num_clips do
            local clip = input[i]:cuda()
            local pred = model:forward(clip)
            local ms, cls = pred:max(1)
            if cls[1] == target then
                correct = correct + 1
            end
            scores  = scores + pred:double()
		    confusion:add(pred, target)
        end
        local average_scores = scores / num_clips
        local max_score, idx = average_scores:max(1)
        print(string.format('The %d-th vedio, [%d] predicted with probability %f, [%d] expected', vid, idx[1], max_score[1], target))
        print(string.format('Correct Ratio for clips: %f', correct * 1.0 / num_clips))
        td = sys.clock() - td
        print(string.format('Total time: %f, average time: %f\n', td, td / num_clips))

        -- update
        if idx[1] == target then
            correctVideo = correctVideo + 1
        end
        totalClip = totalClip + num_clips
        correctClip = correctClip + correct
	end


    print(string.format('Accuracy Rate for Clip level: %f', correctClip * 1.0 / totalClip))
    print(string.format('Accuracy Rate for Video level: %f', correctVideo * 1.0 / numVideo))


	--timing
	time = sys.clock() - time
	time = time / numVideo

	print("\n ==> time to test 1 sample = " .. time .. 's')

    --[=[
	print(confusion)

	-- update log/plot
	testLogger:add{['% mean class accuracy (test set)'] = confusion.totalValid * 100}
	testLogger:style{['% mean class accuracy (test set)'] = '-'}
	testLogger:plot()
    ]=]

	-- next iteration
	confusion:zero()
end
init()
test()
