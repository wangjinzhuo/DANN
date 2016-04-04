require 'lfs'
require 'image'

function reset()
    local t = sys.clock()
    currVideo = currVideo + 1
    currPointNum = 1
    local videoPath = dataPath .. '/' .. string.sub(tr_content[shuffleVideo[currVideo]], 1, -5)
    if calculateFrames(videoPath) < 60 then
--        print(shuffleVideo[currVideo])
--        print('error: less than 60 frames')
        f_err = io.open('less_60.video', 'w+')
        f_err:write(shuffleVideo[currVideo])
        f_err:write('less than 60 frames \n')
        f_err:close()
        reset()
    end
    currVideoImages = getImagesGivenVideo()
--    print('currVideoImages Size: ', #currVideoImages)
    currPoints = randGenerateSamplingPoints(#currVideoImages)
    shufflePoint = torch.randperm(#currPoints)
--    print('currPoints Size: ', #currPoints)
--    print('reset Time: ', sys.clock() - t)
end

function calculateFrames(videoPath)
    local t = sys.clock()
    ret, files, iter = pcall(lfs.dir, videoPath)
    local frameNum = 0
    for f in files, iter do
        frameNum = frameNum + 1
    end
--    print('calculateFrames Time: ', sys.clock() - t)
    return frameNum - 2
end
function randGenerateSamplingPoints(frameNum)
    local t = sys.clock()
    local points = {} 
    local widthStart  = 30 
    local widthEnd = 320 - 30
    local heightStart = 30
    local heightEnd = 240 - 30
    local frameStart = 30
    local frameEnd = frameNum - 30 
    local frameStride = 4
    for frame = frameStart, frameEnd, frameStride do
        for ww = 1, 4 do
            x = math.random(widthStart, widthEnd)
            for hh = 1, 4 do
                y = math.random(heightStart, heightEnd)
                local point = {x, y, frame}       
                table.insert(points, point)
            end
        end
    end
   -- print('randGenerateSamplingPoint Time: ', sys.clock() - t)
    return points -- return all the sampling points given a video containing frameNum frames
end

function getOneDataGivenOneSamplingPoint()
    local t = sys.clock()
    local clip = torch.Tensor(3, 60, 58, 58)
    local index = 1
    for t = 1, 30 do
        local im = currVideoImages[currPoints[shufflePoint[currPointNum]][3] + t]
        clip[{ {}, {index}, {}, {} }] = im[{ {}, {currPoints[shufflePoint[currPointNum]][2] - 29, currPoints[shufflePoint[currPointNum]][2] + 29 - 1}, {currPoints[shufflePoint[currPointNum]][1] - 29, currPoints[shufflePoint[currPointNum]][1] + 29 - 1} }]
        index = index + 1
    end
    for t = 0, 29 do
        local im = currVideoImages[currPoints[shufflePoint[currPointNum]][3] - t]
        clip[{ {}, {index}, {}, {} }] = im[{ {}, {currPoints[shufflePoint[currPointNum]][2] - 29, currPoints[shufflePoint[currPointNum]][2] + 29 - 1}, {currPoints[shufflePoint[currPointNum]][1] - 29, currPoints[shufflePoint[currPointNum]][1] + 29 - 1} }]
        index = index + 1
    end
   -- print('GetOneDataGivenOneSamplingPoint Time: ', sys.clock() - t)
    return clip -- return one clip ((3, 60, 58, 58)tensor) given a sampling point(x, y, t)
end

function getOneData()     
    local t = sys.clock()
    while #currVideoImages < 60 do
        print(shuffleVideo[currVideo][1], '-th training data has less than 60 frames') 
        reset()
    end
    --print('Processing ', tr_content[shuffleVideo[currVideo]][1], '...')
    local clip = getOneDataGivenOneSamplingPoint()
    currPointNum = currPointNum + 1
    if currPointNum > math.min(100, #currPoints) then
        reset()
    --[[    while #currVideoImages < 60 do
            print(shuffleVideo[currVideo], '-th training data has less than 60 frames') 
            reset() 
        end
        --]]
        if currVideo > #tr_content then
            flag = true
        end
    end
  --  print('getOneData: ', sys.clock() - t) 
    return clip
end

function getImagesGivenVideo()
    local t = sys.clock()
    local images = {}
    local videoPath = dataPath .. '/' .. string.sub(tr_content[shuffleVideo[currVideo]][1], 1, -5)
    local n = calculateFrames(videoPath)
    for i = 0, n - 1 do
        local im = image.loadJPG(videoPath .. '/' .. 'image_' .. string.format('%04d', i) .. '.jpg')
        table.insert(images, im)
    end
--  print('getImagesGivenVideo: ', sys.clock() - t)
    return images
end

function getLabel()
    return tr_content[shuffleVideo[currVideo]][2]
end
function getTestLabel()
	return te_content[testNum][2]
end

function init()
    currVideo = 1
    shuffleVideo = torch.randperm(#tr_content)
    flag = false
    currVideoImages = getImagesGivenVideo()
    currPoints = randGenerateSamplingPoints(#currVideoImages)
    shufflePoint = torch.randperm(#currPoints)
    currPointNum = 1
end

testNum = 1
function getTestClips()
	local res = {}
	local videoPath = dataPath .. '/' .. string.sub(te_content[testNum][1], 1, -5)
	local frameNum = calculateFrames(videoPath)
	local testPoints = randGenerateSamplingPoints(frameNum)
	local num = math.min(400, #testPoints)
	local shuffle = torch.randperm(num) -- for video containing more than 400 clips
	local videoImages = function()
				local images = {}
				for i = 0, frameNum do
					local im = image.loadJPG(videoPath .. '/' .. 'image_' .. string.format('%04d', i) .. '.jpg')
					table.insert(images, im)
				end
				return images
			    end
	for i = 1, num  do
		local testClip = function()
			    local clip = torch.Tensor(3, 60, 58, 58)
			    local index = 1
			    for t = 1, 30 do
				local im = VideoImages[testPoints[shuffle[i]][3] + t]
				clip[{ {}, {index}, {}, {} }] = im[{ {}, {testPoints[shuffle[i]][2] - 29, testPoints[shuffle[i]][2] + 29 - 1}, {testPoints[shuffle[i]][1] - 29, testPoints[shuffle[i]][1] + 29 - 1} }]
				index = index + 1
			    end
			    for t = 0, 29 do
				local im = currVideoImages[testPoints[shuffle[i]][3] - t]
				clip[{ {}, {index}, {}, {} }] = im[{ {}, {testPoints[shuffle[i]][2] - 29, testPoints[shuffle[i]][2] + 29 - 1}, {testPoints[shuffle[i]][1] - 29, testPoints[shuffle[i]][1] + 29 - 1} }]
				index = index + 1
			    end
			    return clip -- return one clip ((3, 60, 58, 58)tensor) given a sampling point(x, y, t)
			end
		table.insert(res, testClip)
	end
	testNum = testNum + 1
	return res
end
trPath = '/media/cnn/My Passport/datasets/trainlist01.txt'
tePath = '/media/cnn/My Passport/datasets/testlist01.txt'
dataPath = '/media/cnn/My Passport/datasets/ucf_imgs'

tr = io.input(trPath)
tr_content = io.read('*a')
tr_content = tr_content:split('\n')
for i = 1, #tr_content do
    tr_content[i] = tr_content[i]:split(' ')
end

te = io.input(tePath)
te_content = io.read('*a')
te_content = te_content:split('\n')
for i = 1, #te_content do
	te_content[i] = te_content[i]:split(' ')
end

init()
-- test
--[=[
local t = sys.clock()
testNum = 10000
for i = 1, testNum do
--    print('Got ', shuffleVideo[currVideo], '-th training data, i.e., centered by point: ', currPoints[shufflePoint[currPointNum]])
    data = getOneData()
    label = getLabel()
 --   print('Got ', shuffleVideo[currVideo], '-th training data, i.e., centered by point: ', currPoints[shufflePoint[currPointNum]])
end
print(testNum, ' clip Time: ', sys.clock() - t)
]=]
